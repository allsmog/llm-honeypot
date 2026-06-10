# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Per-session virtual filesystem for ls / stat / find coherence.
# ABOUTME: A real shell never contradicts itself about what's on disk:
# ABOUTME: `touch /tmp/x; ls /tmp` shows x, a second `ls /tmp` is identical,
# ABOUTME: and `stat /tmp/x` agrees with the `ls -l` size. The LLM, asked
# ABOUTME: cold each turn, drifts. This module renders those commands from
# ABOUTME: a fixed base skeleton overlaid with the session's real WorldState
# ABOUTME: files, so the picture is deterministic and self-consistent.

from __future__ import annotations

import stat as statmod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cowrie.llm.worldstate import WorldState


@dataclass(frozen=True)
class Node:
    name: str
    kind: str  # "file" | "dir" | "link"
    size: int = 4096
    mode: int = 0o644
    uid: int = 0
    gid: int = 0
    link_target: str = ""  # for kind == "link"
    mtime: float = 0.0


# Canonical directory skeleton. Keys are absolute dir paths; values are the
# entries directly inside them. Intentionally small and boring — a plausible,
# lightly-used VPS, nothing that screams "lab". Dirs an attacker commonly
# lists are covered; anything else falls through to the LLM (no regression).
def _base_skeleton(home: str, username: str) -> dict[str, list[Node]]:
    DIR = 0o755
    uid = 0 if username == "root" else 1000
    skel: dict[str, list[Node]] = {
        "/": [
            Node("bin", "link", 7, 0o777, link_target="usr/bin"),
            Node("boot", "dir", 4096, DIR),
            Node("dev", "dir", 4096, DIR),
            Node("etc", "dir", 4096, DIR),
            Node("home", "dir", 4096, DIR),
            Node("lib", "link", 7, 0o777, link_target="usr/lib"),
            Node("media", "dir", 4096, DIR),
            Node("mnt", "dir", 4096, DIR),
            Node("opt", "dir", 4096, DIR),
            Node("proc", "dir", 0, 0o555),
            Node("root", "dir", 4096, 0o700),
            Node("run", "dir", 4096, DIR),
            Node("sbin", "link", 8, 0o777, link_target="usr/sbin"),
            Node("srv", "dir", 4096, DIR),
            Node("sys", "dir", 0, 0o555),
            Node("tmp", "dir", 4096, 0o1777),
            Node("usr", "dir", 4096, DIR),
            Node("var", "dir", 4096, DIR),
        ],
        # /home holds the non-root login user (if any) — never "root".
        "/home": (
            [] if username == "root"
            else [Node(username, "dir", 4096, 0o755, uid=1000, gid=1000)]
        ),
        # The session user's home directory, owned by that user.
        home: [
            Node(".bashrc", "file", 3771 if uid else 3526, 0o644, uid=uid, gid=uid),
            Node(".profile", "file", 807 if uid else 161, 0o644, uid=uid, gid=uid),
            Node(".bash_logout", "file", 220, 0o644, uid=uid, gid=uid),
            Node(".cache", "dir", 4096, 0o700, uid=uid, gid=uid),
            Node(".ssh", "dir", 4096, 0o700, uid=uid, gid=uid),
        ],
        "/tmp": [],
        "/var/tmp": [],
        "/dev/shm": [],
        "/etc": [
            Node("hostname", "file", 12, 0o644),
            Node("hosts", "file", 221, 0o644),
            Node("os-release", "link", 21, 0o777, link_target="../usr/lib/os-release"),
            Node("passwd", "file", 1684, 0o644),
            Node("group", "file", 803, 0o644),
            Node("shadow", "file", 1043, 0o640, gid=42),
            Node("resolv.conf", "link", 39, 0o777,
                 link_target="../run/systemd/resolve/stub-resolv.conf"),
            Node("crontab", "file", 1042, 0o644),
            Node("fstab", "file", 657, 0o644),
            Node("ssh", "dir", 4096, 0o755),
            Node("cron.d", "dir", 4096, 0o755),
            Node("systemd", "dir", 4096, 0o755),
        ],
    }
    return skel


# Directories whose listing we can answer deterministically. A bare `ls`
# of one of these (or one holding session-created files) renders locally;
# anything else returns None and the LLM narrates it.
def _home_for(username: str) -> str:
    return "/root" if username == "root" else f"/home/{username}"


class VFS:
    """A read-only view combining the base skeleton with WorldState files.

    Constructed per render from (world, login_user). Cheap — it's just a
    dict build — so we don't cache it on the session.
    """

    def __init__(self, world: WorldState, username: str) -> None:
        self.username = username
        self.home = _home_for(username)
        self.skeleton = _base_skeleton(self.home, username)
        self.world = world

    # -- directory listing -------------------------------------------------

    def _dir_entries(self, path: str) -> list[Node] | None:
        """Entries directly inside ``path``, or None if we don't model it."""
        path = _normpath(path)
        base = self.skeleton.get(path)
        # Overlay WorldState files whose parent is this dir.
        overlay: list[Node] = []
        seen = {n.name for n in base} if base is not None else set()
        for fpath, fact in self.world.files.items():
            parent, name = _split(fpath)
            if parent == path and name not in seen:
                uid = 0 if self.username == "root" else 1000
                overlay.append(Node(
                    name=name, kind="file", size=fact.size_bytes,
                    mode=0o644, uid=uid, gid=uid, mtime=fact.mtime,
                ))
                seen.add(name)
        if base is None and not overlay:
            return None
        return list(base or []) + overlay

    def is_file(self, path: str) -> Node | None:
        """Return the Node for ``path`` if it's a known file/link, else None."""
        path = _normpath(path)
        parent, name = _split(path)
        if path in self.world.files:
            fact = self.world.files[path]
            uid = 0 if self.username == "root" else 1000
            return Node(name=name, kind="file", size=fact.size_bytes,
                        mode=0o644, uid=uid, gid=uid, mtime=fact.mtime)
        for node in self.skeleton.get(parent, []):
            if node.name == name and node.kind in ("file", "link"):
                return node
        return None

    def is_dir(self, path: str) -> bool:
        path = _normpath(path)
        if path in self.skeleton:
            return True
        parent, name = _split(path)
        return any(n.name == name and n.kind == "dir"
                   for n in self.skeleton.get(parent, []))


# ----------------------------------------------------------------------
# Path helpers


def _normpath(path: str) -> str:
    if not path:
        return "/"
    # Collapse redundant slashes and trailing slash (keep root).
    parts = [p for p in path.split("/") if p not in ("", ".")]
    out: list[str] = []
    for p in parts:
        if p == "..":
            if out:
                out.pop()
        else:
            out.append(p)
    return "/" + "/".join(out)


def _split(path: str) -> tuple[str, str]:
    path = _normpath(path)
    if path == "/":
        return "/", ""
    parent, _, name = path.rpartition("/")
    return parent or "/", name


# ----------------------------------------------------------------------
# Rendering


def _mode_string(node: Node) -> str:
    if node.kind == "dir":
        first = "d"
    elif node.kind == "link":
        first = "l"
    else:
        first = "-"
    perms = ""
    for who in (6, 3, 0):
        bits = (node.mode >> who) & 0o7
        perms += "r" if bits & 0o4 else "-"
        perms += "w" if bits & 0o2 else "-"
        perms += "x" if bits & 0o1 else "-"
    # sticky bit on /tmp
    if node.mode & statmod.S_ISVTX and node.kind == "dir":
        perms = perms[:-1] + ("t" if perms[-1] == "x" else "T")
    return first + perms


def _owner(uid: int, username: str = "") -> str:
    if uid == 0:
        return "root"
    if uid == 1000:
        return username if username and username != "root" else "ubuntu"
    if uid == 42:
        return "shadow"
    return str(uid)


def _fmt_time(mtime: float) -> str:
    if not mtime:
        # Stable, plausible date for skeleton entries.
        return "Mar 14  2024"
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return dt.strftime("%b %e %H:%M")


def render_ls(vfs: VFS, path: str, *, long: bool, all_: bool,
              username: str) -> str | None:
    """Render `ls [path]`. Returns None to defer to the LLM."""
    path = _normpath(path)

    # `ls <file>` just echoes the path (or errors). Handle file targets.
    node = vfs.is_file(path)
    if node is not None and not vfs.is_dir(path):
        if long:
            return _ls_long_line(node, username, name_override=path) + "\n"
        return path + "\n"

    entries = vfs._dir_entries(path)
    if entries is None:
        return None  # unknown dir — let the LLM narrate

    items = list(entries)
    if not all_:
        # Hide dotfiles unless -a/-A (real ls behavior).
        items = [n for n in items if not n.name.startswith(".")]
    if all_:
        # `.` is owned by whoever owns the listed dir (the user, for ~).
        dir_uid = 1000 if path == vfs.home and username != "root" else 0
        dmode = 0o1777 if path == "/tmp" else (0o700 if path == vfs.home else 0o755)
        dot = Node(".", "dir", 4096, dmode, uid=dir_uid, gid=dir_uid)
        dotdot = Node("..", "dir", 4096, 0o755)
        items = [dot, dotdot, *items]
    items.sort(key=lambda n: n.name.lstrip("."))

    if not long:
        names = [n.name for n in items]
        return ("  ".join(names) + "\n") if names else ""

    lines = []
    if items:
        total = sum(max(1, n.size // 1024) for n in items)
        lines.append(f"total {total}")
    for n in items:
        lines.append(_ls_long_line(n, username))
    return "\n".join(lines) + "\n"


def _ls_long_line(node: Node, username: str, *, name_override: str = "") -> str:
    owner = _owner(node.uid, username)
    grp = _owner(node.gid, username)
    nlink = 2 if node.kind == "dir" else 1
    name = name_override or node.name
    suffix = ""
    if node.kind == "link":
        suffix = f" -> {node.link_target}"
    return (
        f"{_mode_string(node)} {nlink:>2} {owner:<8} {grp:<8} "
        f"{node.size:>6} {_fmt_time(node.mtime)} {name}{suffix}"
    )


def render_stat(vfs: VFS, path: str, *, username: str) -> str | None:
    """Render `stat <path>`. Returns None to defer."""
    path = _normpath(path)
    node = vfs.is_file(path)
    is_dir = vfs.is_dir(path)
    if node is None and not is_dir:
        # Unknown path: a clear, correct error is better than LLM guessing.
        return (
            f"stat: cannot statx '{path}': No such file or directory\n"
        )
    if node is None and is_dir:
        node = Node(_split(path)[1] or "/", "dir", 4096, 0o755)
    assert node is not None
    kind = {"dir": "directory", "link": "symbolic link"}.get(node.kind, "regular file")
    if node.kind == "file" and node.size == 0:
        kind = "regular empty file"
    octal = f"{node.mode & 0o7777:04o}"
    perms = _mode_string(node)
    blocks = (node.size + 511) // 512
    mt = _fmt_full_time(node.mtime)
    return (
        f"  File: {path}" + (f" -> {node.link_target}" if node.kind == "link" else "") + "\n"
        f"  Size: {node.size:<15} Blocks: {blocks:<10} IO Block: 4096   {kind}\n"
        f"Device: fd01h/64769d\tInode: {_fake_inode(path):<11} Links: "
        f"{2 if node.kind == 'dir' else 1}\n"
        f"Access: ({octal}/{perms})  Uid: ({node.uid:>5}/{_owner(node.uid, username):>8})   "
        f"Gid: ({node.gid:>5}/{_owner(node.gid, username):>8})\n"
        f"Access: {mt}\nModify: {mt}\nChange: {mt}\n Birth: -\n"
    )


def _fake_inode(path: str) -> int:
    return 130000 + (abs(hash(path)) % 700000)


def _fmt_full_time(mtime: float) -> str:
    if not mtime:
        return "2024-03-14 09:12:33.000000000 +0000"
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.000000000 +0000")
