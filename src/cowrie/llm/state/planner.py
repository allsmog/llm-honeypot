# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Validate parsed mutation intents against the world before any
# ABOUTME: of them are applied. Returns either "commit these" or "refuse,
# ABOUTME: with this stderr and this exit code" — never a partial world.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cowrie.llm.state import permissions as perms

if TYPE_CHECKING:
    from cowrie.llm.cmd_parser import CmdMutation
    from cowrie.llm.vfs import VFS


@dataclass(frozen=True)
class Plan:
    """The outcome of validating one command's intents.

    Either ``mutations`` are safe to commit, or ``exit_code`` is nonzero
    and ``stderr`` says why — in which case nothing is committed at all.
    The all-or-nothing shape is deliberate: a command that half-applies is
    worse than one that fails, because no shell behaves that way.
    """

    mutations: tuple[CmdMutation, ...] = ()
    stderr: str = ""
    exit_code: int = 0
    # True when the refusal is one we modelled, so the caller can render it
    # deterministically instead of asking the model to invent an error.
    refused: bool = False


def _parent(path: str) -> str:
    trimmed = path.rstrip("/")
    parent, _, _ = trimmed.rpartition("/")
    return parent or "/"


def _deny(message: str, code: int = 1) -> Plan:
    return Plan(stderr=message, exit_code=code, refused=True)


def _check_writable_parent(vfs: VFS, path: str, user: str, verb: str) -> Plan | None:
    """Shared precondition for anything that creates a directory entry."""
    parent = _parent(path)
    status = vfs.path_status(parent)
    if status == "absent":
        return _deny(f"{verb}: cannot create '{path}': No such file or directory\n")
    if status == "file":
        return _deny(f"{verb}: cannot create '{path}': Not a directory\n")
    if perms.can_write(vfs, parent, user) is False:
        return _deny(f"{verb}: cannot create '{path}': Permission denied\n")
    return None


def plan_mutations(
    mutations: list[CmdMutation],
    vfs: VFS,
    user: str,
    login_user: str,
) -> Plan:
    """Validate ``mutations`` as one transaction.

    ``user`` is the *effective* user (top of the su stack), which is the
    one the kernel would check. Unmodelled paths are permitted throughout:
    refusing something we never described would invent a restriction that
    no listing of ours supports, and inventing restrictions is as
    detectable as inventing permissions.
    """
    if not mutations:
        return Plan()

    for m in mutations:
        denial = _validate_one(m, vfs, user, login_user)
        if denial is not None:
            return denial

    return Plan(mutations=tuple(mutations))


def _validate_one(
    m: CmdMutation, vfs: VFS, user: str, login_user: str
) -> Plan | None:
    path = m.path or ""

    if m.kind in ("create_file", "append_file"):
        def fail(reason: str) -> Plan:
            # `touch` and a `>` redirect both create a file and report
            # failure completely differently. The redirect failure comes
            # from the shell itself, which is why it is prefixed "bash:"
            # and names no command.
            if m.verb == "touch":
                return _deny(f"touch: cannot touch '{path}': {reason}\n")
            return _deny(f"bash: {path}: {reason}\n")

        # Writing over an existing file needs write on the file itself;
        # creating a new one needs write on the directory.
        if vfs.path_status(path) == "file":
            if perms.can_write(vfs, path, user) is False:
                return fail("Permission denied")
            return None
        parent = _parent(path)
        status = vfs.path_status(parent)
        if status == "absent":
            return fail("No such file or directory")
        if status == "file":
            return fail("Not a directory")
        if perms.can_write(vfs, parent, user) is False:
            return fail("Permission denied")
        return None

    if m.kind == "remove_file":
        parent = _parent(path)
        if perms.can_write(vfs, parent, user) is False:
            return _deny(f"rm: cannot remove '{path}': Permission denied\n")
        return None

    if m.kind in ("copy_file", "move_file"):
        verb = "cp" if m.kind == "copy_file" else "mv"
        if vfs.path_status(path) == "absent":
            return _deny(
                f"{verb}: cannot stat '{path}': No such file or directory\n"
            )
        if m.dst_path:
            denial = _check_writable_parent(vfs, m.dst_path, user, verb)
            if denial is not None:
                return denial
        return None

    if m.kind == "push_user":
        target = m.user or ""
        if target and not perms.user_exists(target, login_user):
            # su's own wording. Getting this wrong is itself a tell, so it
            # matches the real message rather than a generic refusal.
            return _deny(f"su: user {target} does not exist\n")
        return None

    # set_env, pop_user, add_process have no precondition we model.
    return None
