# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Canonicalizes commands that must agree with each other into a
# ABOUTME: single fact key — `uname -a`, `uname -r` and `cat /proc/version`
# ABOUTME: all assert the kernel, so answering one commits us to the
# ABOUTME: others. Used to record what we have already told a session so a
# ABOUTME: re-probe replays it instead of re-inventing it. Pure, no I/O.

"""Fact families for consistency tracking.

An attacker probing for a honeypot asks the same thing twice and compares.
Our own FINGERPRINT_PROBE adversary does exactly that. The defence is to
remember what we said the first time — but "what we said" has to be keyed
by *fact*, not by command string, or `uname -a` and `uname -r` look like
unrelated questions when they are two views of one answer.

Deliberately conservative: an unrecognized command returns None and is not
tracked at all. A wrong key would make two unrelated facts overwrite each
other, which is worse than tracking nothing.
"""

from __future__ import annotations

import re
import shlex

#: (family, pattern) in priority order. First match wins, so more specific
#: patterns must come first.
_FAMILY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("os.kernel", re.compile(r"\buname\b|/proc/version|\bhostnamectl\b")),
    (
        "os.distro",
        re.compile(
            r"/etc/(os-release|issue|lsb-release|debian_version|redhat-release"
            r"|centos-release|system-release|alpine-release)|\blsb_release\b"
        ),
    ),
    ("hw.cpu", re.compile(r"/proc/cpuinfo|\blscpu\b|\bnproc\b")),
    ("hw.mem", re.compile(r"\bfree\b|/proc/meminfo")),
    ("sys.uptime", re.compile(r"\buptime\b|/proc/uptime")),
    ("id.accounts", re.compile(r"/etc/(passwd|shadow|group)|\bgetent\b")),
    ("id.user", re.compile(r"\bwhoami\b|\bid\b|\blogname\b|\bgroups\b")),
    ("net.iface", re.compile(r"\bifconfig\b|\bip\s+(a|addr|link)\b|\bhostname\s+-I\b")),
    ("net.listen", re.compile(r"\bss\b|\bnetstat\b")),
    ("proc.list", re.compile(r"\bps\b|\btop\b|\bhtop\b")),
    ("pkg.list", re.compile(r"\bdpkg\b|\brpm\s+-qa\b|\bapk\s+info\b|\bapt\s+list\b")),
    ("fs.disk", re.compile(r"\bdf\b|\bmount\b|/proc/mounts")),
    ("cron.jobs", re.compile(r"\bcrontab\b|/etc/crontab|/etc/cron\.")),
)

#: `uname -m` and `arch` are the architecture, not the kernel release —
#: checked before the kernel rule so `uname -m` does not collide with
#: `uname -r`.
_ARCH_RE = re.compile(r"^\s*(arch|uname\s+-\w*m\w*|dpkg\s+--print-architecture)\s*$")

#: Commands whose answer is *expected* to change between calls. Tracking
#: them would generate contradictions out of correct behaviour.
#: The \b applies to the whole group — without it, the `w` alternative
#: swallows `whoami`, `wc`, `which` and anything else starting with w.
_VOLATILE_RE = re.compile(r"^\s*(date|w|top)\b")

#: Reading a specific file commits us to that file's content, keyed per
#: path so `cat /tmp/a` and `cat /tmp/b` stay independent.
_FILE_READERS = ("cat", "head", "tail", "less", "more", "stat", "file", "wc")


def fact_family(command: str) -> str | None:
    """The fact ``command`` asserts, or None if we do not track it.

    Never raises: this runs on attacker-controlled input on the response
    path, and a crash here would cost the attacker their prompt.
    """
    try:
        return _fact_family(command)
    except Exception:
        return None


def _fact_family(command: str) -> str | None:
    raw = (command or "").strip()
    if not raw:
        return None

    # Only the first stage of a pipeline determines what is being asked;
    # `free -m | head -2` still asserts the memory figures.
    head = raw.split("|", 1)[0].strip()
    if not head:
        return None

    if _VOLATILE_RE.match(head):
        return None
    if _ARCH_RE.match(head):
        return "os.arch"

    try:
        argv = shlex.split(head)
    except ValueError:
        argv = head.split()
    if not argv:
        return None

    # A per-path key for file reads, so two different files never collide.
    # Checked before the generic rules because `cat /etc/os-release` should
    # key as the distro, not as an arbitrary file — hence the family scan
    # happening first for paths the rules recognize.
    for family, pattern in _FAMILY_RULES:
        if pattern.search(head):
            return family

    if argv[0] in _FILE_READERS:
        paths = [a for a in argv[1:] if not a.startswith("-")]
        if len(paths) == 1:
            return f"file:{paths[0]}"
    return None
