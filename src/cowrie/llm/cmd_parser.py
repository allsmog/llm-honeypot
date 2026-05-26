# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Parse attacker INPUT (not LLM output) for filesystem +
# ABOUTME: environment mutations we can mirror into WorldState. Input
# ABOUTME: parsing is bounded — we handle the 80% common cases; the rest
# ABOUTME: falls through and the LLM narrates inconsistently (same as
# ABOUTME: today, no regression). Output parsing of LLM responses is
# ABOUTME: deliberately not attempted — too fragile.

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Literal, Optional

MutationKind = Literal[
    "create_file",
    "append_file",
    "remove_file",
    "move_file",
    "copy_file",
    "set_env",
]


@dataclass
class CmdMutation:
    kind: MutationKind
    path: Optional[str] = None
    dst_path: Optional[str] = None  # for cp/mv
    content: Optional[str] = None   # for echo/touch
    env_name: Optional[str] = None
    env_value: Optional[str] = None


# ----------------------------------------------------------------------
# Pattern matchers


# `echo <stuff> > <path>` or `>> <path>`. Captures the rhs (content) and
# the destination path. Tolerates quoted content via shlex elsewhere;
# the regex here just detects the redirect operator.
_ECHO_REDIRECT_RE = re.compile(
    r"^\s*echo\s+(?P<content>.*?)\s*(?P<op>>>?)\s*(?P<path>\S+)\s*$"
)

# `touch <path>` (one or more paths; we capture all). Allow flags before
# the path list but ignore them.
_TOUCH_RE = re.compile(r"^\s*touch\s+(?P<args>.*)$")

# `rm [flags] <path>` (one or more). Same shape as touch.
_RM_RE = re.compile(r"^\s*rm\s+(?P<args>.*)$")

# `cp [flags] <src> <dst>` and `mv [flags] <src> <dst>`.
_CP_RE = re.compile(r"^\s*cp\s+(?P<args>.*)$")
_MV_RE = re.compile(r"^\s*mv\s+(?P<args>.*)$")

# `export NAME=VALUE` and bare `NAME=VALUE` (the latter only counts as
# an env mutation when no command follows — `FOO=bar baz` is a one-shot
# env-for-baz, not a session-wide export).
_EXPORT_RE = re.compile(r"^\s*export\s+(?P<assign>\S+=.*)$")
_BARE_ASSIGN_RE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>.*)$")


def _strip_quotes(s: str) -> str:
    """Best-effort unquote: shlex.split handles edge cases, fall back to s."""
    try:
        parts = shlex.split(s)
        if len(parts) == 1:
            return parts[0]
        return " ".join(parts)
    except ValueError:
        return s


def _split_first_command(line: str) -> str:
    """Return the first command in a pipeline-shaped line.

    Same trick as the downloader parser: stop at the first shell
    separator. Don't try to be fully POSIX.
    """
    for sep in (";", "&&", "||", "|", "\n"):
        idx = line.find(sep)
        if idx != -1:
            return line[:idx]
    return line


# ----------------------------------------------------------------------
# Public API


def parse_input_mutations(line: str) -> list[CmdMutation]:
    """Return all world-state mutations a command would make.

    Empty list = no recognized pattern. Multiple = chained commands
    (we look at the first command of a pipeline only — chained
    `echo a > /x; echo b > /y` is not yet handled, defer).

    Tolerant of unparseable input: returns [] on shlex / regex
    failure rather than raising.
    """
    head = _split_first_command(line).strip()
    if not head:
        return []

    mutations: list[CmdMutation] = []

    # echo ... > path  (or >>)
    m = _ECHO_REDIRECT_RE.match(head)
    if m:
        content = _strip_quotes(m.group("content"))
        path = m.group("path").strip("'\"")
        kind: MutationKind = "append_file" if m.group("op") == ">>" else "create_file"
        mutations.append(CmdMutation(kind=kind, path=path, content=content))
        return mutations

    # touch ...
    m = _TOUCH_RE.match(head)
    if m:
        for arg in _tokens(m.group("args")):
            if not arg.startswith("-"):
                mutations.append(CmdMutation(kind="create_file", path=arg, content=""))
        return mutations

    # rm ...
    m = _RM_RE.match(head)
    if m:
        for arg in _tokens(m.group("args")):
            if not arg.startswith("-"):
                mutations.append(CmdMutation(kind="remove_file", path=arg))
        return mutations

    # cp src dst
    m = _CP_RE.match(head)
    if m:
        positional = [a for a in _tokens(m.group("args")) if not a.startswith("-")]
        if len(positional) >= 2:
            mutations.append(
                CmdMutation(kind="copy_file", path=positional[0], dst_path=positional[-1])
            )
        return mutations

    # mv src dst
    m = _MV_RE.match(head)
    if m:
        positional = [a for a in _tokens(m.group("args")) if not a.startswith("-")]
        if len(positional) >= 2:
            mutations.append(
                CmdMutation(kind="move_file", path=positional[0], dst_path=positional[-1])
            )
        return mutations

    # export NAME=VALUE
    m = _EXPORT_RE.match(head)
    if m:
        assign = m.group("assign")
        assign_m = _BARE_ASSIGN_RE.match(assign)
        if assign_m:
            mutations.append(CmdMutation(
                kind="set_env",
                env_name=assign_m.group("name"),
                env_value=_strip_quotes(assign_m.group("value")),
            ))
        return mutations

    # Standalone NAME=VALUE (only when nothing else follows — otherwise
    # bash treats it as a one-shot env for the next command).
    bare = _BARE_ASSIGN_RE.match(head)
    if bare and len(head.split()) == 1:
        mutations.append(CmdMutation(
            kind="set_env",
            env_name=bare.group("name"),
            env_value=_strip_quotes(bare.group("value")),
        ))

    return mutations


def _tokens(args_str: str) -> list[str]:
    try:
        return shlex.split(args_str)
    except ValueError:
        return args_str.split()
