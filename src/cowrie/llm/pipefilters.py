# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Splits a simple shell pipeline and applies head/tail/grep/wc to
# ABOUTME: text the deterministic responder already produced. Exists so
# ABOUTME: `free -m | head -2` stays local, instant and self-consistent
# ABOUTME: instead of falling through to the LLM the way any command
# ABOUTME: containing a pipe used to. Pure: no ShellContext, no I/O.

"""Simple pipeline splitting and output filters.

Measured motivation: of the commands that reached the LLM across the
attacker_sim corpus, 19% were piped forms of commands the emulator already
answers perfectly — including two of the three repeat probes in
FINGERPRINT_PROBE, the adversary written specifically to catch the honeypot
contradicting itself. Those were being routed to the component least able
to stay consistent.

Design follows ``responder._peel_sudo``: pure functions, identity
pass-through when not applicable, and a tri-state return where ``None``
means "recognized but not modelled" so the caller defers the whole pipeline
to the LLM. A wrong local answer is worse than a slow one, so anything
uncertain defers.
"""

from __future__ import annotations

import re
import shlex

#: Separators that mean "more than one command", which we do not model.
#: Checked before splitting on `|` — note `|` is a substring of `||`, so
#: order matters and `||` has to be rejected outright rather than split.
_UNSUPPORTED = (";", "&&", "||", ">", "<", "$(", "`", "\n")

_MAX_STAGES = 4


def split_pipeline(raw: str) -> list[str] | None:
    """Split ``raw`` on `|` into stages, or None if it isn't a simple pipeline.

    Returns ``[raw]`` unchanged when there is no pipe, so the non-piped path
    stays byte-identical to before this module existed.

    Returns None — defer everything — when the line contains any other shell
    operator, when a stage is empty (``ps aux |``, ``| head``, ``a || b``),
    or when it is implausibly long.
    """
    if not raw or "|" not in raw:
        return [raw]
    if any(op in raw for op in _UNSUPPORTED):
        return None

    stages = [segment.strip() for segment in raw.split("|")]
    if len(stages) > _MAX_STAGES:
        return None
    if any(not stage for stage in stages):
        # Empty stage: a trailing pipe, a leading pipe, or `||` that the
        # operator check above somehow missed. Real shells would wait for
        # more input or error; we defer.
        return None
    return stages


def _int_arg(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _lines(text: str) -> list[str]:
    return text.splitlines()


def _rebuild(lines: list[str]) -> str:
    """Reassemble filtered lines, preserving the trailing-newline contract.

    ResponderResult.output ends in a newline when non-empty; an empty
    result is "" and must stay distinguishable from None ("not handled").
    """
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _count_arg(args: list[str], default: int) -> tuple[int | None, bool]:
    """Parse head/tail's `-n N` or `-N`. Returns (count, ok)."""
    count = default
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-n":
            if i + 1 >= len(args):
                return None, False
            parsed = _int_arg(args[i + 1])
            if parsed is None:
                return None, False
            count = parsed
            i += 2
            continue
        if arg.startswith("-n") and len(arg) > 2:
            parsed = _int_arg(arg[2:])
            if parsed is None:
                return None, False
            count = parsed
            i += 1
            continue
        if re.fullmatch(r"-\d+", arg):
            count = int(arg[1:])
            i += 1
            continue
        # Any other flag (-f, -c, -q, --lines=…) is unmodelled.
        return None, False
    return count, True


def _filter_head(text: str, args: list[str]) -> str | None:
    count, ok = _count_arg(args, 10)
    if not ok or count is None:
        return None
    return _rebuild(_lines(text)[:count])


def _filter_tail(text: str, args: list[str]) -> str | None:
    # -f would never terminate; _count_arg rejects it along with every
    # other unmodelled flag.
    count, ok = _count_arg(args, 10)
    if not ok or count is None:
        return None
    lines = _lines(text)
    return _rebuild(lines[-count:] if count else [])


def _filter_grep(text: str, args: list[str]) -> str | None:
    """Modelled: -i, -v, -c, -m N, and a single pattern.

    Patterns are compiled with Python's `re`, not POSIX BRE/ERE, so
    edge-case syntax differs from GNU grep. -E and -P are rejected rather
    than approximated, because claiming to support a dialect we do not
    implement is worse than deferring.
    """
    ignore_case = invert = count_only = False
    max_count: int | None = None
    pattern: str | None = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-m":
            if i + 1 >= len(args):
                return None
            max_count = _int_arg(args[i + 1])
            if max_count is None:
                return None
            i += 2
            continue
        if arg.startswith("-m") and len(arg) > 2:
            max_count = _int_arg(arg[2:])
            if max_count is None:
                return None
            i += 1
            continue
        if arg.startswith("-") and len(arg) > 1 and not arg.startswith("--"):
            for flag in arg[1:]:
                if flag == "i":
                    ignore_case = True
                elif flag == "v":
                    invert = True
                elif flag == "c":
                    count_only = True
                else:
                    return None  # -E, -P, -r, -A/-B/-C, …
            i += 1
            continue
        if arg.startswith("--"):
            return None
        if pattern is not None:
            # A second positional would be a file operand; at this point in
            # a pipeline grep reads stdin, so this is not a shape we model.
            return None
        pattern = arg
        i += 1

    if pattern is None:
        return None
    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error:
        return None

    matched = [ln for ln in _lines(text) if bool(regex.search(ln)) != invert]
    if max_count is not None:
        matched = matched[:max_count]
    if count_only:
        return f"{len(matched)}\n"
    return _rebuild(matched)


def _filter_wc(text: str, args: list[str]) -> str | None:
    lines = _lines(text)
    mode = None
    for arg in args:
        if arg in ("-l", "-w", "-c"):
            if mode is not None:
                return None  # combined counters print several columns
            mode = arg
        else:
            return None
    if mode == "-l":
        return f"{len(lines)}\n"
    if mode == "-w":
        return f"{sum(len(ln.split()) for ln in lines)}\n"
    if mode == "-c":
        return f"{len(text)}\n"
    # Bare `wc`: lines, words, bytes.
    words = sum(len(ln.split()) for ln in lines)
    return f"{len(lines):>7}{words:>8}{len(text):>8}\n"


_FILTERS = {
    "head": _filter_head,
    "tail": _filter_tail,
    "grep": _filter_grep,
    "wc": _filter_wc,
}


def is_filter(stage: str) -> bool:
    """True if ``stage``'s command is one we can model at all."""
    try:
        argv = shlex.split(stage)
    except ValueError:
        return False
    return bool(argv) and argv[0] in _FILTERS


def apply_filter(text: str, stage: str) -> str | None:
    """Run one pipeline stage over ``text``.

    None means defer the whole pipeline: an unknown command, an unmodelled
    flag, or a malformed argument. Never raises.
    """
    try:
        argv = shlex.split(stage)
    except ValueError:
        return None
    if not argv:
        return None
    handler = _FILTERS.get(argv[0])
    if handler is None:
        return None
    try:
        return handler(text, argv[1:])
    except Exception:
        return None
