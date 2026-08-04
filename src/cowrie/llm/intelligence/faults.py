# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Fault-injection switches: re-introduce a historically real,
# ABOUTME: since-fixed bug for the duration of a with-block. Harness-side
# ABOUTME: monkeypatching only — nothing here is reachable from a running
# ABOUTME: honeypot, and no production code has a switch to accommodate it.

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

# Every fault except `interactive` re-creates a bug this repository shipped
# and then fixed. That is the whole reason the diagnoser's ground truth is
# defensible: the labels come from real regressions, not from invented
# failure modes.
FAULTS: dict[str, str] = {
    "chain_dispatch": (
        "Chained mutations applied twice: the remainder handed to the model "
        "was re-parsed, and the parser reads only a line's head, so the same "
        "segment was committed again. `echo a >> f && ls` wrote 'aa'."
    ),
    "transition_state": (
        "Eager mutation: the world was written from parsed input before "
        "anything established the command was allowed, so a refused write "
        "still left the file in WorldState and the next ls listed it."
    ),
    "vfs": (
        "VFS not authoritative: cd accepted any path, and /tmp fell back to "
        "root-owned 0755 so the planner refused the writes payloads use."
    ),
    "persona": (
        "Persona facts answered from flat hardcoded tables: vendor_id "
        "GenuineIntel three lines above an AMD model name, and `which` "
        "vouching for binaries the persona's package list omits. Two "
        "separate defects, both found together and both persona-blindness."
    ),
    "fact_ledger": (
        "Claims never recorded, so a re-probe got a fresh answer from the "
        "model with nothing to hold it to what the session already said."
    ),
    "downloader": (
        "Download intercepted per-segment inside a chain and treated as "
        "synchronous success, so a later segment rendered its output and "
        "prompt before the fetch narration arrived."
    ),
    "interactive": (
        "Full-screen programs deferred to the model. Not a historical bug: "
        "this switch ablates the subsystem, and is labelled as such."
    ),
    "no_fault": "Nothing injected.",
}


@contextmanager
def _patched(target: object, name: str, value: object) -> Iterator[None]:
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


@contextmanager
def inject(fault: str) -> Iterator[None]:
    """Re-introduce ``fault`` for the duration of the block.

    Patches are applied to classes and modules, not instances, so a session
    constructed inside the block picks them up. Everything is restored in a
    finally, and a test asserts the seams are identical afterwards.
    """
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; known: {sorted(FAULTS)}")

    from cowrie.llm import protocol as protomod
    from cowrie.llm import responder as respondermod
    from cowrie.llm import vfs as vfsmod

    proto_cls = protomod.HoneyPotBaseProtocol

    with ExitStack() as stack:
        if fault == "chain_dispatch":
            stack.enter_context(
                _patched(proto_cls, "_dispatch_chain", _chain_double_apply)
            )
        elif fault == "transition_state":
            stack.enter_context(
                _patched(proto_cls, "_plan_input_mutations", _plan_eagerly)
            )
        elif fault == "vfs":
            stack.enter_context(
                _patched(proto_cls, "_cd_status", lambda self, path: "unknown")
            )
            stack.enter_context(
                _patched(vfsmod, "_SKELETON_DIR_MODES", {})
            )
        elif fault == "persona":
            stack.enter_context(
                _patched(
                    respondermod,
                    "_cpu_identity",
                    lambda persona: ("GenuineIntel", "6", "85", "7"),
                )
            )
            stack.enter_context(
                _patched(respondermod, "_which_path", _which_from_flat_set)
            )
        elif fault == "fact_ledger":
            stack.enter_context(
                _patched(
                    proto_cls,
                    "_record_claim",
                    lambda self, command, answer, *, source: None,
                )
            )
        elif fault == "downloader":
            stack.enter_context(
                _patched(proto_cls, "_dispatch_chain", _chain_sync_download)
            )
        elif fault == "interactive":
            stack.enter_context(
                _patched(proto_cls, "_try_interactive", lambda self, command: False)
            )
        yield


def _chain_double_apply(self, segments: list[tuple[str, str]]) -> None:
    """The chain dispatcher as it stood before the double-apply fix: the
    remainder is re-planned and re-committed even though _dispatch_local
    already applied the head segment's mutations."""
    from cowrie.llm import cmdchain

    succeeded = True
    for index, (operator, command) in enumerate(segments):
        if not cmdchain.should_run(operator, succeeded):
            succeeded = True
            continue
        self._suppress_prompt = True
        try:
            handled, succeeded = self._dispatch_local(command)
        finally:
            self._suppress_prompt = False
        if not handled:
            remainder = command
            for op, cmd in segments[index + 1 :]:
                remainder += f" {op} {cmd}"
            self._commit_mutations(self._plan_input_mutations(remainder))
            if self._try_download_intercept(remainder):
                return
            self._process_command_with_llm(remainder)
            return
    self._show_prompt()


def _chain_sync_download(self, segments: list[tuple[str, str]]) -> None:
    """The chain dispatcher as it stood when the interceptor ran per
    segment and reported synchronous success, letting later segments
    render ahead of the fetch narration."""
    from cowrie.llm import cmdchain

    succeeded = True
    for index, (operator, command) in enumerate(segments):
        if not cmdchain.should_run(operator, succeeded):
            succeeded = True
            continue
        self._suppress_prompt = True
        try:
            handled, succeeded = self._dispatch_local(command)
            if not handled and self._try_download_intercept(command):
                handled, succeeded = True, True
        finally:
            self._suppress_prompt = False
        if not handled:
            remainder = command
            for op, cmd in segments[index + 1 :]:
                remainder += f" {op} {cmd}"
            self._process_command_with_llm(remainder)
            return
    self._show_prompt()


#: The flat binary set `which` used to answer from, before it consulted the
#: persona. It reported /usr/bin/bash on a busybox box and vouched for
#: git/perl/rsync that the persona's package list never claimed.
_FLAT_BINS = frozenset(
    ("bash", "sh", "ls", "cat", "grep", "ps", "wget", "curl", "python3",
     "git", "vim", "perl", "rsync", "tar", "gzip", "ssh", "nc")
)


def _which_from_flat_set(name: str, ctx) -> str | None:
    return f"/usr/bin/{name}" if name in _FLAT_BINS else None


def _plan_eagerly(self, command: str):
    """Pre-transactional behaviour: commit whatever the parser found, then
    hand back a plan that claims nothing needs doing. The refusal is still
    computed and rendered, so the shell says "Permission denied" while the
    file quietly exists — exactly the inconsistency the transaction fixed."""
    from cowrie.llm import cmd_parser
    from cowrie.llm import state as statemod
    from cowrie.llm import vfs as vfsmod

    try:
        mutations = cmd_parser.parse_input_mutations(command)
        if not mutations:
            return statemod.Plan()
        vfs = vfsmod.VFS(self.world, str(self.user.username))
        plan = statemod.plan_mutations(
            mutations,
            vfs,
            user=self._effective_user(),
            login_user=str(self.user.username),
        )
        # The bug: apply first, ask later.
        self._commit_mutations(statemod.Plan(mutations=tuple(mutations)))
    except Exception:
        return statemod.Plan()
    else:
        return plan
