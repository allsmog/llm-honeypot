# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: The state-audit battery: drive the real protocol through a
# ABOUTME: fixed script and read the resulting world directly. This exists
# ABOUTME: because five of the eight fault families are invisible to the
# ABOUTME: fidelity invariants and the probe alphabet.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Self

# Determinism constants. These MUST agree with scripts/probe_search.py --
# a test asserts it -- but the package does not import from scripts/, since
# src/ depending on scripts/ is the wrong direction.
FROZEN_NOW = 1_700_000_000.0
PINNED_BOOT = FROZEN_NOW - (771 * 86400 + 61_860)

PINNED_IP = "203.0.113.45"

MODEL_MARKER = "<model>"


class RecordingStub:
    """Stands in for LLMClient. Records every call; never varies."""

    def __init__(self) -> None:
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


@contextmanager
def frozen_clock():
    """Freeze the clock the responder reads, so uptime/loadavg/top derived
    values do not vary between the training run and the check run."""
    from cowrie.llm import responder as respondermod

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


@dataclass
class AuditSession:
    """One honeypot session driven for audit purposes.

    Deliberately not shared with probe_search.Session: that one carries
    search-specific machinery (transposition keys, model-hit tracking) and
    lives in scripts/, which src/ must not import from.
    """

    persona_slug: str = "auto"
    proto: object = None
    transport: object = None
    stub: RecordingStub = None
    events: list = field(default_factory=list)
    transcript: list = field(default_factory=list)

    def start(self) -> AuditSession:
        from cowrie.core.config import CowrieConfig
        from cowrie.llm import persona as personamod
        from cowrie.llm import protocol as protomod
        from cowrie.test.fake_server import FakeAvatar, FakeServer
        from cowrie.test.fake_transport import FakeTransport

        CowrieConfig.set("llm", "fastpath_jitter_ms_min", "0")
        CowrieConfig.set("llm", "fastpath_jitter_ms_max", "0")
        CowrieConfig.set("llm", "stream", "false")

        self.stub = RecordingStub()
        server = FakeServer()
        server.llm_client = self.stub
        avatar = FakeAvatar(server)
        self.proto = protomod.HoneyPotInteractiveProtocol(avatar)
        self.transport = FakeTransport("", "31337")
        self.proto.makeConnection(self.transport)
        self.proto.realClientIP = PINNED_IP
        if self.persona_slug != "auto":
            self.proto.persona = personamod.pick_persona(
                PINNED_IP, override=self.persona_slug
            )
        self.proto.boot_time = PINNED_BOOT
        self.transport.clear()

        self.events = []
        self._restore_log = protomod.log.msg
        protomod.log.msg = lambda *a, **k: self.events.append(k)
        self._restore_fetch = self._stub_downloader()
        return self

    @staticmethod
    def _stub_downloader():
        """Keep the battery off the network.

        The download probe drives a real wget through the interceptor,
        which would otherwise open a socket to example.com and leave a
        10-second timeout pending. That makes the battery depend on DNS
        (so training would not be reproducible) and leaves delayed calls
        behind that trip Trial's reactor-cleanliness check. The interceptor
        logs its event before dispatching, so stubbing the fetch itself
        leaves everything the audit measures intact.
        """
        from twisted.internet import defer

        from cowrie.llm import downloader as dlmod

        original = dlmod.fetch

        def _no_network(intent, *, log_event):
            return defer.succeed(
                dlmod.FetchResult(
                    outcome="failed_connection",
                    url=intent.url,
                    error_message="offline audit battery: fetch not attempted",
                )
            )

        dlmod.fetch = _no_network
        return original

    def stop(self) -> None:
        from cowrie.llm import downloader as dlmod
        from cowrie.llm import protocol as protomod

        protomod.log.msg = self._restore_log
        dlmod.fetch = self._restore_fetch
        try:
            self.proto.setTimeout(None)
        except Exception:
            pass

    def send(self, command: str) -> str:
        self.transport.clear()
        try:
            self.proto.lineReceived(command.encode())
        except Exception as e:
            out = f"<raised {type(e).__name__}: {e}>"
            self.transcript.append((command, out))
            return out
        out = self.transport.value().decode("utf-8", errors="replace")
        self.transcript.append((command, out))
        return out

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- observations ---------------------------------------------------

    def saw_event(self, eventid: str) -> bool:
        return any(e.get("eventid") == eventid for e in self.events)

    def snippet(self, path: str) -> str | None:
        fact = self.proto.world.files.get(path)
        return None if fact is None else (fact.content_snippet or "")


# Paths the battery writes to. Named so a stray leftover in a transcript is
# obviously ours and not something an attacker pattern created.
_APPEND_PATH = "/tmp/diag-append"
_DENIED_PATH = "/root/diag-denied"
# /var/tmp, not /tmp: `ls /` enumerates tmp, so node_for finds it in the
# parent listing and never reaches the skeleton-mode fallback. /var/tmp and
# /dev/shm have no enumerated parent, which is exactly why the historical
# over-refusal hit those two and left /tmp looking fine.
_PERMITTED_PATH = "/var/tmp/diag-permitted"


def run_audit(persona_slug: str) -> dict[str, str]:
    """Drive one session through the battery and return evidence values.

    Each check is independent of the others' outcomes: a fault that breaks
    one probe must not silently change what a later probe measures, or the
    evidence stops being conditionally independent given the fault.
    """
    with frozen_clock(), AuditSession(persona_slug=persona_slug) as session:
        return {
            "audit_append": _check_append(session),
            "audit_cd_absent": _check_cd_absent(session),
            "audit_ledger": _check_ledger(session),
            "audit_interactive": _check_interactive(session),
            "download_path": _check_download(session),
            **_check_permissions(session),
        }


def _check_append(session: AuditSession) -> str:
    """A chained append must be applied exactly once.

    Append is the only non-idempotent mutation kind, which is precisely why
    the historical double-apply survived so long: create/remove/cp/mv all
    produce the same world twice.
    """
    session.send(f"echo a >> {_APPEND_PATH} && ls")
    return "ok" if session.snippet(_APPEND_PATH) == "a" else "doubled"


def _check_cd_absent(session: AuditSession) -> str:
    before = session.proto.cwd
    session.send("cd /etc/diag-nonexistent")
    return "ok" if session.proto.cwd == before else "entered"


def _check_ledger(session: AuditSession) -> str:
    session.send("uname -a")
    session.send("free -m")
    return "ok" if session.proto.world.told_facts else "empty"


def _check_interactive(session: AuditSession) -> str:
    before = len(session.stub.calls)
    session.send("vi /tmp/diag-editor")
    hit_model = len(session.stub.calls) > before
    session.send("\x1b:q!\r")  # leave the editor so later probes see a shell
    return "model" if hit_model else "local"


def _check_download(session: AuditSession) -> str:
    """Fetch inside a chain: the interceptor must claim the whole remainder.

    The fidelity invariant, the probe `dropped-command` signal and this
    check are three views of one event, so they collapse into one variable
    rather than three that would triple-count the same observation.
    """
    before = len(session.events)
    out = session.send("cd /tmp && wget http://example.com/x.sh && echo diagdone")
    events = session.events[before:]
    if "diagdone" in out:
        # A later segment rendered before the asynchronous fetch narrated.
        return "chain_reorder"
    if not any(e.get("eventid") == "cowrie.llm.download_intercept" for e in events):
        return "intercept_missing"
    return "ok"


def _check_permissions(session: AuditSession) -> dict[str, str]:
    """As an unprivileged user: /root must refuse, /tmp must not.

    Both directions matter. Over-refusal is the more dangerous failure —
    refusing writes to /tmp rejects exactly the payloads the honeypot
    exists to capture — so it gets its own evidence node.
    """
    session.send("su nobody")
    session.send(f"echo x > {_DENIED_PATH}")
    refused_denied = session.saw_event("cowrie.llm.refused")
    leaked = session.snippet(_DENIED_PATH) is not None
    persist = "ok" if (refused_denied and not leaked) else "leaked"

    before = len(session.events)
    session.send(f"echo x > {_PERMITTED_PATH}")
    over_refused = any(
        e.get("eventid") == "cowrie.llm.refused" for e in session.events[before:]
    )
    return {
        "audit_refused_persist": persist,
        "refusal_anomaly": "refused" if over_refused else "ok",
    }
