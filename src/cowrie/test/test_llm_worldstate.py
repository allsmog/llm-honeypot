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
        self.assertNotIn(f"/tmp/f0  size=0", section)

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
