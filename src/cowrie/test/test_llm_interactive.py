# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cowrie.llm.interactive — full-screen program state machines.

These are the believability-critical exits: the SoK paper names vim/top as
the commands that reliably unmask honeypots. We verify each program paints a
plausible screen and, crucially, that the attacker can get back OUT the way
they expect (top: q; vi: :q / :q! / :wq / ZZ; less: q / space-at-end).
"""

from __future__ import annotations

from twisted.trial import unittest

from cowrie.llm import interactive as I


class TestTop(unittest.TestCase):
    def _top(self):
        return I.TopProgram(frame_provider=lambda: (
            "top - 12:00:00 up 5 days,  1 user,  load average: 0.1, 0.2, 0.3\n"
            "Tasks: 95 total\nMiB Mem : 3921.0 total\n"
            "  PID USER  COMMAND\n    1 root  systemd\n"
        ))

    def test_initial_clears_and_paints(self):
        out = self._top().render_initial()
        self.assertIn(b"\x1b[2J", out)       # clear screen
        self.assertIn(b"load average", out)
        self.assertIn(b"systemd", out)

    def test_q_quits(self):
        res = self._top().handle_key(b"q")
        self.assertTrue(res.done)

    def test_ctrl_c_quits(self):
        self.assertTrue(self._top().handle_key(b"\x03").done)

    def test_other_key_repaints_not_done(self):
        res = self._top().handle_key(b" ")
        self.assertFalse(res.done)
        self.assertIn(b"load average", res.output)

    def test_refresh_repaints(self):
        out = self._top().on_refresh()
        self.assertIn(b"load average", out)

    def test_frame_provider_exception_is_safe(self):
        def boom():
            raise RuntimeError("x")
        prog = I.TopProgram(frame_provider=boom)
        out = prog.render_initial()  # must not raise
        self.assertIn(b"load average", out)


class TestVi(unittest.TestCase):
    def test_initial_paints_empty_buffer(self):
        prog = I.ViProgram(filename="new.txt", content="", _new_file=True)
        out = prog.render_initial()
        self.assertIn(b"\x1b[2J", out)
        self.assertIn(b"~", out)             # empty-line markers
        self.assertIn(b"new.txt", out)
        self.assertIn(b"[New]", out)

    def test_initial_paints_existing_content(self):
        prog = I.ViProgram(filename="x.sh", content="line1\nline2", _new_file=False)
        out = prog.render_initial()
        self.assertIn(b"line1", out)
        self.assertIn(b"line2", out)
        self.assertIn(b"x.sh", out)

    def test_colon_q_quits(self):
        prog = I.ViProgram(filename="f")
        self.assertFalse(prog.handle_key(b":").done)
        self.assertFalse(prog.handle_key(b"q").done)
        self.assertTrue(prog.handle_key(b"\r").done)

    def test_colon_q_bang_quits(self):
        prog = I.ViProgram(filename="f")
        res = prog.handle_key(b":q!\r")
        self.assertTrue(res.done)

    def test_colon_wq_quits(self):
        prog = I.ViProgram(filename="f", content="x")
        self.assertTrue(prog.handle_key(b":wq\r").done)

    def test_ZZ_quits(self):
        prog = I.ViProgram(filename="f")
        self.assertFalse(prog.handle_key(b"Z").done)
        self.assertTrue(prog.handle_key(b"Z").done)

    def test_unknown_ex_command_does_not_quit(self):
        prog = I.ViProgram(filename="f")
        res = prog.handle_key(b":set number\r")
        self.assertFalse(res.done)
        self.assertIn(b"E492", res.output)

    def test_esc_cancels_command_line(self):
        prog = I.ViProgram(filename="f")
        prog.handle_key(b":")
        prog.handle_key(b"q")
        res = prog.handle_key(b"\x1b")  # ESC — abandons :q
        self.assertFalse(res.done)
        # A following Enter should NOT quit (command was cancelled).
        self.assertFalse(prog.handle_key(b"\r").done)


class TestLess(unittest.TestCase):
    def _content(self, n=100):
        return "\n".join(f"line {i}" for i in range(n))

    def test_initial_shows_first_page(self):
        prog = I.LessProgram(content=self._content(), rows=10)
        out = prog.render_initial()
        self.assertIn(b"line 0", out)
        self.assertIn(b"line 8", out)
        self.assertNotIn(b"line 50", out)

    def test_q_quits(self):
        self.assertTrue(I.LessProgram(content="x").handle_key(b"q").done)

    def test_space_pages_forward(self):
        prog = I.LessProgram(content=self._content(), rows=10)
        prog.render_initial()
        res = prog.handle_key(b" ")
        self.assertFalse(res.done)
        self.assertIn(b"line 9", res.output)

    def test_space_at_end_quits(self):
        prog = I.LessProgram(content="a\nb", rows=10)  # fits one page
        res = prog.handle_key(b" ")
        self.assertTrue(res.done)

    def test_short_content_shows_end_marker(self):
        prog = I.LessProgram(content="only one line", rows=10)
        out = prog.render_initial()
        self.assertIn(b"(END)", out)


class TestFactory(unittest.TestCase):
    def test_top(self):
        prog = I.make_program("top", top_frame=lambda: "top - x\n")
        self.assertIsInstance(prog, I.TopProgram)

    def test_htop_maps_to_top(self):
        prog = I.make_program("htop", top_frame=lambda: "x")
        self.assertIsInstance(prog, I.TopProgram)

    def test_top_without_provider_defers(self):
        self.assertIsNone(I.make_program("top"))

    def test_vi_new_file(self):
        prog = I.make_program("vi /tmp/new.sh", file_content=lambda p: None)
        self.assertIsInstance(prog, I.ViProgram)
        self.assertTrue(prog._new_file)
        self.assertEqual(prog.filename, "/tmp/new.sh")

    def test_vi_existing_file(self):
        prog = I.make_program("vim /etc/x", file_content=lambda p: "body")
        self.assertEqual(prog.content, "body")
        self.assertFalse(prog._new_file)

    def test_less_existing_file(self):
        prog = I.make_program("less /var/log/x", file_content=lambda p: "log\nlog")
        self.assertIsInstance(prog, I.LessProgram)

    def test_less_missing_file_defers(self):
        self.assertIsNone(I.make_program("less /nope", file_content=lambda p: None))

    def test_unknown_command_defers(self):
        self.assertIsNone(I.make_program("ls -la", file_content=lambda p: None))

    def test_empty_defers(self):
        self.assertIsNone(I.make_program("  "))
