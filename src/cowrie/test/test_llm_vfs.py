# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cowrie.llm.vfs and the responder's ls/stat handlers.

The point of the virtual filesystem is coherence: ls/stat never contradict
each other or a second identical call, dotfiles hide without -a, ownership
and the $/# distinction are right, and session-created files (WorldState)
show up. Anything not modeled defers to the LLM.
"""

from __future__ import annotations

from twisted.trial import unittest

from cowrie.llm import responder as R
from cowrie.llm import vfs as V
from cowrie.llm.persona import pick_persona, roll_boot_time
from cowrie.llm.worldstate import WorldState


def _ctx(login_user="root", cwd="/root", seed="seed"):
    p = pick_persona("ubuntu_22_04", override="ubuntu_22_04")
    return R.ShellContext(
        persona=p, boot_time=roll_boot_time(p, seed), world=WorldState(),
        cwd=cwd, login_user=login_user, hostname="web01", seed=seed,
    )


class TestPathHelpers(unittest.TestCase):
    def test_normpath(self):
        self.assertEqual(V._normpath("/tmp/../etc//passwd"), "/etc/passwd")
        self.assertEqual(V._normpath(""), "/")
        self.assertEqual(V._normpath("/tmp/"), "/tmp")

    def test_split(self):
        self.assertEqual(V._split("/etc/passwd"), ("/etc", "passwd"))
        self.assertEqual(V._split("/"), ("/", ""))


class TestLsBasic(unittest.TestCase):
    def test_ls_root_dir(self):
        out = R.respond("ls /", _ctx()).output
        for d in ("bin", "etc", "home", "root", "tmp", "usr", "var"):
            self.assertIn(d, out)

    def test_bare_ls_hides_dotfiles(self):
        # cwd is /root which contains only dotfiles -> bare ls is empty.
        self.assertEqual(R.respond("ls", _ctx(cwd="/root")).output, "")

    def test_ls_a_shows_dotfiles_and_dot_entries(self):
        out = R.respond("ls -a", _ctx(cwd="/root")).output
        self.assertIn(".bashrc", out)
        self.assertIn(".", out.split())
        self.assertIn("..", out.split())

    def test_ls_empty_tmp(self):
        self.assertEqual(R.respond("ls /tmp", _ctx()).output, "")

    def test_ls_unknown_dir_defers(self):
        self.assertIsNone(R.respond("ls /opt/secret", _ctx()))

    def test_ls_recursive_or_sort_flags_defer(self):
        for cmd in ("ls -R /", "ls -lt /tmp", "ls -S /tmp"):
            self.assertIsNone(R.respond(cmd, _ctx()), cmd)

    def test_multi_path_ls_defers(self):
        self.assertIsNone(R.respond("ls /tmp /etc", _ctx()))


class TestLsLong(unittest.TestCase):
    def test_ls_l_root_home_owned_by_root(self):
        out = R.respond("ls -la /root", _ctx(login_user="root")).output
        self.assertIn("root     root", out)
        self.assertNotIn("ubuntu", out)
        self.assertIn(".bashrc", out)
        self.assertTrue(out.startswith("total "))

    def test_ls_l_user_home_owned_by_user(self):
        out = R.respond(
            "ls -la", _ctx(login_user="deploy", cwd="/home/deploy")
        ).output
        self.assertIn("deploy   deploy", out)
        # The '.' entry (the home dir itself) is owned by the user.
        dot_line = next(ln for ln in out.splitlines() if ln.endswith(" ."))
        self.assertIn("deploy", dot_line)

    def test_ls_l_mode_strings(self):
        out = R.respond("ls -la /root", _ctx()).output
        self.assertRegex(out, r"drwx------  2 root     root .*\.ssh")

    def test_tmp_has_sticky_bit_in_root_listing(self):
        out = R.respond("ls -la /", _ctx()).output
        tmp_line = next(ln for ln in out.splitlines() if ln.endswith(" tmp"))
        self.assertTrue(tmp_line.startswith("drwxrwxrwt"))


class TestWorldStateOverlay(unittest.TestCase):
    def test_downloaded_file_appears_in_ls(self):
        ctx = _ctx()
        ctx.world.add_file(path="/tmp/payload.sh", size_bytes=4823,
                           source="downloaded")
        self.assertIn("payload.sh", R.respond("ls /tmp", ctx).output)

    def test_ls_l_shows_real_size(self):
        ctx = _ctx()
        ctx.world.add_file(path="/tmp/x.bin", size_bytes=98765, source="downloaded")
        out = R.respond("ls -l /tmp", ctx).output
        self.assertIn("98765", out)

    def test_created_file_in_home(self):
        ctx = _ctx(login_user="root", cwd="/root")
        ctx.world.add_file(path="/root/notes.txt", size_bytes=42, source="created")
        out = R.respond("ls /root", ctx).output  # non-dotfile, shows without -a
        self.assertIn("notes.txt", out)


class TestConsistency(unittest.TestCase):
    def test_ls_is_stable_across_calls(self):
        ctx = _ctx()
        ctx.world.add_file(path="/tmp/a", size_bytes=10, source="created")
        self.assertEqual(
            R.respond("ls -la /tmp", ctx).output,
            R.respond("ls -la /tmp", ctx).output,
        )

    def test_ls_and_stat_agree_on_size(self):
        ctx = _ctx()
        ctx.world.add_file(path="/tmp/big", size_bytes=55555, source="downloaded")
        ls = R.respond("ls -l /tmp", ctx).output
        st = R.respond("stat /tmp/big", ctx).output
        self.assertIn("55555", ls)
        self.assertIn("Size: 55555", st)


class TestStat(unittest.TestCase):
    def test_stat_known_file(self):
        out = R.respond("stat /etc/passwd", _ctx()).output
        self.assertIn("File: /etc/passwd", out)
        self.assertIn("regular file", out)
        self.assertIn("0644", out)

    def test_stat_directory(self):
        out = R.respond("stat /tmp", _ctx()).output
        self.assertIn("directory", out)

    def test_stat_missing_file_errors(self):
        out = R.respond("stat /no/such/path", _ctx()).output
        self.assertIn("No such file or directory", out)

    def test_stat_worldstate_file(self):
        ctx = _ctx()
        ctx.world.add_file(path="/tmp/m", size_bytes=7, source="created")
        out = R.respond("stat /tmp/m", ctx).output
        self.assertIn("Size: 7", out)

    def test_stat_format_flag_defers(self):
        self.assertIsNone(R.respond("stat -c %s /etc/passwd", _ctx()))
