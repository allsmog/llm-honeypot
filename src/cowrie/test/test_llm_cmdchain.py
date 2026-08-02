# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for cowrie.llm.cmdchain — shell chain splitting.

The conditional-semantics tests are the substance. A splitter that ran
both sides of `cd /tmp || cd /var/run` would relocate the attacker to
/var/run where a real shell leaves them in /tmp — trading one tell for a
subtler one.
"""

from __future__ import annotations

import unittest

from cowrie.llm.cmdchain import is_simple, should_run, split_chain


class TestSplit(unittest.TestCase):
    def test_plain_command_is_one_segment(self):
        self.assertEqual(split_chain("uname -a"), [("", "uname -a")])

    def test_operators(self):
        self.assertEqual(
            split_chain("cd /tmp && ls"), [("", "cd /tmp"), ("&&", "ls")]
        )
        self.assertEqual(
            split_chain("cd /tmp || ls"), [("", "cd /tmp"), ("||", "ls")]
        )
        self.assertEqual(split_chain("pwd ; ls"), [("", "pwd"), (";", "ls")])

    def test_the_mirai_line(self):
        """The exact line from attacker_sim that exposed the bug."""
        got = split_chain("cd /tmp || cd /var/run || cd /mnt")
        self.assertEqual(
            got, [("", "cd /tmp"), ("||", "cd /var/run"), ("||", "cd /mnt")]
        )

    def test_no_space_around_operators(self):
        self.assertEqual(split_chain("pwd;ls"), [("", "pwd"), (";", "ls")])
        self.assertEqual(split_chain("pwd&&ls"), [("", "pwd"), ("&&", "ls")])

    def test_double_ampersand_is_not_two_backgrounds(self):
        """`&&` must be matched before a bare `&`, or the chain splits wrong."""
        self.assertEqual(split_chain("a && b"), [("", "a"), ("&&", "b")])

    def test_operators_inside_quotes_are_literal(self):
        self.assertEqual(
            split_chain('echo "a && b"'), [("", 'echo "a && b"')]
        )
        self.assertEqual(split_chain("echo 'x ; y'"), [("", "echo 'x ; y'")])

    def test_escaped_operator_is_literal(self):
        self.assertEqual(split_chain(r"echo a \; b"), [("", r"echo a \; b")])

    def test_backgrounding_declines(self):
        """A lone `&` is job control, handled elsewhere — do not mis-split it."""
        self.assertIsNone(split_chain("sleep 10 &"))

    def test_malformed_chains_decline(self):
        for line in ("cd /tmp &&", "; ls", "cd /tmp ; ; ls", "&& ls"):
            self.assertIsNone(split_chain(line), line)

    def test_unterminated_quote_declines(self):
        self.assertIsNone(split_chain('echo "unterminated'))

    def test_absurdly_long_chains_decline(self):
        self.assertIsNone(split_chain(" ; ".join(["ls"] * 20)))

    def test_empty(self):
        self.assertIsNone(split_chain(""))
        self.assertIsNone(split_chain("   "))


class TestConditionalSemantics(unittest.TestCase):
    """`&&` and `||` are not decoration — they decide what runs."""

    def test_first_segment_always_runs(self):
        self.assertTrue(should_run("", True))
        self.assertTrue(should_run("", False))

    def test_semicolon_always_runs(self):
        self.assertTrue(should_run(";", True))
        self.assertTrue(should_run(";", False))

    def test_and_runs_only_after_success(self):
        self.assertTrue(should_run("&&", True))
        self.assertFalse(should_run("&&", False))

    def test_or_runs_only_after_failure(self):
        self.assertFalse(should_run("||", True))
        self.assertTrue(should_run("||", False))


class TestIsSimple(unittest.TestCase):
    """Guards the fastpath, which used to accept any line whose first token
    happened to be a fastpath verb."""

    def test_plain_commands_are_simple(self):
        for cmd in ("cd /tmp", "pwd", "clear", "exit", "ls -la"):
            self.assertTrue(is_simple(cmd), cmd)

    def test_unsupported_constructs_are_not(self):
        for cmd in (
            "cd /tmp > /dev/null",
            "cd $(pwd)",
            "cd `pwd`",
            "cd /tmp < x",
            "cd /tmp\nls",
        ):
            self.assertFalse(is_simple(cmd), cmd)

    def test_empty_is_not_simple(self):
        self.assertFalse(is_simple(""))


if __name__ == "__main__":
    unittest.main()
