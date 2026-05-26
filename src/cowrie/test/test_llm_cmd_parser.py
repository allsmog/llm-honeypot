# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cowrie.llm.cmd_parser — parsing attacker input for
filesystem/env mutations to mirror into WorldState."""

from __future__ import annotations

from twisted.trial import unittest

from cowrie.llm.cmd_parser import parse_input_mutations


class TestEchoRedirect(unittest.TestCase):
    def test_echo_overwrite(self):
        muts = parse_input_mutations('echo hello > /tmp/x')
        self.assertEqual(len(muts), 1)
        self.assertEqual(muts[0].kind, "create_file")
        self.assertEqual(muts[0].path, "/tmp/x")
        self.assertEqual(muts[0].content, "hello")

    def test_echo_append(self):
        muts = parse_input_mutations('echo line2 >> /tmp/x')
        self.assertEqual(muts[0].kind, "append_file")
        self.assertEqual(muts[0].content, "line2")

    def test_echo_quoted_content(self):
        muts = parse_input_mutations('echo "hello world" > /tmp/y')
        self.assertEqual(muts[0].content, "hello world")
        self.assertEqual(muts[0].path, "/tmp/y")


class TestTouch(unittest.TestCase):
    def test_touch_single(self):
        muts = parse_input_mutations("touch /tmp/a")
        self.assertEqual(len(muts), 1)
        self.assertEqual(muts[0].kind, "create_file")
        self.assertEqual(muts[0].path, "/tmp/a")
        self.assertEqual(muts[0].content, "")

    def test_touch_multiple(self):
        muts = parse_input_mutations("touch /tmp/a /tmp/b /tmp/c")
        self.assertEqual(len(muts), 3)
        self.assertEqual([m.path for m in muts], ["/tmp/a", "/tmp/b", "/tmp/c"])


class TestRemove(unittest.TestCase):
    def test_rm_single(self):
        muts = parse_input_mutations("rm /tmp/x")
        self.assertEqual(len(muts), 1)
        self.assertEqual(muts[0].kind, "remove_file")
        self.assertEqual(muts[0].path, "/tmp/x")

    def test_rm_recursive(self):
        muts = parse_input_mutations("rm -rf /tmp/dir")
        self.assertEqual(len(muts), 1)
        self.assertEqual(muts[0].path, "/tmp/dir")

    def test_rm_multiple(self):
        muts = parse_input_mutations("rm -f /tmp/a /tmp/b")
        self.assertEqual([m.path for m in muts], ["/tmp/a", "/tmp/b"])


class TestCopyMove(unittest.TestCase):
    def test_cp(self):
        muts = parse_input_mutations("cp /tmp/a /tmp/b")
        self.assertEqual(muts[0].kind, "copy_file")
        self.assertEqual(muts[0].path, "/tmp/a")
        self.assertEqual(muts[0].dst_path, "/tmp/b")

    def test_mv(self):
        muts = parse_input_mutations("mv /tmp/a /tmp/b")
        self.assertEqual(muts[0].kind, "move_file")
        self.assertEqual(muts[0].path, "/tmp/a")
        self.assertEqual(muts[0].dst_path, "/tmp/b")

    def test_cp_with_flags(self):
        muts = parse_input_mutations("cp -r /tmp/src /tmp/dst")
        self.assertEqual(muts[0].path, "/tmp/src")
        self.assertEqual(muts[0].dst_path, "/tmp/dst")


class TestEnv(unittest.TestCase):
    def test_export(self):
        muts = parse_input_mutations("export PATH=/usr/bin")
        self.assertEqual(muts[0].kind, "set_env")
        self.assertEqual(muts[0].env_name, "PATH")
        self.assertEqual(muts[0].env_value, "/usr/bin")

    def test_export_quoted(self):
        muts = parse_input_mutations('export NAME="evil binary"')
        self.assertEqual(muts[0].env_value, "evil binary")

    def test_bare_assignment(self):
        muts = parse_input_mutations("FOO=bar")
        self.assertEqual(muts[0].kind, "set_env")
        self.assertEqual(muts[0].env_name, "FOO")
        self.assertEqual(muts[0].env_value, "bar")

    def test_oneshot_env_not_a_mutation(self):
        # `FOO=bar cmd` is a one-shot env for cmd, not a session-wide export.
        muts = parse_input_mutations("FOO=bar /usr/bin/env")
        self.assertEqual(muts, [])


class TestNonMutating(unittest.TestCase):
    def test_ls_returns_no_mutations(self):
        self.assertEqual(parse_input_mutations("ls -la /tmp"), [])

    def test_whoami(self):
        self.assertEqual(parse_input_mutations("whoami"), [])

    def test_empty_returns_empty(self):
        self.assertEqual(parse_input_mutations(""), [])
        self.assertEqual(parse_input_mutations("   "), [])

    def test_pipeline_only_first_command(self):
        # ls | tee > /tmp/x is NOT an echo-redirect (the redirect is
        # part of the second segment). We only look at the first
        # segment of pipelines, so this returns no mutations.
        self.assertEqual(parse_input_mutations("ls | tee > /tmp/x"), [])

    def test_unparseable_quotes_dont_crash(self):
        # Single unterminated quote shouldn't raise; the regex may still
        # match the redirect pattern and produce a best-effort mutation,
        # which is fine — the contract is "no crash."
        try:
            parse_input_mutations("echo 'oops > /tmp/x")
        except Exception as e:
            self.fail(f"raised on unbalanced quotes: {e!r}")
