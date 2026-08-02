# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for cowrie.llm.factkeys.

The point of a fact family is that commands which *must agree* share a key:
`uname -a` and `uname -r` are two views of one answer, so answering either
commits us to the other. Getting a key wrong is worse than having none —
two unrelated facts sharing a key would overwrite each other and
manufacture a contradiction out of correct behaviour.

So the aliasing tests below are the substance, and "unrecognized returns
None" is a deliberate design property rather than a gap.
"""

from __future__ import annotations

import unittest

from cowrie.llm.factkeys import fact_family


class TestAliasing(unittest.TestCase):
    """Commands that must agree share a key."""

    def test_kernel_views(self):
        for cmd in ("uname -a", "uname -r", "uname -sr", "cat /proc/version"):
            self.assertEqual(fact_family(cmd), "os.kernel", cmd)

    def test_distro_views(self):
        for cmd in (
            "cat /etc/os-release",
            "cat /etc/issue",
            "lsb_release -a",
            "cat /etc/debian_version",
            "cat /etc/alpine-release",
        ):
            self.assertEqual(fact_family(cmd), "os.distro", cmd)

    def test_memory_views(self):
        for cmd in ("free", "free -m", "free -h", "cat /proc/meminfo"):
            self.assertEqual(fact_family(cmd), "hw.mem", cmd)

    def test_cpu_views(self):
        for cmd in ("cat /proc/cpuinfo", "lscpu", "nproc"):
            self.assertEqual(fact_family(cmd), "hw.cpu", cmd)

    def test_identity_views(self):
        for cmd in ("whoami", "id", "groups", "logname"):
            self.assertEqual(fact_family(cmd), "id.user", cmd)

    def test_account_views(self):
        for cmd in ("cat /etc/passwd", "cat /etc/shadow", "getent passwd"):
            self.assertEqual(fact_family(cmd), "id.accounts", cmd)


class TestDisambiguation(unittest.TestCase):
    """Keys that would be wrong if they collided."""

    def test_arch_is_not_the_kernel(self):
        """`uname -m` reports the architecture, `uname -r` the release.

        Same binary, different facts — if these shared a key, asking both
        would look like the honeypot contradicting itself.
        """
        self.assertEqual(fact_family("uname -m"), "os.arch")
        self.assertEqual(fact_family("arch"), "os.arch")
        self.assertNotEqual(fact_family("uname -m"), fact_family("uname -r"))

    def test_different_files_get_different_keys(self):
        self.assertEqual(fact_family("cat /tmp/a"), "file:/tmp/a")
        self.assertEqual(fact_family("cat /tmp/b"), "file:/tmp/b")
        self.assertNotEqual(fact_family("cat /tmp/a"), fact_family("cat /tmp/b"))

    def test_known_paths_key_by_meaning_not_by_path(self):
        """`cat /etc/os-release` is the distro, not an arbitrary file, so
        `lsb_release -a` agrees with it."""
        self.assertEqual(fact_family("cat /etc/os-release"), "os.distro")

    def test_accounts_and_user_are_distinct(self):
        self.assertNotEqual(fact_family("whoami"), fact_family("cat /etc/passwd"))


class TestVolatile(unittest.TestCase):
    """Commands whose answer is *supposed* to change are not tracked.

    Recording them would turn correct behaviour into a contradiction the
    moment the attacker asked twice.
    """

    def test_clock_and_session_commands_are_untracked(self):
        for cmd in ("date", "date -u", "w", "top", "top -bn1"):
            self.assertIsNone(fact_family(cmd), cmd)

    def test_the_w_rule_does_not_swallow_other_w_commands(self):
        """Regression: `^\\s*(date|w|top\\b)` matched the w in `whoami`.

        The word boundary belonged on the group, not on one alternative.
        """
        self.assertEqual(fact_family("whoami"), "id.user")
        self.assertIsNone(fact_family("wc -l"))
        self.assertIsNone(fact_family("which curl"))


class TestPipelines(unittest.TestCase):
    def test_only_the_first_stage_determines_the_fact(self):
        """`free -m | head -2` still asserts the memory figures."""
        self.assertEqual(fact_family("free -m | head -2"), "hw.mem")
        self.assertEqual(fact_family("cat /etc/os-release | head -3"), "os.distro")
        self.assertEqual(
            fact_family("cat /proc/cpuinfo | grep -m1 'model name'"), "hw.cpu"
        )

    def test_piped_and_unpiped_forms_share_a_key(self):
        """This is what makes a re-probe in piped form detectable."""
        self.assertEqual(fact_family("free -m"), fact_family("free -m | head -2"))


class TestTotality(unittest.TestCase):
    def test_unrecognized_commands_return_none(self):
        for cmd in ("", "   ", "ls -la", "cd /tmp", "echo hi", "wget http://x"):
            self.assertIsNone(fact_family(cmd), cmd)

    def test_never_raises(self):
        junk = (
            "\x00bad",
            "cat '" + "x" * 5000,
            "|||",
            "cat " + "/" * 3000,
            "uname " + "-" * 2000,
        )
        for value in junk:
            try:
                fact_family(value)
            except Exception as e:  # pragma: no cover
                self.fail(f"fact_family raised on {value!r}: {e}")


if __name__ == "__main__":
    unittest.main()
