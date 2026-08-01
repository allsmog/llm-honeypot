# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cowrie.llm.fidelity — the believability evaluation harness.

Covers the normalization/similarity scoring, the consistency invariants
(which double as a CI regression gate — they must hold for every persona),
coverage accounting, and the reference-comparison plumbing (with an
injected runner so the suite never shells out)."""

from __future__ import annotations

from twisted.trial import unittest

from cowrie.llm import fidelity
from cowrie.llm.persona import PERSONAS


class TestNormalizeSimilarity(unittest.TestCase):
    def test_normalize_masks_numbers_and_ips(self):
        n = fidelity.normalize("load 0.42 at 10.0.0.5 pid 1234")
        self.assertNotIn("0.42", n)
        self.assertNotIn("10.0.0.5", n)
        self.assertNotIn("1234", n)
        self.assertIn("IP", n)

    def test_normalize_drops_hostname(self):
        n = fidelity.normalize("user@web01:/root", drop=("web01",))
        self.assertNotIn("web01", n)
        self.assertIn("HOST", n)

    def test_identical_structure_scores_high(self):
        a = "MemTotal:        4015488 kB\nMemFree:         1586973 kB\n"
        b = "MemTotal:       16466560 kB\nMemFree:        15745588 kB\n"
        self.assertGreater(fidelity.similarity(a, b), 0.95)

    def test_different_structure_scores_low(self):
        a = "uid=0(root) gid=0(root) groups=0(root)\n"
        b = "total 48\ndrwxr-xr-x 2 root root 4096 Jun 1 x\n"
        self.assertLess(fidelity.similarity(a, b), 0.6)

    def test_both_empty_is_one(self):
        self.assertEqual(fidelity.similarity("", ""), 1.0)


class TestConsistencyInvariants(unittest.TestCase):
    """Every invariant must hold for every built-in persona — this is the
    regression gate the CLI exits non-zero on."""

    def test_all_personas_fully_consistent(self):
        for persona in PERSONAS:
            ctx = fidelity.build_context(persona.slug)
            results = fidelity.run_consistency(ctx)
            failures = [r for r in results if not r.passed]
            self.assertEqual(
                failures, [],
                msg=f"{persona.slug}: " + "; ".join(
                    f"{f.name} ({f.detail})" for f in failures
                ),
            )

    def test_invariants_are_nonempty(self):
        ctx = fidelity.build_context("ubuntu_22_04")
        self.assertGreaterEqual(len(fidelity.run_consistency(ctx)), 12)

    def test_gate_is_not_vacuous_detects_real_contradiction(self):
        # Prove the gate actually flips to FAIL when an invariant is broken:
        # patch one command's output to contradict /etc/hostname, and confirm
        # the hostname invariant reports passed=False (and the CLI would exit
        # non-zero).
        from cowrie.llm import responder as R

        ctx = fidelity.build_context("ubuntu_22_04")
        real_respond = R.respond

        def fake(cmd, c):
            if cmd == "cat /etc/hostname":
                return R.ResponderResult(output="WRONGHOST\n")
            return real_respond(cmd, c)

        self.patch(R, "respond", fake)
        results = fidelity.run_consistency(ctx)
        hostname_checks = [r for r in results if "hostname" in r.name]
        self.assertTrue(hostname_checks)
        self.assertTrue(
            any(not r.passed for r in hostname_checks),
            msg="gate did not detect the injected hostname contradiction",
        )


class TestCoverage(unittest.TestCase):
    def test_recon_corpus_fully_handled(self):
        # The recon corpus is curated to be exactly what the deterministic
        # layer should answer — coverage must be 100%.
        ctx = fidelity.build_context("ubuntu_22_04")
        rep = fidelity.coverage(ctx)
        self.assertEqual(rep.handled, rep.total)
        self.assertEqual(rep.deferred_commands, [])
        self.assertEqual(rep.rate, 1.0)

    def test_coverage_by_category_present(self):
        ctx = fidelity.build_context("alpine_3_19")
        rep = fidelity.coverage(ctx)
        self.assertIn("identity", rep.by_category)
        self.assertIn("kernel", rep.by_category)


class TestReferenceCompare(unittest.TestCase):
    def test_reference_uses_injected_runner(self):
        ctx = fidelity.build_context("ubuntu_22_04")

        # Fake host: echo back the honeypot's own output → perfect score.
        from cowrie.llm import responder as R

        def echo_runner(cmd):
            r = R.respond(cmd, ctx)
            return r.output if r else ""

        results = fidelity.reference_compare(
            ctx, commands=("whoami", "uname -r"), runner=echo_runner
        )
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertTrue(r.ran_on_host)
            self.assertEqual(r.similarity, 1.0)

    def test_reference_marks_absent_binary(self):
        ctx = fidelity.build_context("ubuntu_22_04")
        results = fidelity.reference_compare(
            ctx, commands=("whoami",), runner=lambda cmd: None
        )
        self.assertFalse(results[0].ran_on_host)
        self.assertIn("absent", results[0].note)

    def test_reference_safe_command_list_is_read_only(self):
        # Guard: nothing in the reference allowlist can mutate the host.
        dangerous = ("rm", "dd", "mkfs", "wget", "curl", "kill", ">", "mv",
                     "chmod", "chown", "nc", "bash", "sh ")
        for cmd in fidelity.SAFE_REFERENCE_COMMANDS:
            for d in dangerous:
                self.assertNotIn(d, cmd, f"{cmd!r} contains {d!r}")


class TestBuildContext(unittest.TestCase):
    def test_build_context_returns_usable_context(self):
        ctx = fidelity.build_context("centos_7", hostname="h", login_user="bob")
        self.assertEqual(ctx.hostname, "h")
        self.assertEqual(ctx.login_user, "bob")
        self.assertEqual(ctx.persona.slug, "centos_7")
