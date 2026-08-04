# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Evidence grouping, the committed model artifact, and the
# ABOUTME: diagnosis itself. The grouping tests double as a drift guard:
# ABOUTME: a renamed fidelity invariant must fail here, loudly, rather
# ABOUTME: than fall into whichever group happens to be nearby.

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from cowrie.llm import fidelity
from cowrie.llm import persona as personamod
from cowrie.llm.intelligence import evidence, structure
from cowrie.llm.intelligence.diagnostics import (
    MODEL_PATH,
    diagnose,
    expected_information_gain,
    load_model,
)
from cowrie.llm.intelligence.network import BayesNet


@dataclass
class FakeCheck:
    name: str
    passed: bool


@dataclass
class FakeFinding:
    signal: str


class TestInvariantGrouping(unittest.TestCase):
    def test_every_live_invariant_name_maps_to_a_group(self) -> None:
        """The drift guard. fidelity.py owns these names; if one is added
        or renamed without classifying it, the diagnoser would silently
        stop seeing a whole class of failure."""
        for p in personamod.PERSONAS:
            checks = fidelity.run_consistency(fidelity.build_context(p.slug))
            for check in checks:
                # Raises KeyError for anything unclassified.
                structure.group_for_invariant(check.name)

    def test_groups_cover_the_full_invariant_count(self) -> None:
        checks = fidelity.run_consistency(
            fidelity.build_context("ubuntu_22_04")
        )
        grouped = sum(
            1 for c in checks if structure.group_for_invariant(c.name) is not None
        )
        # Every invariant is either in a group or is the download one.
        self.assertEqual(grouped + 1, len(checks))

    def test_unknown_invariant_raises(self) -> None:
        with self.assertRaises(KeyError):
            structure.group_for_invariant("a brand new invariant")

    def test_ls_fallback_spelling_is_grouped(self) -> None:
        self.assertEqual(
            structure.group_for_invariant("ls -la / lists tmp"), "fid_fsperm"
        )

    def test_download_invariant_is_not_its_own_group(self) -> None:
        self.assertIsNone(
            structure.group_for_invariant(structure.DOWNLOAD_INVARIANT)
        )

    def test_group_failures_marks_only_the_failing_group(self) -> None:
        checks = [
            FakeCheck("uname -r == persona kernel", False),
            FakeCheck("free total == persona memtotal", True),
            FakeCheck(structure.DOWNLOAD_INVARIANT, True),
        ]
        groups = evidence.group_failures(checks)
        self.assertEqual(groups["fid_kernel_cpu"], "fail")
        self.assertEqual(groups["fid_memory"], "ok")
        self.assertEqual(groups["_download_invariant_ok"], "ok")

    def test_group_fails_if_any_member_fails(self) -> None:
        checks = [
            FakeCheck("uname -r == persona kernel", True),
            FakeCheck("lscpu vendor agrees with cpuinfo", False),
        ]
        self.assertEqual(
            evidence.group_failures(checks)["fid_kernel_cpu"], "fail"
        )


class TestDownloadMerge(unittest.TestCase):
    """The three download views collapse into one node, by specificity."""

    def test_audit_wins_over_the_weaker_views(self) -> None:
        self.assertEqual(
            evidence.merge_download(
                "chain_reorder", invariant_ok=False, dropped_command=True
            ),
            "chain_reorder",
        )

    def test_invariant_beats_the_probe_signal(self) -> None:
        self.assertEqual(
            evidence.merge_download("ok", invariant_ok=False, dropped_command=True),
            "defer_leak",
        )

    def test_probe_signal_alone(self) -> None:
        self.assertEqual(
            evidence.merge_download("ok", invariant_ok=True, dropped_command=True),
            "intercept_missing",
        )

    def test_all_clear(self) -> None:
        self.assertEqual(
            evidence.merge_download("ok", invariant_ok=True, dropped_command=False),
            "ok",
        )

    def test_dropped_command_is_not_also_a_probe_node(self) -> None:
        # It feeds download_path only; counting it twice would let one
        # broken interceptor push the posterior harder than anything else.
        signals = evidence.probe_signals([FakeFinding("dropped-command")])
        self.assertEqual(signals, {"probe_prompt": "clear",
                                   "probe_contradiction": "clear"})

    def test_unknown_probe_signal_is_ignored(self) -> None:
        signals = evidence.probe_signals([FakeFinding("world-invariant")])
        self.assertEqual(signals["probe_prompt"], "clear")


class TestModelArtifact(unittest.TestCase):
    def test_committed_artifact_hash_is_self_consistent(self) -> None:
        doc = json.loads(MODEL_PATH.read_text())
        net = BayesNet.from_json(MODEL_PATH.read_text())
        self.assertEqual(doc["sha256"], net.content_hash())

    def test_artifact_structure_matches_the_declaration(self) -> None:
        """If structure.py changes without a retrain, the artifact is
        describing a network that no longer exists."""
        net = load_model()
        self.assertEqual(
            [v.name for v in structure.STRUCTURE], list(net.variables)
        )
        for declared in structure.STRUCTURE:
            stored = net.variables[declared.name]
            self.assertEqual(stored.domain, declared.domain, msg=declared.name)
            self.assertEqual(stored.parents, declared.parents, msg=declared.name)

    def test_artifact_records_its_own_inert_nodes(self) -> None:
        # The limitation travels with the model rather than living only in
        # a README someone may not read.
        net = load_model()
        self.assertIn("inert_nodes", net.meta)
        for node in net.meta["inert_nodes"]:
            self.assertIn(node, structure.EVIDENCE_NODES)


class TestDeterminismConstants(unittest.TestCase):
    def test_audit_pins_match_probe_search(self) -> None:
        """audit.py deliberately does not import probe_search — src/ must
        not depend on scripts/ — so the shared pins are asserted equal
        instead. If they drift, the two harnesses stop describing the same
        box and their signals can no longer be trained together."""
        import sys
        from pathlib import Path

        scripts = Path(__file__).resolve().parents[3] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            import probe_search
        finally:
            sys.path.remove(str(scripts))

        from cowrie.llm.intelligence import audit

        self.assertEqual(audit.FROZEN_NOW, probe_search.FROZEN_NOW)
        self.assertEqual(audit.PINNED_BOOT, probe_search.PINNED_BOOT)
        self.assertEqual(audit.MODEL_MARKER, probe_search.MODEL_MARKER)


class TestDiagnosis(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.net = load_model()

    def test_clean_evidence_names_no_fault(self) -> None:
        result = diagnose(self.net, {**structure.CLEAN_EVIDENCE, "persona": "debian"})
        self.assertEqual(result.top, "no_fault")
        self.assertTrue(result.confident)

    def test_a_leaked_refusal_points_at_the_transaction(self) -> None:
        result = diagnose(
            self.net,
            {**structure.CLEAN_EVIDENCE, "persona": "debian",
             "audit_append": "doubled", "audit_refused_persist": "leaked"},
        )
        self.assertEqual(result.top, "transition_state")

    def test_ambiguous_evidence_is_reported_as_ambiguous(self) -> None:
        """A doubled append alone cannot separate chain_dispatch from
        transition_state — both cause it. Saying so is the point."""
        result = diagnose(self.net, {"persona": "debian", "audit_append": "doubled"})
        self.assertFalse(result.confident)
        top_two = sorted(result.posterior.items(), key=lambda kv: -kv[1])[:2]
        self.assertEqual(
            {name for name, _ in top_two}, {"chain_dispatch", "transition_state"}
        )

    def test_next_evidence_names_the_check_that_separates_them(self) -> None:
        result = diagnose(self.net, {"persona": "debian", "audit_append": "doubled"})
        self.assertEqual(result.next_evidence[0][0], "audit_refused_persist")
        self.assertGreater(result.next_evidence[0][1], 0.1)

    def test_information_gain_is_zero_for_an_observed_node(self) -> None:
        evidence_set = {"persona": "debian", "audit_append": "doubled"}
        self.assertEqual(
            expected_information_gain(
                self.net, "fault", "audit_append", evidence_set
            ),
            0.0,
        )

    def test_information_gain_is_never_negative(self) -> None:
        evidence_set = {"persona": "debian"}
        for node in structure.EVIDENCE_NODES:
            gain = expected_information_gain(self.net, "fault", node, evidence_set)
            self.assertGreaterEqual(gain, 0.0, msg=node)

    def test_inert_nodes_carry_no_information(self) -> None:
        """A node that never fired in training must not move the posterior.
        This is what makes an unmodelled fault surface as low confidence
        rather than as a confident wrong answer."""
        inert = self.net.meta["inert_nodes"]
        self.assertTrue(inert, "expected the model to record inert nodes")
        base = diagnose(self.net, {"persona": "debian"}).posterior
        for node in inert:
            var = next(v for v in structure.STRUCTURE if v.name == node)
            for value in var.domain:
                shifted = diagnose(
                    self.net, {"persona": "debian", node: value}
                ).posterior
                for fault, p in base.items():
                    self.assertAlmostEqual(
                        p, shifted[fault], places=9,
                        msg=f"{node}={value} moved {fault}",
                    )

    def test_render_states_the_posterior_and_the_next_step(self) -> None:
        text = diagnose(
            self.net, {"persona": "debian", "audit_append": "doubled"}
        ).render()
        self.assertIn("posterior over fault family", text)
        self.assertIn("chain_dispatch", text)
        self.assertIn("most informative next evidence", text)

    def test_render_of_a_confident_result_names_the_fault(self) -> None:
        text = diagnose(
            self.net, {**structure.CLEAN_EVIDENCE, "persona": "debian"}
        ).render()
        self.assertIn("likely fault: no_fault", text)


class TestRecordValidation(unittest.TestCase):
    def test_build_record_produces_a_complete_record(self) -> None:
        checks = fidelity.run_consistency(fidelity.build_context("ubuntu_22_04"))
        record = evidence.build_record(
            fault="no_fault",
            persona_family="debian",
            checks=checks,
            findings=[],
            audit={**structure.CLEAN_EVIDENCE},
        )
        evidence.validate_record(record)
        self.assertEqual(record["fault"], "no_fault")

    def test_validate_rejects_a_missing_node(self) -> None:
        with self.assertRaises(ValueError):
            evidence.validate_record({"fault": "no_fault", "persona": "debian"})

    def test_validate_rejects_an_out_of_domain_value(self) -> None:
        checks = fidelity.run_consistency(fidelity.build_context("ubuntu_22_04"))
        record = evidence.build_record(
            fault="no_fault", persona_family="debian", checks=checks,
            findings=[], audit={**structure.CLEAN_EVIDENCE},
        )
        record["audit_append"] = "sideways"
        with self.assertRaises(ValueError):
            evidence.validate_record(record)


if __name__ == "__main__":
    unittest.main()
