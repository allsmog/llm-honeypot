# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Anti-vacuity for the fault switches. A switch that does not
# ABOUTME: actually break anything would train the network on labels with
# ABOUTME: no signal behind them, so each one is proven to flip the exact
# ABOUTME: observation it claims and the clean run is proven not to.

from __future__ import annotations

import unittest

from cowrie.llm import fidelity
from cowrie.llm.intelligence import structure
from cowrie.llm.intelligence.audit import run_audit
from cowrie.llm.intelligence.faults import FAULTS, inject


class TestFaultSwitchesAreNotVacuous(unittest.TestCase):
    """Each switch must move its own observation and leave the rest alone.

    These are slow-ish (each case drives a real protocol session) but they
    are the load-bearing tests of the whole package: without them the
    training labels are assertions rather than measurements.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.clean = run_audit("ubuntu_22_04")

    def _audit(self, fault: str, slug: str = "ubuntu_22_04") -> dict[str, str]:
        with inject(fault):
            return run_audit(slug)

    def test_clean_run_has_no_findings(self) -> None:
        for node, value in self.clean.items():
            self.assertEqual(
                value, structure.CLEAN_EVIDENCE[node], msg=f"{node} on a clean run"
            )

    def test_chain_dispatch_doubles_a_chained_append(self) -> None:
        self.assertEqual(self.clean["audit_append"], "ok")
        self.assertEqual(self._audit("chain_dispatch")["audit_append"], "doubled")

    def test_transition_state_lets_a_refused_write_persist(self) -> None:
        self.assertEqual(self.clean["audit_refused_persist"], "ok")
        got = self._audit("transition_state")
        self.assertEqual(got["audit_refused_persist"], "leaked")

    def test_vfs_accepts_a_cd_to_an_absent_path(self) -> None:
        self.assertEqual(self.clean["audit_cd_absent"], "ok")
        self.assertEqual(self._audit("vfs")["audit_cd_absent"], "entered")

    def test_vfs_over_refuses_writes_to_var_tmp(self) -> None:
        # The dangerous half of the historical VFS bug: refusing writes to
        # a world-writable directory rejects exactly the payload drops the
        # honeypot exists to capture.
        self.assertEqual(self.clean["refusal_anomaly"], "ok")
        self.assertEqual(self._audit("vfs")["refusal_anomaly"], "refused")

    def test_fact_ledger_records_nothing(self) -> None:
        self.assertEqual(self.clean["audit_ledger"], "ok")
        self.assertEqual(self._audit("fact_ledger")["audit_ledger"], "empty")

    def test_downloader_lets_a_later_segment_render_first(self) -> None:
        self.assertEqual(self.clean["download_path"], "ok")
        self.assertEqual(self._audit("downloader")["download_path"], "chain_reorder")

    def test_interactive_defers_full_screen_programs_to_the_model(self) -> None:
        self.assertEqual(self.clean["audit_interactive"], "local")
        self.assertEqual(self._audit("interactive")["audit_interactive"], "model")

    def test_no_fault_changes_nothing(self) -> None:
        self.assertEqual(self._audit("no_fault"), self.clean)

    def test_persona_fault_contradicts_cpu_identity_on_alpine(self) -> None:
        # Only alpine carries an AMD model name, so this is the one persona
        # where a hardcoded GenuineIntel contradicts the line beneath it.
        with inject("persona"):
            failed = self._failed_invariants("alpine_3_19")
        self.assertIn("cpuinfo vendor_id matches model name", failed)
        self.assertIn("lscpu vendor agrees with cpuinfo", failed)

    def test_persona_fault_vouches_for_uninstalled_binaries(self) -> None:
        # The flat binary set is wrong on every persona, not just alpine.
        for slug in ("ubuntu_22_04", "centos_7", "alpine_3_19"):
            with inject("persona"):
                failed = self._failed_invariants(slug)
            self.assertTrue(
                any(name.startswith("which ") for name in failed),
                msg=f"no which invariant failed on {slug}",
            )

    def test_clean_personas_pass_every_invariant(self) -> None:
        for slug in ("ubuntu_22_04", "centos_7", "alpine_3_19"):
            self.assertEqual(self._failed_invariants(slug), [], msg=slug)

    @staticmethod
    def _failed_invariants(slug: str) -> list[str]:
        ctx = fidelity.build_context(slug)
        return [c.name for c in fidelity.run_consistency(ctx) if not c.passed]


class TestSeamsAreRestored(unittest.TestCase):
    def test_every_switch_restores_what_it_patched(self) -> None:
        from cowrie.llm import protocol as protomod
        from cowrie.llm import responder as respondermod
        from cowrie.llm import vfs as vfsmod

        proto_cls = protomod.HoneyPotBaseProtocol
        seams = {
            "_dispatch_chain": proto_cls,
            "_plan_input_mutations": proto_cls,
            "_cd_status": proto_cls,
            "_record_claim": proto_cls,
            "_try_interactive": proto_cls,
            "_cpu_identity": respondermod,
            "_which_path": respondermod,
            "_SKELETON_DIR_MODES": vfsmod,
        }
        before = {name: getattr(owner, name) for name, owner in seams.items()}
        for fault in FAULTS:
            with inject(fault):
                pass
            for name, owner in seams.items():
                self.assertIs(
                    getattr(owner, name), before[name],
                    msg=f"{name} not restored after {fault}",
                )

    def test_unknown_fault_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            with inject("not-a-fault"):
                pass

    def test_seams_restored_even_when_the_block_raises(self) -> None:
        from cowrie.llm import responder as respondermod

        before = respondermod._cpu_identity
        with self.assertRaises(RuntimeError):
            with inject("persona"):
                raise RuntimeError("boom")
        self.assertIs(respondermod._cpu_identity, before)


if __name__ == "__main__":
    unittest.main()
