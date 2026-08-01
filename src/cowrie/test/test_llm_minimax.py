# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the search's semantics and output contract.

Scope: the shape of what plan() returns, its edge cases, and the one
property that makes it minimax rather than just tree-walking — that the
reported value really is the worst case over the attacker's replies.

Not in scope: whether the recommended policy is a *good* choice. That
follows from the hand-tuned effect tables, and pinning specific
recommendations here would freeze numbers we expect to retune. Exactly one
directed sanity check is kept, at the bottom, for the case where any
reasonable weighting should agree.

Equivalence with the plain minimax oracle lives in
test_llm_minimax_equivalence.py.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from cowrie.llm.planner.actions import AttackerAction, CutoffStatus, DefenderPolicy
from cowrie.llm.planner.evaluate import components_from, evaluate
from cowrie.llm.planner.search import plan
from cowrie.llm.planner.state import GameState, UtilityWeights
from cowrie.llm.planner.transition import (
    apply_attacker,
    apply_defender,
    legal_attacker_actions,
)

A = AttackerAction
P = DefenderPolicy


def recon_state(**kw) -> GameState:
    return GameState(**kw).with_attack(A.OS_FINGERPRINT)


class TestEdgeCases(unittest.TestCase):
    def test_depth_zero_scores_the_root_and_recommends_nothing(self):
        s = recon_state()
        r = plan(s, depth=0)
        self.assertIsNone(r.best_policy)
        self.assertEqual(r.value, evaluate(s, UtilityWeights()))
        self.assertEqual(r.pv, ())
        self.assertIs(r.cutoff, CutoffStatus.DEPTH_LIMIT)
        self.assertEqual(r.nodes_expanded, 1)

    def test_terminal_root_reports_terminal(self):
        s = replace(recon_state(), terminated=True)
        r = plan(s, depth=4)
        self.assertIsNone(r.best_policy)
        self.assertIs(r.cutoff, CutoffStatus.TERMINAL)
        self.assertEqual(r.pv, ())

    def test_negative_depth_is_rejected(self):
        with self.assertRaises(ValueError):
            plan(recon_state(), depth=-1)

    def test_unknown_algorithm_is_rejected(self):
        with self.assertRaises(ValueError):
            plan(recon_state(), depth=2, algorithm="mcts")

    def test_empty_mask_still_produces_a_decision(self):
        """The protocol seam declines the turn before calling us when no
        policy is available, but if it ever does call us we must not
        return an infinitely-good phantom move."""
        r = plan(recon_state(), depth=3, allowed=frozenset())
        self.assertIs(r.best_policy, P.DELAY_PRESSURE)


class TestOutputContract(unittest.TestCase):
    def test_pv_alternates_and_is_bounded_by_depth(self):
        r = plan(recon_state(), depth=4)
        self.assertLessEqual(len(r.pv), 4)
        for i, move in enumerate(r.pv):
            self.assertIsInstance(move, DefenderPolicy if i % 2 == 0 else AttackerAction)

    def test_worst_case_reply_is_the_second_pv_entry(self):
        r = plan(recon_state(), depth=4)
        self.assertEqual(r.worst_case_reply, r.pv[1])

    def test_components_describe_the_pv_leaf(self):
        """The reported breakdown must be the leaf the PV actually reaches,
        not the root — otherwise the explanation describes a different
        state than the decision was made on."""
        r = plan(recon_state(), depth=4)
        leaf = recon_state()
        for move in r.pv:
            leaf = (
                apply_defender(leaf, move)
                if isinstance(move, DefenderPolicy)
                else apply_attacker(leaf, move)
            )
        self.assertEqual(r.components, components_from(leaf))

    def test_reported_depth_is_the_requested_depth(self):
        self.assertEqual(plan(recon_state(), depth=3).depth_searched, 3)

    def test_minimax_reports_no_prunes(self):
        self.assertEqual(plan(recon_state(), depth=4, algorithm="minimax").prunes, 0)

    def test_alphabeta_reports_prunes(self):
        self.assertGreater(plan(recon_state(), depth=4, algorithm="alphabeta").prunes, 0)


class TestMinimaxGuarantee(unittest.TestCase):
    """The property that makes this minimax and not just a tree walk."""

    def test_value_is_the_worst_case_over_attacker_replies(self):
        """Brute-forced at depth 2, where the whole tree is checkable.

        For the recommended policy, the reported value must equal the
        minimum over every legal attacker reply — and no reply may score
        below it.
        """
        w = UtilityWeights()
        s = recon_state()
        r = plan(s, depth=2, weights=w)

        after_policy = apply_defender(s, r.best_policy)
        replies = legal_attacker_actions(after_policy)
        self.assertTrue(replies)

        scores = [
            evaluate(apply_attacker(after_policy, a), w) for a in replies
        ]
        self.assertEqual(r.value, min(scores))
        for score in scores:
            self.assertGreaterEqual(score, r.value)

    def test_no_other_root_policy_beats_the_chosen_one(self):
        from cowrie.llm.planner.transition import legal_defender_policies

        w = UtilityWeights()
        s = recon_state()
        r = plan(s, depth=2, weights=w)
        for policy in legal_defender_policies(s):
            child = apply_defender(s, policy)
            worst = min(
                evaluate(apply_attacker(child, a), w)
                for a in legal_attacker_actions(child)
            )
            self.assertLessEqual(worst, r.value)


class TestBudgets(unittest.TestCase):
    def test_node_budget_truncates_and_says_so(self):
        r = plan(recon_state(), depth=4, algorithm="minimax", node_budget=50)
        self.assertIs(r.cutoff, CutoffStatus.NODE_BUDGET)
        self.assertIsNotNone(r.best_policy)

    def test_should_stop_truncates_and_says_so(self):
        r = plan(
            recon_state(),
            depth=4,
            algorithm="minimax",
            should_stop=lambda: True,
        )
        self.assertIs(r.cutoff, CutoffStatus.STOPPED)
        self.assertIsNotNone(r.best_policy)

    def test_a_truncated_search_still_returns_a_usable_move(self):
        """Budget exhaustion is a degraded answer, never a failure — the
        live path has to have something to execute."""
        for budget in (2, 5, 20, 200):
            r = plan(recon_state(), depth=4, node_budget=budget)
            self.assertIsInstance(r.best_policy, DefenderPolicy)

    def test_generous_budget_does_not_truncate(self):
        r = plan(recon_state(), depth=4, node_budget=10**6)
        self.assertIn(r.cutoff, (CutoffStatus.TERMINAL, CutoffStatus.DEPTH_LIMIT))

    def test_unbudgeted_search_is_bounded_by_the_tree(self):
        """7,129 is the depth-4 worst case with full 8x10 branching;
        legal-move masking keeps the real number well under it."""
        r = plan(recon_state(), depth=4, algorithm="minimax")
        self.assertLessEqual(r.nodes_expanded, 7129)
        self.assertGreater(r.nodes_expanded, 1)


class TestMaskRestrictsChoice(unittest.TestCase):
    def test_planner_only_recommends_permitted_policies(self):
        allowed = frozenset({P.DETERMINISTIC, P.DELAY_PRESSURE})
        r = plan(recon_state(), depth=4, allowed=allowed)
        self.assertIn(r.best_policy, allowed)

    def test_masking_never_recommends_an_illegal_policy(self):
        """An illegal move that scored well would poison the value even if
        the runtime later refused it, so masking happens at generation."""
        for policy in DefenderPolicy:
            r = plan(recon_state(), depth=3, allowed=frozenset({policy}))
            self.assertIn(r.best_policy, (policy, P.DELAY_PRESSURE))


class TestDirectedSanity(unittest.TestCase):
    def test_a_payload_transfer_gets_intercepted(self):
        """The one 'does it do the obviously right thing' check.

        When the attacker has just fetched a payload and capture is
        weighted heavily, interception should win — if it does not, the
        wiring between legality, transitions and scoring is broken in a way
        the structural tests would miss.
        """
        s = GameState().with_attack(A.PAYLOAD_TRANSFER)
        w = UtilityWeights.from_override("payload_capture=40.0")
        r = plan(s, depth=4, weights=w)
        self.assertIs(r.best_policy, P.DOWNLOADER_INTERCEPT)


if __name__ == "__main__":
    unittest.main()
