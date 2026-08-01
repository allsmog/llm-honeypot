# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the planner's move generation, transitions and scoring.

These pin *structure and sign*: that a policy is legal only when it makes
sense, that spending tokens increases spend, that masking a timing probe
lowers exposure rather than raising it.

They deliberately do NOT pin the magnitudes in transition.py's effect
tables. Those numbers are hand-tuned guesses that we expect to retune as
the planner meets real traffic; a test asserting `fingerprint == 0.06`
would just be a second copy of the constant, and would turn every honest
retune into a red build.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

from cowrie.llm.planner.actions import (
    ATTACKER_MOVES,
    AttackerAction,
    AttackerObjective,
    DefenderPolicy,
)
from cowrie.llm.planner.evaluate import components_from, evaluate
from cowrie.llm.planner.state import HISTORY_WINDOW, GameState, UtilityWeights
from cowrie.llm.planner.transition import (
    CONTRADICTION_THRESHOLD,
    apply_attacker,
    apply_defender,
    legal_attacker_actions,
    legal_defender_policies,
)

A = AttackerAction
P = DefenderPolicy


def state_after(*actions: AttackerAction, **kw) -> GameState:
    s = GameState(**kw)
    for a in actions:
        s = s.with_attack(a)
    return s


class TestActionSets(unittest.TestCase):
    def test_move_counts_match_the_spec(self):
        self.assertEqual(len(ATTACKER_MOVES), 10)
        self.assertEqual(len(DefenderPolicy), 8)

    def test_unknown_is_not_a_move(self):
        """UNKNOWN is a classifier output only.

        If it leaked into the move set the depth-4 tree would grow from
        7,129 to 8,545 nodes and every published node count would be wrong.
        """
        self.assertIn(A.UNKNOWN, list(AttackerAction))
        self.assertNotIn(A.UNKNOWN, ATTACKER_MOVES)

    def test_enum_values_are_unique(self):
        for enum in (AttackerAction, DefenderPolicy):
            values = [m.value for m in enum]
            self.assertEqual(len(values), len(set(values)))

    def test_exit_is_ordered_last(self):
        """Tie-breaking is first-in-enum-order, so putting EXIT last means
        an equal-valued continuation is preferred over ending the session.
        That is what keeps a zero-sum MIN from choosing EXIT everywhere and
        quietly collapsing a 4-ply search into a 2-ply one."""
        self.assertIs(ATTACKER_MOVES[-1], A.EXIT)


class TestImmutability(unittest.TestCase):
    def test_apply_defender_does_not_mutate_input(self):
        before = state_after(A.OS_FINGERPRINT)
        snapshot = replace(before)
        after = apply_defender(before, P.PERSONA_LLM)
        self.assertEqual(before, snapshot)
        self.assertIsNot(after, before)

    def test_apply_attacker_does_not_mutate_input(self):
        before = state_after(A.OS_FINGERPRINT)
        snapshot = replace(before)
        apply_attacker(before, A.PAYLOAD_TRANSFER)
        self.assertEqual(before, snapshot)


class TestDefenderLegality(unittest.TestCase):
    def test_delay_is_always_legal(self):
        """Load-bearing: it is what guarantees the legal set is non-empty,
        which is what stops a +/-inf seed escaping as a node value."""
        for action in ATTACKER_MOVES:
            s = state_after(action)
            self.assertIn(P.DELAY_PRESSURE, legal_defender_policies(s))

    def test_terminated_state_has_no_defender_moves(self):
        s = replace(state_after(A.OS_FINGERPRINT), terminated=True)
        self.assertEqual(legal_defender_policies(s), ())

    def test_intercept_requires_a_payload_transfer(self):
        yes = state_after(A.PAYLOAD_TRANSFER)
        no = state_after(A.OS_FINGERPRINT)
        self.assertIn(P.DOWNLOADER_INTERCEPT, legal_defender_policies(yes))
        self.assertNotIn(P.DOWNLOADER_INTERCEPT, legal_defender_policies(no))

    def test_llm_illegal_when_token_budget_would_be_exceeded(self):
        broke = state_after(
            A.OS_FINGERPRINT,
            tokens_budget=1000,
            tokens_spent=900,
            est_tokens_per_llm_turn=500,
        )
        rich = replace(broke, tokens_spent=0)
        self.assertNotIn(P.PERSONA_LLM, legal_defender_policies(broke))
        self.assertIn(P.PERSONA_LLM, legal_defender_policies(rich))

    def test_zero_budget_means_unlimited(self):
        s = state_after(A.OS_FINGERPRINT, tokens_budget=0, tokens_spent=10**9)
        self.assertIn(P.PERSONA_LLM, legal_defender_policies(s))

    def test_terminate_requires_exit_or_repeated_unsafe_signals(self):
        self.assertIn(
            P.TERMINATE_SESSION, legal_defender_policies(state_after(A.EXIT))
        )
        calm = state_after(A.OS_FINGERPRINT)
        self.assertNotIn(P.TERMINATE_SESSION, legal_defender_policies(calm))
        alarmed = replace(calm, unsafe_events=2)
        self.assertIn(P.TERMINATE_SESSION, legal_defender_policies(alarmed))

    def test_mask_narrows_but_never_empties(self):
        s = state_after(A.OS_FINGERPRINT)
        only_llm = legal_defender_policies(s, allowed=frozenset({P.PERSONA_LLM}))
        self.assertEqual(only_llm, (P.PERSONA_LLM,))
        impossible = legal_defender_policies(
            s, allowed=frozenset({P.DOWNLOADER_INTERCEPT})
        )
        self.assertEqual(impossible, (P.DELAY_PRESSURE,))


class TestAttackerLegality(unittest.TestCase):
    def test_exit_is_legal_even_at_zero_command_budget(self):
        s = state_after(A.OS_FINGERPRINT, commands_remaining=0)
        self.assertEqual(legal_attacker_actions(s), (A.EXIT,))

    def test_repeat_check_needs_something_to_repeat(self):
        self.assertNotIn(A.REPEAT_CONSISTENCY_CHECK, legal_attacker_actions(GameState()))
        self.assertIn(
            A.REPEAT_CONSISTENCY_CHECK,
            legal_attacker_actions(state_after(A.OS_FINGERPRINT)),
        )

    def test_no_back_to_back_payload_transfer(self):
        s = state_after(A.PAYLOAD_TRANSFER)
        self.assertNotIn(A.PAYLOAD_TRANSFER, legal_attacker_actions(s))

    def test_terminated_state_has_no_attacker_moves(self):
        s = replace(state_after(A.OS_FINGERPRINT), terminated=True)
        self.assertEqual(legal_attacker_actions(s), ())

    def test_unknown_never_offered(self):
        for s in (GameState(), state_after(A.OS_FINGERPRINT, A.TOOL_ABUSE)):
            self.assertNotIn(A.UNKNOWN, legal_attacker_actions(s))


class TestDefenderEffects(unittest.TestCase):
    def test_llm_spends_tokens(self):
        s = state_after(A.OS_FINGERPRINT, est_tokens_per_llm_turn=700)
        after = apply_defender(s, P.PERSONA_LLM)
        self.assertEqual(after.tokens_spent, s.tokens_spent + 700)

    def test_deterministic_is_free(self):
        s = state_after(A.OS_FINGERPRINT)
        self.assertEqual(apply_defender(s, P.DETERMINISTIC).tokens_spent, s.tokens_spent)

    def test_deterministic_repays_consistency_debt(self):
        s = replace(state_after(A.OS_FINGERPRINT), consistency_debt=0.5)
        self.assertLess(
            apply_defender(s, P.DETERMINISTIC).consistency_debt, s.consistency_debt
        )

    def test_llm_accrues_consistency_debt(self):
        s = state_after(A.OS_FINGERPRINT, n_obligations_llm=3)
        self.assertGreater(
            apply_defender(s, P.PERSONA_LLM).consistency_debt, s.consistency_debt
        )

    def test_delay_masks_a_timing_probe(self):
        s = state_after(A.TIMING_PROBE)
        delayed = apply_defender(s, P.DELAY_PRESSURE)
        instant = apply_defender(s, P.DETERMINISTIC)
        self.assertLess(delayed.fingerprint_exposure, instant.fingerprint_exposure)

    def test_denial_is_expected_when_probing_privilege(self):
        probing = state_after(A.USER_PRIV_DISCOVERY)
        unrelated = state_after(A.NETWORK_DISCOVERY)
        self.assertLess(
            apply_defender(probing, P.PERMISSION_DENIAL).fingerprint_exposure,
            apply_defender(unrelated, P.PERMISSION_DENIAL).fingerprint_exposure,
        )

    def test_intercept_captures_a_payload(self):
        s = state_after(A.PAYLOAD_TRANSFER)
        self.assertEqual(
            apply_defender(s, P.DOWNLOADER_INTERCEPT).payloads_captured,
            s.payloads_captured + 1,
        )

    def test_second_decoy_costs_more_exposure_than_the_first(self):
        s = state_after(A.FS_CONSISTENCY_PROBE)
        first = apply_defender(s, P.DECOY_DISCLOSURE)
        second = apply_defender(first.with_attack(A.FS_CONSISTENCY_PROBE), P.DECOY_DISCLOSURE)
        self.assertGreater(
            second.fingerprint_exposure - first.fingerprint_exposure,
            first.fingerprint_exposure - s.fingerprint_exposure,
        )

    def test_repeating_a_policy_raises_exposure(self):
        """A session where every answer is the same plausible failure does
        not look like a Linux box, however good the failure string is."""
        s = state_after(A.OS_FINGERPRINT)
        once = apply_defender(s, P.PLAUSIBLE_FAILURE)
        twice = apply_defender(once.with_attack(A.OS_FINGERPRINT), P.PLAUSIBLE_FAILURE)
        self.assertGreater(twice.fingerprint_exposure, once.fingerprint_exposure)

    def test_terminate_sets_the_flag(self):
        s = state_after(A.EXIT)
        self.assertTrue(apply_defender(s, P.TERMINATE_SESSION).terminated)


class TestAttackerEffects(unittest.TestCase):
    def test_exit_terminates_without_zeroing_prospects(self):
        """Freeze, do not zero.

        A terminal state has to stay commensurate with a depth-limited one,
        because alpha-beta compares them inside the same max/min. Zeroing
        engagement would give every terminal leaf a fixed artificial
        discount that has nothing to do with the decision being made.
        """
        s = replace(state_after(A.OS_FINGERPRINT), engagement=0.7, intel_value=1.2)
        after = apply_attacker(s, A.EXIT)
        self.assertTrue(after.terminated)
        self.assertEqual(after.engagement, 0.7)
        self.assertEqual(after.intel_value, 1.2)

    def test_commands_remaining_decrements(self):
        s = state_after(A.OS_FINGERPRINT, commands_remaining=5)
        self.assertEqual(apply_attacker(s, A.NETWORK_DISCOVERY).commands_remaining, 4)

    def test_repeat_check_realizes_a_contradiction_once_debt_is_high(self):
        risky = replace(
            state_after(A.OS_FINGERPRINT),
            consistency_debt=CONTRADICTION_THRESHOLD + 0.2,
        )
        safe = replace(state_after(A.OS_FINGERPRINT), consistency_debt=0.0)
        self.assertEqual(
            apply_attacker(risky, A.REPEAT_CONSISTENCY_CHECK).contradictions, 1
        )
        self.assertEqual(
            apply_attacker(safe, A.REPEAT_CONSISTENCY_CHECK).contradictions, 0
        )

    def test_realized_contradiction_discharges_the_debt(self):
        risky = replace(
            state_after(A.OS_FINGERPRINT),
            consistency_debt=CONTRADICTION_THRESHOLD + 0.2,
        )
        after = apply_attacker(risky, A.REPEAT_CONSISTENCY_CHECK)
        self.assertLess(after.consistency_debt, risky.consistency_debt)

    def test_unsafe_events_charged_to_the_policy_that_permitted_them(self):
        permissive = replace(state_after(A.OS_FINGERPRINT), prev_defender=P.PERSONA_LLM)
        refusing = replace(
            state_after(A.OS_FINGERPRINT), prev_defender=P.PERMISSION_DENIAL
        )
        self.assertEqual(apply_attacker(permissive, A.TOOL_ABUSE).unsafe_events, 1)
        self.assertEqual(apply_attacker(refusing, A.TOOL_ABUSE).unsafe_events, 0)

    def test_novelty_is_marginal_not_cumulative(self):
        """The fifth `uname` must be worth less than the first.

        A cumulative-then-clamped novelty term saturates mid-session, after
        which every deep line scores identically and the search degenerates
        to enum order.
        """
        fresh = GameState()
        first = apply_attacker(fresh, A.OS_FINGERPRINT)
        repeated = fresh
        for _ in range(4):
            repeated = apply_attacker(repeated, A.OS_FINGERPRINT)
        fifth = apply_attacker(repeated, A.OS_FINGERPRINT)
        first_gain = first.intel_value - fresh.intel_value
        fifth_gain = fifth.intel_value - repeated.intel_value
        self.assertGreater(first_gain, fifth_gain)

    def test_history_window_is_capped(self):
        s = GameState()
        for _ in range(HISTORY_WINDOW + 5):
            s = s.with_attack(A.OS_FINGERPRINT)
        self.assertEqual(len(s.recent_attacks), HISTORY_WINDOW)

    def test_objective_is_unknown_until_enough_history(self):
        s = state_after(A.OS_FINGERPRINT)
        self.assertIs(apply_attacker(s, A.OS_FINGERPRINT).objective,
                      AttackerObjective.UNKNOWN)

    def test_objective_infers_recon(self):
        s = state_after(A.OS_FINGERPRINT, A.NETWORK_DISCOVERY, A.USER_PRIV_DISCOVERY)
        self.assertIs(
            apply_attacker(s, A.FS_CONSISTENCY_PROBE).objective,
            AttackerObjective.RECON,
        )


class TestEvaluation(unittest.TestCase):
    def test_components_are_within_unit_range(self):
        extreme = GameState(
            intel_value=1e6,
            ttp_value=1e6,
            payloads_captured=10**6,
            consistency_debt=1e6,
            contradictions=10**6,
            unsafe_events=10**6,
            latency_cost=1e6,
            tokens_spent=10**9,
            engagement=1.0,
            fingerprint_exposure=1.0,
        )
        for name, value in components_from(extreme).as_dict().items():
            self.assertGreaterEqual(value, 0.0, name)
            self.assertLessEqual(value, 1.0, name)

    def test_evaluation_is_bounded_by_the_weight_sum(self):
        w = UtilityWeights()
        bound = w.bound()
        for s in (
            GameState(),
            GameState(intel_value=1e6, payloads_captured=99),
            GameState(unsafe_events=99, contradictions=99, latency_cost=1e6),
        ):
            self.assertLessEqual(abs(evaluate(s, w)), bound)

    def test_saturation_is_soft_so_deep_lines_stay_distinguishable(self):
        """A hard clamp would make these two score identically, and the
        search would lose the ability to prefer the better one."""
        big = GameState(intel_value=50.0)
        bigger = GameState(intel_value=100.0)
        self.assertLess(
            components_from(big).intel_novelty,
            components_from(bigger).intel_novelty,
        )

    def test_unsafe_behaviour_outweighs_intelligence(self):
        w = UtilityWeights()
        smart_but_unsafe = GameState(intel_value=1e6, ttp_value=1e6, unsafe_events=1)
        plain = GameState()
        self.assertLess(evaluate(smart_but_unsafe, w), evaluate(plain, w))

    def test_token_cost_registers_without_a_configured_budget(self):
        """max_tokens_per_session defaults to 0. The term must still charge
        for spend, or the planner has no reason to prefer a free answer."""
        free = GameState(tokens_budget=0, tokens_spent=0)
        pricey = GameState(tokens_budget=0, tokens_spent=30_000)
        self.assertGreater(
            components_from(pricey).token_cost, components_from(free).token_cost
        )

    def test_terminal_and_live_states_score_on_the_same_scale(self):
        live = GameState(intel_value=1.0, engagement=0.6)
        ended = replace(live, terminated=True)
        self.assertEqual(evaluate(live, UtilityWeights()), evaluate(ended, UtilityWeights()))


class TestWeights(unittest.TestCase):
    def test_empty_override_returns_defaults(self):
        self.assertEqual(UtilityWeights.from_override(""), UtilityWeights())
        self.assertEqual(UtilityWeights.from_override("   "), UtilityWeights())

    def test_valid_override_applies(self):
        w = UtilityWeights.from_override("contradictions=7.5, token_cost=0.25")
        self.assertEqual(w.contradictions, 7.5)
        self.assertEqual(w.token_cost, 0.25)
        self.assertEqual(w.engagement, UtilityWeights().engagement)

    def test_unknown_name_is_an_error_not_a_silent_noop(self):
        """A silently-ignored typo makes an experiment unreproducible in
        the worst possible way: it looks like it worked."""
        with self.assertRaises(ValueError) as ctx:
            UtilityWeights.from_override("contradiction=7.5")
        self.assertIn("unknown weight", str(ctx.exception))

    def test_non_finite_weights_are_rejected(self):
        """NaN makes every comparison False, so the search returns
        first-move garbage from *both* algorithms — equivalence would pass
        while the output is meaningless."""
        for bad in ("engagement=nan", "engagement=inf", "engagement=-inf"):
            with self.assertRaises(ValueError):
                UtilityWeights.from_override(bad)

    def test_non_numeric_is_rejected(self):
        with self.assertRaises(ValueError):
            UtilityWeights.from_override("engagement=high")

    def test_missing_equals_is_rejected(self):
        with self.assertRaises(ValueError):
            UtilityWeights.from_override("engagement")

    def test_errors_are_reported_together(self):
        with self.assertRaises(ValueError) as ctx:
            UtilityWeights.from_override("nope=1,engagement=nan")
        message = str(ctx.exception)
        self.assertIn("nope", message)
        self.assertIn("engagement", message)

    def test_bound_is_the_sum_of_magnitudes(self):
        w = UtilityWeights.from_override("engagement=-2.0")
        self.assertTrue(math.isfinite(w.bound()))
        self.assertGreater(w.bound(), 0.0)

    def test_fingerprint_is_stable_and_sensitive(self):
        self.assertEqual(UtilityWeights().fingerprint(), UtilityWeights().fingerprint())
        self.assertNotEqual(
            UtilityWeights().fingerprint(),
            UtilityWeights.from_override("engagement=9.0").fingerprint(),
        )
        self.assertEqual(len(UtilityWeights().fingerprint()), 12)


if __name__ == "__main__":
    unittest.main()
