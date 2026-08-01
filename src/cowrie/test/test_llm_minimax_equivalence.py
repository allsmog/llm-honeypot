# SPDX-License-Identifier: BSD-3-Clause

"""The alpha-beta == minimax equivalence proof.

This is the week's centerpiece. It pins that the pruned search returns the
same value and the same root move as the plain minimax oracle, over a wide
sample of states and depths, while expanding strictly fewer nodes.

What this file deliberately does NOT assert, and why:

  * **Deep principal-variation equality.** Below a cutoff, alpha-beta's PV
    witnesses a *bound* — a line good enough to prove the node could not
    matter — not the minimax-optimal continuation. Asserting the two PVs
    match all the way down would be asserting something alpha-beta never
    promises, and the test would be pinning an implementation accident.
    Only pv[0] (the root move) is compared, and even that is really a
    plumbing check.

  * **That agreement implies correctness.** It does not. Both searches
    share move generation and evaluation, so a bug there produces two
    identically wrong answers and this file stays green. TestNoSentinelLeak
    below exists specifically because equality cannot catch that class of
    defect: an empty move set would return a +/-inf seed as a real node
    value, both algorithms would propagate it, and TERMINATE_SESSION would
    score as infinitely good with every equality assertion still passing.

Determinism note: no wall clock and no node budget are used anywhere in
this file. Both would make the two searches stop at different points and
turn the comparison into a race.
"""

from __future__ import annotations

import math
import random
import unittest

from cowrie.llm.planner.actions import ATTACKER_MOVES, AttackerAction, DefenderPolicy
from cowrie.llm.planner.search import alphabeta, minimax, new_context, plan
from cowrie.llm.planner.state import GameState, UtilityWeights

#: Fixed so failures are reproducible. Nothing else in the planner uses RNG.
SEED = 20260731
N_STATES = 150
DEPTHS = (1, 2, 3, 4)

INF = float("inf")


def random_state(rng: random.Random) -> GameState:
    """A non-terminal state drawn from the legal ranges.

    Accumulators are deliberately drawn past their nominal scales so the
    soft-saturation path in evaluate() gets exercised too.
    """
    history = tuple(
        rng.choice(ATTACKER_MOVES) for _ in range(rng.randint(1, 6))
    )
    counts = [0] * len(AttackerAction)
    for a in history:
        counts[list(AttackerAction).index(a)] += 1
    return GameState(
        persona_slug=rng.choice(["ubuntu_22_04", "alpine_3_19", "debian_12"]),
        recent_attacks=history,
        action_counts=tuple(counts),
        policy_counts=tuple(
            rng.randint(0, 3) for _ in range(len(DefenderPolicy))
        ),
        prev_defender=rng.choice([None, *DefenderPolicy]),
        n_obligations_llm=rng.randint(0, 6),
        n_obligations_det=rng.randint(0, 6),
        n_obligations_world=rng.randint(0, 3),
        current_probe_risk=rng.choice([0.0, 0.1, 1.0]),
        commands_remaining=rng.randint(1, 200),
        tokens_spent=rng.randint(0, 20_000),
        tokens_budget=rng.choice([0, 10_000, 40_000]),
        est_tokens_per_llm_turn=rng.randint(300, 1500),
        intel_value=rng.uniform(0.0, 3.0),
        payloads_captured=rng.randint(0, 3),
        ttp_value=rng.uniform(0.0, 2.0),
        engagement=rng.uniform(0.0, 1.0),
        fingerprint_exposure=rng.uniform(0.0, 1.0),
        consistency_debt=rng.uniform(0.0, 1.5),
        contradictions=rng.randint(0, 3),
        unsafe_events=rng.randint(0, 2),
        latency_cost=rng.uniform(0.0, 1.5),
        terminated=False,
    )


_CORPUS: list[GameState] | None = None
_RESULTS: dict[tuple[int, int, str], object] = {}


def _corpus() -> list[GameState]:
    global _CORPUS
    if _CORPUS is None:
        rng = random.Random(SEED)
        _CORPUS = [random_state(rng) for _ in range(N_STATES)]
    return _CORPUS


def _search(index: int, depth: int, algorithm: str):
    """Memoized plan() over the fixed corpus.

    Several tests need the same (state, depth, algorithm) result, and a
    depth-4 minimax sweep over 150 states is not cheap. Caching is safe
    precisely because the search is deterministic — which
    TestDeterminism verifies independently.
    """
    key = (index, depth, algorithm)
    hit = _RESULTS.get(key)
    if hit is None:
        hit = plan(_corpus()[index], depth=depth, algorithm=algorithm)
        _RESULTS[key] = hit
    return hit


class TestValueEquivalence(unittest.TestCase):
    """The core claim: same value, exactly."""

    def test_values_are_bit_identical(self):
        """assertEqual on floats, not assertAlmostEqual — and that is correct.

        At the root the window is (-inf, +inf), so alpha-beta returns the
        exact minimax value under both fail-soft and fail-hard regimes. Both
        implementations compute leaves with the same evaluate() on
        identically-reached frozen states, and back values up by pure
        max/min *selection* — no arithmetic is ever done on a propagated
        value. So the same float reaches the root by both routes.

        If this ever needs a tolerance, someone has introduced arithmetic
        into the backup step, and the loud failure is the point.
        """
        for i in range(N_STATES):
            for depth in DEPTHS:
                mm = _search(i, depth, "minimax")
                ab = _search(i, depth, "alphabeta")
                self.assertEqual(
                    mm.value,
                    ab.value,
                    f"state #{i} depth {depth}: {mm.value!r} != {ab.value!r}",
                )

    def test_root_move_is_identical(self):
        """Root-move equality is provable, not incidental.

        Root child i is searched with window (best_so_far, +inf). If its
        true value is <= best, alpha-beta returns something <= best and the
        strict `>` rejects it — exactly as minimax's true value would be
        rejected. If its true value is > best, the window is open above and
        the value comes back exact, so both accept. The accept/reject
        sequence is therefore identical, and first-strict-improvement picks
        the same index.

        This holds only because alpha is raised to best-so-far between root
        children, there is no transposition table, and there are no
        aspiration windows. All three are invariants of search.py.
        """
        for i in range(N_STATES):
            for depth in DEPTHS:
                self.assertEqual(
                    _search(i, depth, "minimax").best_policy,
                    _search(i, depth, "alphabeta").best_policy,
                    f"state #{i} depth {depth}",
                )

    def test_pv_head_matches_best_policy_and_length_is_bounded(self):
        """Weak by design — pv[0] tests plumbing, not the algorithm."""
        for i in range(40):
            for depth in DEPTHS:
                for algo in ("minimax", "alphabeta"):
                    r = _search(i, depth, algo)
                    self.assertEqual(r.pv[0], r.best_policy)
                    self.assertLessEqual(len(r.pv), depth)
                    # Alternating: even indices defend, odd indices attack.
                    for j, move in enumerate(r.pv):
                        expected = DefenderPolicy if j % 2 == 0 else AttackerAction
                        self.assertIsInstance(move, expected)


class TestNodeDominance(unittest.TestCase):
    """Pruning must never cost nodes, and should usually save many."""

    def test_alphabeta_never_expands_more_nodes(self):
        """Per-pair, not just on average.

        With identical move ordering and no transposition table,
        alpha-beta's visited set is a subset of minimax's: it can skip
        children, never add them. Equality happens on trees too small to
        prune.
        """
        for i in range(N_STATES):
            for depth in DEPTHS:
                mm = _search(i, depth, "minimax")
                ab = _search(i, depth, "alphabeta")
                self.assertLessEqual(
                    ab.nodes_expanded,
                    mm.nodes_expanded,
                    f"state #{i} depth {depth}: alpha-beta expanded MORE nodes",
                )

    def test_aggregate_reduction_is_substantial(self):
        """Empirical for this seed, not a theorem — say so rather than
        dressing a sample statistic as a proof."""
        for depth in (3, 4):
            mm_total = sum(
                _search(i, depth, "minimax").nodes_expanded for i in range(N_STATES)
            )
            ab_total = sum(
                _search(i, depth, "alphabeta").nodes_expanded for i in range(N_STATES)
            )
            self.assertLess(ab_total, mm_total)
            self.assertGreater(mm_total / ab_total, 1.5, f"depth {depth}")

    def test_pruning_actually_happens(self):
        """Guards against an alpha-beta that silently degraded to minimax."""
        prunes = sum(
            _search(i, 4, "alphabeta").prunes for i in range(N_STATES)
        )
        self.assertGreater(prunes, 0)


class TestMoveOrdering(unittest.TestCase):
    """Ordering changes cost, never the answer."""

    @staticmethod
    def _orderer(weights: UtilityWeights):
        from cowrie.llm.planner.evaluate import evaluate
        from cowrie.llm.planner.transition import apply_attacker, apply_defender

        def order(state, moves):
            if isinstance(moves[0], DefenderPolicy):
                key = lambda m: -evaluate(apply_defender(state, m), weights)  # noqa: E731
            else:
                key = lambda m: evaluate(apply_attacker(state, m), weights)  # noqa: E731
            return tuple(sorted(moves, key=key))

        return order

    def test_ordering_preserves_the_value(self):
        """The value is a property of the game tree, not of the order the
        tree happens to be walked in."""
        w = UtilityWeights()
        order = self._orderer(w)
        for i in range(40):
            for depth in DEPTHS:
                natural = _search(i, depth, "alphabeta")
                ordered = plan(
                    _corpus()[i], depth=depth, weights=w, algorithm="alphabeta", order=order
                )
                self.assertEqual(natural.value, ordered.value, f"state #{i} depth {depth}")

    def test_ordering_expands_fewer_nodes(self):
        w = UtilityWeights()
        order = self._orderer(w)
        natural = sum(_search(i, 4, "alphabeta").nodes_expanded for i in range(40))
        ordered = sum(
            plan(
                _corpus()[i], depth=4, weights=w, algorithm="alphabeta", order=order
            ).nodes_expanded
            for i in range(40)
        )
        self.assertLess(ordered, natural)

    def test_both_searches_share_one_ordering_hook(self):
        """Root-move equality requires identical move order in both
        algorithms, so the hook has to reach minimax too — not just the
        implementation we care about making fast."""
        w = UtilityWeights()
        order = self._orderer(w)
        for i in range(20):
            mm = plan(_corpus()[i], depth=3, weights=w, algorithm="minimax", order=order)
            ab = plan(
                _corpus()[i], depth=3, weights=w, algorithm="alphabeta", order=order
            )
            self.assertEqual(mm.value, ab.value)
            self.assertEqual(mm.best_policy, ab.best_policy)


class TestNoSentinelLeak(unittest.TestCase):
    """The class of bug that value-equality cannot catch.

    Both searches seed their loops with -inf/+inf. If move generation ever
    returns an empty tuple for a non-terminal state, the seed becomes the
    node's value and propagates. Both algorithms agree on the garbage, so
    every assertion in TestValueEquivalence still passes.
    """

    def test_returned_values_are_finite_and_bounded(self):
        bound = UtilityWeights().bound()
        for i in range(N_STATES):
            for depth in DEPTHS:
                for algo in ("minimax", "alphabeta"):
                    r = _search(i, depth, algo)
                    self.assertTrue(math.isfinite(r.value))
                    self.assertLessEqual(abs(r.value), bound)

    def test_defender_moves_never_empty_for_live_state(self):
        from cowrie.llm.planner.transition import legal_defender_policies

        for state in _corpus():
            self.assertTrue(legal_defender_policies(state))

    def test_attacker_moves_never_empty_for_live_state(self):
        from cowrie.llm.planner.transition import legal_attacker_actions

        for state in _corpus():
            self.assertTrue(legal_attacker_actions(state))

    def test_empty_mask_falls_back_rather_than_returning_nothing(self):
        from cowrie.llm.planner.transition import legal_defender_policies

        moves = legal_defender_policies(_corpus()[0], allowed=frozenset())
        self.assertEqual(moves, (DefenderPolicy.DELAY_PRESSURE,))


class TestDeterminism(unittest.TestCase):
    """No wall clock, no RNG, no dict-ordering dependence in the search."""

    def test_repeat_search_is_identical(self):
        for state in _corpus()[:30]:
            for algo in ("minimax", "alphabeta"):
                a = plan(state, depth=4, algorithm=algo)
                b = plan(state, depth=4, algorithm=algo)
                self.assertEqual(a.value, b.value)
                self.assertEqual(a.best_policy, b.best_policy)
                self.assertEqual(a.pv, b.pv)
                self.assertEqual(a.nodes_expanded, b.nodes_expanded)
                self.assertEqual(a.prunes, b.prunes)

    def test_search_never_reads_the_clock(self):
        """Pins that time cannot leak back into the pure search.

        The live path bounds work with an injected should_stop closure;
        every offline caller passes None. If someone reintroduces a
        time.monotonic() call inside the search, this fails.
        """
        import time as time_module

        original = time_module.monotonic

        def explode() -> float:
            raise AssertionError("the search must not read the clock")

        time_module.monotonic = explode
        try:
            plan(_corpus()[0], depth=4, algorithm="alphabeta")
            plan(_corpus()[0], depth=4, algorithm="minimax")
        finally:
            time_module.monotonic = original


class TestDirectDrive(unittest.TestCase):
    """The bench drives minimax()/alphabeta() directly; keep that working."""

    def test_direct_calls_agree_with_plan(self):
        state = _corpus()[0]
        w = UtilityWeights()

        ctx_mm = new_context(w)
        mm = minimax(state, 4, True, ctx_mm)
        ctx_ab = new_context(w)
        ab = alphabeta(state, 4, -INF, INF, True, ctx_ab)

        self.assertEqual(mm.value, ab.value)
        self.assertEqual(mm.pv[0], ab.pv[0])
        self.assertLessEqual(ctx_ab.nodes, ctx_mm.nodes)
        self.assertEqual(mm.value, plan(state, depth=4, algorithm="minimax").value)


if __name__ == "__main__":
    unittest.main()
