# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Human-facing benchmark for the response planner's search. Prints
# ABOUTME: the classic alpha-beta pruning table (nodes per depth, effective
# ABOUTME: branching factor) and the EXIT-degeneracy check that tells you
# ABOUTME: whether the depth knob is doing real work. Exits non-zero if
# ABOUTME: alpha-beta ever disagrees with the minimax oracle, so the demo
# ABOUTME: doubles as a smoke test.

from __future__ import annotations

import argparse
import random
import sys
import time

from cowrie.llm.planner.actions import ATTACKER_MOVES, AttackerAction, DefenderPolicy
from cowrie.llm.planner.evaluate import evaluate
from cowrie.llm.planner.search import alphabeta, minimax, new_context, plan
from cowrie.llm.planner.state import GameState, UtilityWeights
from cowrie.llm.planner.transition import apply_attacker, apply_defender

INF = float("inf")
DEFAULT_SEED = 20260731
DEFAULT_STATES = 60


def random_state(rng: random.Random) -> GameState:
    """Same shape as the equivalence test's generator, kept separate so a
    change here can never silently weaken the proof."""
    history = tuple(rng.choice(ATTACKER_MOVES) for _ in range(rng.randint(1, 6)))
    counts = [0] * len(AttackerAction)
    for a in history:
        counts[list(AttackerAction).index(a)] += 1
    return GameState(
        recent_attacks=history,
        action_counts=tuple(counts),
        policy_counts=tuple(rng.randint(0, 3) for _ in range(len(DefenderPolicy))),
        prev_defender=rng.choice([None, *DefenderPolicy]),
        n_obligations_llm=rng.randint(0, 6),
        n_obligations_det=rng.randint(0, 6),
        current_probe_risk=rng.choice([0.0, 0.1, 1.0]),
        commands_remaining=rng.randint(1, 200),
        tokens_spent=rng.randint(0, 20_000),
        tokens_budget=rng.choice([0, 10_000, 40_000]),
        intel_value=rng.uniform(0.0, 3.0),
        payloads_captured=rng.randint(0, 3),
        ttp_value=rng.uniform(0.0, 2.0),
        engagement=rng.uniform(0.0, 1.0),
        fingerprint_exposure=rng.uniform(0.0, 1.0),
        consistency_debt=rng.uniform(0.0, 1.5),
        contradictions=rng.randint(0, 3),
        unsafe_events=rng.randint(0, 2),
        latency_cost=rng.uniform(0.0, 1.5),
    )


def make_orderer(weights: UtilityWeights):
    """Best-first by a 1-ply lookahead — the cheapest useful move ordering.

    MAX tries the most promising policy first, MIN the most damaging reply
    first. Getting a good move in early raises alpha (or lowers beta)
    sooner, which is what lets the remaining siblings be cut. Ordering
    never changes the value; it only changes how many nodes it costs to
    find it.
    """

    def order(state: GameState, moves: tuple) -> tuple:
        first = moves[0]
        if isinstance(first, DefenderPolicy):
            return tuple(
                sorted(moves, key=lambda m: -evaluate(apply_defender(state, m), weights))
            )
        return tuple(
            sorted(moves, key=lambda m: evaluate(apply_attacker(state, m), weights))
        )

    return order


def run_depth(states: list[GameState], depth: int, weights: UtilityWeights) -> dict:
    """Search every state three ways and collect node counts and timings."""
    mm_nodes = ab_nodes = ord_nodes = 0
    mm_time = ab_time = 0.0
    mismatches = 0
    orderer = make_orderer(weights)

    for state in states:
        t0 = time.perf_counter()
        ctx_mm = new_context(weights)
        mm = minimax(state, depth, True, ctx_mm)
        mm_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        ctx_ab = new_context(weights)
        ab = alphabeta(state, depth, -INF, INF, True, ctx_ab)
        ab_time += time.perf_counter() - t0

        ctx_ord = new_context(weights, order=orderer)
        ordered = alphabeta(state, depth, -INF, INF, True, ctx_ord)

        mm_nodes += ctx_mm.nodes
        ab_nodes += ctx_ab.nodes
        ord_nodes += ctx_ord.nodes

        # Ordering must not change the value. It *may* change which of two
        # equal-valued root moves wins, since ties break first-in-order —
        # so only the value is compared for the ordered variant.
        if mm.value != ab.value or (mm.pv and ab.pv and mm.pv[0] != ab.pv[0]):
            mismatches += 1
        if mm.value != ordered.value:
            mismatches += 1

    n = len(states)
    return {
        "depth": depth,
        "mm_nodes": mm_nodes / n,
        "ab_nodes": ab_nodes / n,
        "ord_nodes": ord_nodes / n,
        "reduction": (mm_nodes / ab_nodes) if ab_nodes else float("nan"),
        "ord_reduction": (mm_nodes / ord_nodes) if ord_nodes else float("nan"),
        "mm_ms": 1000 * mm_time / n,
        "ab_ms": 1000 * ab_time / n,
        "mismatches": mismatches,
    }


def exit_degeneracy(states: list[GameState], weights: UtilityWeights) -> float:
    """Fraction of states where MIN's best reply is simply to leave.

    This is the honest health check on the whole premise. Under a zero-sum
    formulation, EXIT is optimal for MIN whenever continuing helps us — so
    if this number is high, plies 3 and 4 are never really consulted and a
    "4-ply search" is doing 2-ply work. Report it rather than let the depth
    setting imply depth that is not there.
    """
    exits = 0
    for state in states:
        r = plan(state, depth=4, weights=weights, algorithm="alphabeta")
        if r.worst_case_reply is AttackerAction.EXIT:
            exits += 1
    return exits / len(states) if states else 0.0


def render(rows: list[dict], exit_fraction: float) -> str:
    out = [
        "",
        "nodes expanded per state (mean), searching the same states three ways",
        "",
        f"{'depth':>5}  {'minimax':>9}  {'a-b':>9}  {'saving':>7}  "
        f"{'a-b+order':>10}  {'saving':>7}  {'eff. b':>7}  "
        f"{'mm ms':>7}  {'ab ms':>7}  {'agree':>6}",
        f"{'-' * 5}  {'-' * 9}  {'-' * 9}  {'-' * 7}  {'-' * 10}  "
        f"{'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 6}",
    ]
    for r in rows:
        eff_b = r["ord_nodes"] ** (1.0 / r["depth"]) if r["depth"] else float("nan")
        out.append(
            f"{r['depth']:>5}  {r['mm_nodes']:>9.1f}  {r['ab_nodes']:>9.1f}  "
            f"{r['reduction']:>6.2f}x  {r['ord_nodes']:>10.1f}  "
            f"{r['ord_reduction']:>6.2f}x  {eff_b:>7.2f}  {r['mm_ms']:>7.2f}  "
            f"{r['ab_ms']:>7.2f}  {'yes' if not r['mismatches'] else 'NO':>6}"
        )

    out += [
        "",
        "'a-b+order' tries the most promising move first (1-ply lookahead),",
        "so a good move raises alpha sooner and cuts more siblings. Ordering",
        "changes node counts, never values.",
        "",
        "'eff. b' is ordered-alpha-beta's nodes^(1/depth) — the branching",
        "factor it effectively searches. Falling as depth grows is the",
        "textbook b -> b^(3/4) -> b^(1/2) behaviour in real numbers.",
        "",
        f"MIN's best reply is EXIT in {exit_fraction:>.0%} of sampled states.",
    ]
    if exit_fraction > 0.9:
        out += [
            "",
            "  WARNING: a zero-sum MIN that almost always walks away means",
            "  plies 3-4 are rarely consulted — the depth setting is not",
            "  buying what it appears to. Say so in the README rather than",
            "  reporting a 4-ply search that behaves like a 2-ply one.",
        ]
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark alpha-beta against the minimax oracle."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--states", type=int, default=DEFAULT_STATES)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument(
        "--weights", default="", help="e.g. 'contradictions=4,token_cost=0.2'"
    )
    args = parser.parse_args()

    weights = UtilityWeights.from_override(args.weights)
    rng = random.Random(args.seed)
    states = [random_state(rng) for _ in range(args.states)]

    rows = [
        run_depth(states, d, weights) for d in range(1, args.max_depth + 1)
    ]
    print(render(rows, exit_degeneracy(states, weights)))

    total_mismatches = sum(r["mismatches"] for r in rows)
    if total_mismatches:
        print(
            f"FAIL: alpha-beta disagreed with the oracle on {total_mismatches} "
            "state/depth pairs.",
            file=sys.stderr,
        )
        return 1
    print("OK: alpha-beta matched the minimax oracle on every state and depth.")
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
