# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Minimax and alpha-beta over response policies, plus the plan()
# ABOUTME: wrapper. The two searches are kept deliberately parallel — same
# ABOUTME: move order, same selection rule, same node counting — because
# ABOUTME: their provable equivalence is the point of the exercise. Read
# ABOUTME: test_llm_minimax_equivalence.py alongside this file.

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

from cowrie.llm.planner.actions import (
    AttackerAction,
    CutoffStatus,
    DefenderPolicy,
)
from cowrie.llm.planner.evaluate import components_from, evaluate
from cowrie.llm.planner.state import GameState, SearchResult, UtilityWeights
from cowrie.llm.planner.transition import (
    apply_attacker,
    apply_defender,
    legal_attacker_actions,
    legal_defender_policies,
)

INF = float("inf")

#: How often to consult should_stop. Checking every node costs more than it
#: saves; 64 keeps the worst-case overshoot negligible at our node counts.
_STOP_CHECK_INTERVAL = 64

DEFAULT_DEPTH = 4


class _Node(NamedTuple):
    """One backed-up result: value, the line that produced it, and why it
    stopped."""

    value: float
    pv: tuple[object, ...]
    cutoff: CutoffStatus


class _Ctx:
    """Per-search mutable bookkeeping.

    Separated from the recursion's return value so both algorithms can
    share one counting contract: *every* entry into a search function
    increments `nodes`, in both implementations, with no exceptions. An
    asymmetry here (say, alpha-beta also counting the evaluate() at a
    cutoff test) would break the per-pair node-dominance guarantee.
    """

    __slots__ = (
        "node_budget",
        "nodes",
        "order",
        "prunes",
        "should_stop",
        "stopped",
        "weights",
    )

    def __init__(
        self,
        weights: UtilityWeights,
        node_budget: int | None,
        should_stop: Callable[[], bool] | None,
        order: Callable[[GameState, tuple], tuple] | None = None,
    ) -> None:
        self.weights = weights
        self.nodes = 0
        self.prunes = 0
        self.node_budget = node_budget
        self.should_stop = should_stop
        self.stopped: CutoffStatus | None = None
        # Optional move-ordering hook. Alpha-beta prunes far more when good
        # moves are tried first, so this is the difference between a modest
        # saving and the textbook one. It changes node counts, never values.
        #
        # It CAN change which of two equal-valued moves is chosen, because
        # ties break first-in-order — so minimax and alpha-beta must share
        # the same ordering for root-move equality to hold.
        self.order = order

    def enter(self) -> None:
        self.nodes += 1
        if self.stopped is not None:
            return
        if self.node_budget is not None and self.nodes >= self.node_budget:
            self.stopped = CutoffStatus.NODE_BUDGET
        elif (
            self.should_stop is not None
            and self.nodes % _STOP_CHECK_INTERVAL == 0
            and self.should_stop()
        ):
            self.stopped = CutoffStatus.STOPPED


def _ordered(ctx: _Ctx, state: GameState, moves: tuple) -> tuple:
    """Apply the context's move ordering, if one was supplied.

    Both searches call this, so an ordering is always applied to both or
    to neither — which is what keeps their tie-breaking identical.
    """
    if ctx.order is None or len(moves) < 2:
        return moves
    return ctx.order(state, moves)


def _leaf(state: GameState, depth: int, ctx: _Ctx) -> _Node | None:
    """Terminal / horizon test, run *before* move generation.

    Doing this before generating moves is what keeps a +/-inf seed from
    ever surfacing as a node value. If a terminal state fell through to an
    empty move loop, the loop would return its seed, `+inf` would propagate
    upward, and TERMINATE_SESSION would score as infinitely good. Both
    search implementations would do that identically — so the equivalence
    test would pass while the planner was thoroughly broken.
    """
    if state.terminated:
        return _Node(evaluate(state, ctx.weights), (), CutoffStatus.TERMINAL)
    if depth <= 0:
        return _Node(evaluate(state, ctx.weights), (), CutoffStatus.DEPTH_LIMIT)
    return None


def minimax(
    state: GameState,
    depth: int,
    maximizing: bool,
    ctx: _Ctx,
    allowed: frozenset[DefenderPolicy] | None = None,
) -> _Node:
    """Plain minimax. The oracle: obviously correct, deliberately slow.

    This exists to be compared against alphabeta(), not to run on the live
    response path.
    """
    ctx.enter()
    leaf = _leaf(state, depth, ctx)
    if leaf is not None:
        return leaf

    if maximizing:
        moves = _ordered(ctx, state, legal_defender_policies(state, allowed))
        best = _Node(-INF, (), CutoffStatus.DEPTH_LIMIT)
        for i, policy in enumerate(moves):
            child = minimax(apply_defender(state, policy), depth - 1, False, ctx)
            # `i == 0 or` guarantees a move is always selected. Strict `>`
            # alone would leave best unassigned if every child scored -inf.
            if i == 0 or child.value > best.value:
                best = _Node(child.value, (policy, *child.pv), child.cutoff)
            if ctx.stopped is not None:
                break
        return best

    moves_a = _ordered(ctx, state, legal_attacker_actions(state))
    best = _Node(INF, (), CutoffStatus.DEPTH_LIMIT)
    for i, action in enumerate(moves_a):
        child = minimax(apply_attacker(state, action), depth - 1, True, ctx)
        if i == 0 or child.value < best.value:
            best = _Node(child.value, (action, *child.pv), child.cutoff)
        if ctx.stopped is not None:
            break
    return best


def alphabeta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    ctx: _Ctx,
    allowed: frozenset[DefenderPolicy] | None = None,
) -> _Node:
    """Alpha-beta. Same answer as minimax(), fewer nodes.

    Fail-soft: a cut node returns the bound it proved rather than the
    window edge. That is fine for our claims — at the *root* the window is
    (-inf, +inf), so no root-level pruning happens and the returned value
    is exact under either regime.

    Three invariants are load-bearing for root-move equality and must not
    be "optimized" away: alpha is raised to best-so-far between root
    children; there is no transposition table; there are no aspiration
    windows. Any of the three would break the proof.
    """
    ctx.enter()
    leaf = _leaf(state, depth, ctx)
    if leaf is not None:
        return leaf

    if maximizing:
        moves = _ordered(ctx, state, legal_defender_policies(state, allowed))
        best = _Node(-INF, (), CutoffStatus.DEPTH_LIMIT)
        for i, policy in enumerate(moves):
            child = alphabeta(
                apply_defender(state, policy), depth - 1, alpha, beta, False, ctx
            )
            if i == 0 or child.value > best.value:
                best = _Node(child.value, (policy, *child.pv), child.cutoff)
            if best.value > alpha:
                alpha = best.value
            if alpha >= beta:
                ctx.prunes += 1
                break
            if ctx.stopped is not None:
                break
        return best

    moves_a = _ordered(ctx, state, legal_attacker_actions(state))
    best = _Node(INF, (), CutoffStatus.DEPTH_LIMIT)
    for i, action in enumerate(moves_a):
        child = alphabeta(
            apply_attacker(state, action), depth - 1, alpha, beta, True, ctx
        )
        if i == 0 or child.value < best.value:
            best = _Node(child.value, (action, *child.pv), child.cutoff)
        if best.value < beta:
            beta = best.value
        if alpha >= beta:
            ctx.prunes += 1
            break
        if ctx.stopped is not None:
            break
    return best


def _replay(state: GameState, pv: tuple[object, ...]) -> GameState:
    """Walk the principal variation to reach the leaf it describes.

    Used only to recover the component breakdown for reporting. It runs
    outside the search and never touches the node counter — a replay that
    counted would break the per-pair node-dominance guarantee.
    """
    s = state
    for move in pv:
        if isinstance(move, DefenderPolicy):
            s = apply_defender(s, move)
        elif isinstance(move, AttackerAction):
            s = apply_attacker(s, move)
    return s


def plan(
    state: GameState,
    *,
    depth: int = DEFAULT_DEPTH,
    weights: UtilityWeights | None = None,
    algorithm: str = "alphabeta",
    allowed: frozenset[DefenderPolicy] | None = None,
    node_budget: int | None = None,
    should_stop: Callable[[], bool] | None = None,
    order: Callable[[GameState, tuple], tuple] | None = None,
) -> SearchResult:
    """Choose a response policy for an already-observed attacker command.

    The root is a MAX node: by the time we are called the attacker's
    command has happened and its effects are already folded into ``state``
    (``state.recent_attacks[-1]`` is the classified action). So depth=4
    means defend, attack, defend, attack.

    ``should_stop`` is the only way wall-clock time enters the search, and
    it is injected from the protocol seam. Every offline caller — tests,
    the bench, the evaluator — passes None and bounds work with
    ``node_budget`` instead, so their results are reproducible.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    w = weights or UtilityWeights()
    ctx = _Ctx(w, node_budget, should_stop, order)

    root_moves = legal_defender_policies(state, allowed)
    if state.terminated or depth == 0 or not root_moves:
        ctx.enter()
        cutoff = CutoffStatus.TERMINAL if state.terminated else CutoffStatus.DEPTH_LIMIT
        return SearchResult(
            best_policy=None,
            value=evaluate(state, w),
            components=components_from(state),
            pv=(),
            worst_case_reply=None,
            depth_searched=depth,
            nodes_expanded=ctx.nodes,
            cutoff=cutoff,
        )

    if algorithm == "minimax":
        node = minimax(state, depth, True, ctx, allowed)
    elif algorithm == "alphabeta":
        node = alphabeta(state, depth, -INF, INF, True, ctx, allowed)
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")

    # A sentinel that escaped as a node value means move generation
    # produced an empty set somewhere it shouldn't have. Fail loudly here
    # rather than let the caller act on an infinitely-good phantom move.
    if not math.isfinite(node.value):
        raise AssertionError(
            f"search returned non-finite value {node.value!r} — a +/-inf seed "
            "escaped move generation"
        )

    best_policy = node.pv[0] if node.pv else None
    reply = node.pv[1] if len(node.pv) > 1 else None
    return SearchResult(
        best_policy=best_policy if isinstance(best_policy, DefenderPolicy) else None,
        value=node.value,
        components=components_from(_replay(state, node.pv)),
        pv=node.pv,
        worst_case_reply=reply if isinstance(reply, AttackerAction) else None,
        depth_searched=depth,
        nodes_expanded=ctx.nodes,
        cutoff=ctx.stopped or node.cutoff,
        prunes=ctx.prunes,
    )


def new_context(
    weights: UtilityWeights,
    node_budget: int | None = None,
    should_stop: Callable[[], bool] | None = None,
    order: Callable[[GameState, tuple], tuple] | None = None,
) -> _Ctx:
    """Build a search context. For callers driving minimax()/alphabeta()
    directly — the equivalence test and the bench."""
    return _Ctx(weights, node_budget, should_stop, order)
