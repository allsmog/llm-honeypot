# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Scores a GameState into ten normalized components and a scalar.
# ABOUTME: One evaluator serves both terminal and depth-limited leaves, on
# ABOUTME: purpose: alpha-beta compares the two kinds of leaf inside the
# ABOUTME: same max/min, so scoring them on different scales would make the
# ABOUTME: comparison meaningless.

from __future__ import annotations

from cowrie.llm.planner.state import GameState, UtilityComponents, UtilityWeights

# Soft-saturation scales. Each is "the value at which this term reaches
# half its maximum" — see _soft below.
_K_INTEL = 2.0
_K_PAYLOAD = 1.0
_K_TTP = 1.5
_K_DEBT = 0.5
_K_CONTRADICTION = 1.0
_K_UNSAFE = 0.5
_K_LATENCY = 1.0

#: When no token budget is configured, cost is measured against this
#: nominal session size instead. Roughly a 20-turn LLM-heavy session.
NOMINAL_TOKEN_SCALE = 18_000


def _soft(x: float, k: float) -> float:
    """Map [0, inf) into [0, 1) as x/(x+k). Strictly increasing everywhere.

    Deliberately not min(1.0, x/k). A hard clamp saturates: once two lines
    both exceed the cap they score identically on this term, the search
    loses its ability to tell them apart, and the choice degenerates to
    whichever policy comes first in enum order. Soft saturation keeps every
    comparison meaningful at any depth while staying bounded — which is
    what the +/-inf sentinels need in order to strictly dominate.
    """
    if x <= 0.0:
        return 0.0
    return x / (x + k)


def components_from(state: GameState) -> UtilityComponents:
    """Project a state onto the ten scored terms, each in [0, 1].

    Terminal states go through this unchanged. Their prospect terms are
    whatever they were when the session ended — frozen, not zeroed. Zeroing
    would give every terminal leaf a fixed artificial discount, which reads
    to the search as a reason to prefer (or avoid) ending the session that
    has nothing to do with the actual decision.
    """
    if state.tokens_budget > 0:
        token_cost = min(1.0, state.tokens_spent / state.tokens_budget)
    else:
        # No configured cap. Still charge for spend, against a nominal
        # session, so the search has some reason to prefer a free answer.
        token_cost = _soft(state.tokens_spent, NOMINAL_TOKEN_SCALE)

    return UtilityComponents(
        intel_novelty=_soft(state.intel_value, _K_INTEL),
        engagement=state.engagement,
        payload_capture=_soft(state.payloads_captured, _K_PAYLOAD),
        ttp_value=_soft(state.ttp_value, _K_TTP),
        persona_consistency=1.0 - _soft(state.consistency_debt, _K_DEBT),
        fingerprint_exposure=state.fingerprint_exposure,
        contradictions=_soft(state.contradictions, _K_CONTRADICTION),
        unsafe_behavior=_soft(state.unsafe_events, _K_UNSAFE),
        latency=_soft(state.latency_cost, _K_LATENCY),
        token_cost=token_cost,
    )


def evaluate(state: GameState, weights: UtilityWeights) -> float:
    """Scalar utility of a leaf. |result| is bounded by weights.bound()."""
    return components_from(state).total(weights)
