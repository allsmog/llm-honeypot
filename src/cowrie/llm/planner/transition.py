# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Move generation and the state transitions for both players.
# ABOUTME: Every number in the effect tables below is a HAND-TUNED GUESS,
# ABOUTME: not a measurement. What the search *proves* (alpha-beta ==
# ABOUTME: minimax) is independent of these numbers; what it *recommends*
# ABOUTME: is only as good as they are. Tests pin the signs, not magnitudes.

from __future__ import annotations

from dataclasses import replace

from cowrie.llm.planner.actions import (
    ATTACKER_MOVES,
    AttackerAction,
    AttackerObjective,
    DefenderPolicy,
)
from cowrie.llm.planner.state import GameState

A = AttackerAction
P = DefenderPolicy

# --------------------------------------------------------------------------
# Move generation
# --------------------------------------------------------------------------

#: Attacker actions the deterministic emulator can plausibly answer. Used
#: for *interior* legality only — at the root the real responder is probed.
_DETERMINISTIC_ANSWERABLE = frozenset(
    {
        A.OS_FINGERPRINT,
        A.FS_CONSISTENCY_PROBE,
        A.USER_PRIV_DISCOVERY,
        A.NETWORK_DISCOVERY,
        A.REPEAT_CONSISTENCY_CHECK,
    }
)

#: Where offering a decoy artifact makes sense at all.
_DECOY_RELEVANT = frozenset(
    {A.FS_CONSISTENCY_PROBE, A.USER_PRIV_DISCOVERY, A.NETWORK_DISCOVERY}
)

#: How many unsafe signals before hanging up is even considered. Mirrors
#: the runtime gate; kept here so the search never scores a move the
#: runtime would refuse.
UNSAFE_SIGNALS_FOR_TERMINATE = 2


def _interior_ok(state: GameState, policy: DefenderPolicy) -> bool:
    """Table-driven legality for a node with no concrete command.

    Interior nodes cannot probe the real subsystems — MIN played an
    abstract category, there is no command string to hand to
    responder.respond(). So interior legality is a pure function of state.
    The root additionally intersects this with runtime probe results.
    """
    last = state.last_attack

    if policy is P.DELAY_PRESSURE:
        # Unconditionally legal. This is load-bearing: it guarantees the
        # legal set is never empty for a non-terminal state, which is what
        # keeps a +/-inf sentinel from escaping as a node value.
        return True

    if policy is P.DOWNLOADER_INTERCEPT:
        return last is A.PAYLOAD_TRANSFER

    if policy is P.TERMINATE_SESSION:
        return last is A.EXIT or state.unsafe_events >= UNSAFE_SIGNALS_FOR_TERMINATE

    if policy is P.PERSONA_LLM:
        return _tokens_available(state)

    if policy is P.DECOY_DISCLOSURE:
        return _tokens_available(state) and last in _DECOY_RELEVANT

    if policy is P.DETERMINISTIC:
        return last in _DETERMINISTIC_ANSWERABLE

    if policy in (P.PLAUSIBLE_FAILURE, P.PERMISSION_DENIAL):
        # A refusal only makes sense in response to an attempt.
        return last is not A.EXIT

    return True


def _tokens_available(state: GameState) -> bool:
    if state.tokens_budget <= 0:
        return True  # unlimited
    return state.tokens_spent + state.est_tokens_per_llm_turn <= state.tokens_budget


def legal_defender_policies(
    state: GameState, allowed: frozenset[DefenderPolicy] | None = None
) -> tuple[DefenderPolicy, ...]:
    """MAX's moves, in deterministic enum order.

    ``allowed`` is the runtime mask from PolicyLegality at the root; None
    at interior nodes.

    Guaranteed non-empty for any non-terminal state. If the mask and the
    table disagree completely we fall back to DELAY_PRESSURE rather than
    returning (), because an empty move set makes the max loop return its
    -inf seed as a real node value — and both search implementations would
    do that identically, so the equivalence test would pass while the
    planner is corrupt.
    """
    if state.terminated:
        return ()
    moves = [p for p in DefenderPolicy if _interior_ok(state, p)]
    if allowed is not None:
        moves = [p for p in moves if p in allowed]
    if not moves:
        moves = [P.DELAY_PRESSURE]
    return tuple(moves)


def legal_attacker_actions(state: GameState) -> tuple[AttackerAction, ...]:
    """MIN's moves, in deterministic enum order.

    Note this returns a plain tuple, not (action, probability) pairs.
    Minimax takes a max over *damage*, so likelihood cannot change the
    result — only legality can. A log-derived prior would be dead weight
    here and false rigor besides. Switching to expectimax later changes
    this one return type and nothing else.

    UNKNOWN is never a move; it is a classifier output only.
    """
    if state.terminated:
        return ()
    if state.commands_remaining <= 0:
        # Out of budget: leaving is the only thing left.
        return (A.EXIT,)

    moves = []
    for a in ATTACKER_MOVES:
        if a is A.REPEAT_CONSISTENCY_CHECK and not state.recent_attacks:
            continue  # nothing has been asked yet, so nothing to re-ask
        if a is A.PAYLOAD_TRANSFER and state.last_attack is A.PAYLOAD_TRANSFER:
            continue  # don't model back-to-back drops
        moves.append(a)
    return tuple(moves)


# --------------------------------------------------------------------------
# Defender transitions
# --------------------------------------------------------------------------

#: Tokens charged for a policy that goes through the model.
def _llm_cost(state: GameState) -> int:
    return max(1, state.est_tokens_per_llm_turn)


def _repetition_penalty(state: GameState, policy: DefenderPolicy) -> float:
    """Emitting the same policy over and over is itself a fingerprint.

    A session where every command returns a near-identical plausible
    failure does not look like a Linux box, however good each individual
    failure string is. Grows with prior use of the same policy.
    """
    # HEURISTIC: 0.02 per prior use, so the 5th repeat costs 0.08.
    return 0.02 * state.policy_count_of(policy)


def apply_defender(state: GameState, policy: DefenderPolicy) -> GameState:
    """MAX moves. Returns a new state; ``state`` is never mutated."""
    s = state.with_policy(policy)
    last = state.last_attack
    fp = _repetition_penalty(state, policy)

    if policy is P.DETERMINISTIC:
        # Free, instant, and provably reproducible — so it *repays*
        # consistency debt rather than adding to it.
        s = replace(
            s,
            intel_value=s.intel_value + 0.05,
            consistency_debt=max(0.0, s.consistency_debt - 0.05),
            fingerprint_exposure=s.fingerprint_exposure
            + fp
            # HEURISTIC: a deterministic answer returns in microseconds. To
            # a timing probe that is *too* fast to be a real command.
            + (0.06 if last is A.TIMING_PROBE else 0.0),
        )

    elif policy is P.PLAUSIBLE_FAILURE:
        s = replace(
            s,
            engagement=s.engagement - 0.05,
            fingerprint_exposure=s.fingerprint_exposure + 0.02 + fp,
        )

    elif policy is P.PERMISSION_DENIAL:
        # Denial is *expected* when probing privilege, and it teaches us
        # what the attacker was reaching for.
        expected = last is A.USER_PRIV_DISCOVERY
        s = replace(
            s,
            engagement=s.engagement - 0.08,
            ttp_value=s.ttp_value + (0.1 if expected else 0.0),
            fingerprint_exposure=s.fingerprint_exposure
            + (0.0 if expected else 0.04)
            + fp,
        )

    elif policy is P.DELAY_PRESSURE:
        # Masks the LLM's latency signature when someone is timing us.
        masks = last is A.TIMING_PROBE
        s = replace(
            s,
            latency_cost=s.latency_cost + 0.15,
            engagement=s.engagement - 0.04,
            fingerprint_exposure=max(
                0.0, s.fingerprint_exposure + (-0.05 if masks else 0.0) + fp
            ),
        )

    elif policy is P.PERSONA_LLM:
        cost = _llm_cost(s)
        # Every free-form answer risks contradicting something we already
        # asserted. The risk scales with how much we owe.
        debt = 0.02 * s.obligation_pressure
        if last is A.REPEAT_CONSISTENCY_CHECK:
            # HEURISTIC: re-answering a probe from scratch is the single
            # most dangerous thing the model can do.
            debt = 0.10 + 0.02 * s.current_probe_risk
        s = replace(
            s,
            tokens_spent=s.tokens_spent + cost,
            intel_value=s.intel_value + 0.20,
            engagement=s.engagement + 0.10,
            consistency_debt=s.consistency_debt + debt,
            n_obligations_llm=s.n_obligations_llm + 1,
            fingerprint_exposure=s.fingerprint_exposure + fp,
        )

    elif policy is P.DECOY_DISCLOSURE:
        s = replace(
            s,
            tokens_spent=s.tokens_spent + _llm_cost(s),
            intel_value=s.intel_value + 0.15,
            engagement=s.engagement + 0.12,
            ttp_value=s.ttp_value + 0.2,
            n_obligations_world=s.n_obligations_world + 1,
            # HEURISTIC: a second convenient discovery starts to smell.
            fingerprint_exposure=s.fingerprint_exposure
            + (0.08 if state.policy_count_of(P.DECOY_DISCLOSURE) else 0.0)
            + fp,
        )

    elif policy is P.DOWNLOADER_INTERCEPT:
        s = replace(
            s,
            tokens_spent=s.tokens_spent + _llm_cost(s),
            payloads_captured=s.payloads_captured + 1,
            intel_value=s.intel_value + 0.4,
            ttp_value=s.ttp_value + 0.4,
            fingerprint_exposure=s.fingerprint_exposure + fp,
        )

    elif policy is P.TERMINATE_SESSION:
        # Stop-loss. No further cost accrues; prospects are frozen rather
        # than zeroed so terminal and depth-limited leaves stay on the
        # same scale (an artificial discount here would hand MIN a reason
        # to prefer lines that end the session).
        s = replace(s, terminated=True)

    return _clamp(s)


# --------------------------------------------------------------------------
# Attacker transitions
# --------------------------------------------------------------------------

_OBJECTIVE_OF = {
    A.OS_FINGERPRINT: AttackerObjective.RECON,
    A.FS_CONSISTENCY_PROBE: AttackerObjective.RECON,
    A.USER_PRIV_DISCOVERY: AttackerObjective.RECON,
    A.NETWORK_DISCOVERY: AttackerObjective.RECON,
    A.PAYLOAD_TRANSFER: AttackerObjective.PAYLOAD_DEPLOY,
    A.TOOL_ABUSE: AttackerObjective.PAYLOAD_DEPLOY,
    A.PERSISTENCE_ATTEMPT: AttackerObjective.PERSISTENCE,
    A.TIMING_PROBE: AttackerObjective.HONEYPOT_DETECTION,
    A.REPEAT_CONSISTENCY_CHECK: AttackerObjective.HONEYPOT_DETECTION,
}

#: Above this, an accumulated risk of contradiction becomes a realized one
#: when the attacker actually re-checks.
CONTRADICTION_THRESHOLD = 0.30


def _infer_objective(state: GameState) -> AttackerObjective:
    """Majority vote over the history window; UNKNOWN until we have 3."""
    if len(state.recent_attacks) < 3:
        return AttackerObjective.UNKNOWN
    tally: dict[AttackerObjective, int] = {}
    for a in state.recent_attacks:
        obj = _OBJECTIVE_OF.get(a)
        if obj is not None:
            tally[obj] = tally.get(obj, 0) + 1
    if not tally:
        return AttackerObjective.UNKNOWN
    # Ties break by AttackerObjective definition order, so this is
    # deterministic regardless of dict insertion order.
    best = max(tally.values())
    for obj in AttackerObjective:
        if tally.get(obj) == best:
            return obj
    return AttackerObjective.UNKNOWN


def apply_attacker(state: GameState, action: AttackerAction) -> GameState:
    """MIN moves. Returns a new state; ``state`` is never mutated."""
    if action is A.EXIT:
        # Freeze, do not zero. Terminal states must stay commensurate with
        # depth-limited ones or the search acquires a systematic bias for
        # or against ending the session. EXIT is also *last* in enum order,
        # so with strict-improvement selection a tie breaks away from it —
        # which is what keeps a zero-sum MIN from picking EXIT everywhere
        # and quietly collapsing depth 4 into depth 2.
        return replace(state.with_attack(action), terminated=True)

    s = state.with_attack(action)
    s = replace(s, commands_remaining=max(0, s.commands_remaining - 1))

    prev = state.prev_defender
    # Marginal novelty: the fifth `uname` teaches us less than the first.
    novelty = 1.0 / (1.0 + state.count_of(action))

    if action is A.REPEAT_CONSISTENCY_CHECK:
        realized = s.consistency_debt > CONTRADICTION_THRESHOLD
        s = replace(
            s,
            intel_value=s.intel_value + 0.05 * novelty,
            contradictions=s.contradictions + (1 if realized else 0),
            # A caught contradiction discharges the debt that produced it.
            consistency_debt=(
                max(0.0, s.consistency_debt - CONTRADICTION_THRESHOLD)
                if realized
                else s.consistency_debt
            ),
            fingerprint_exposure=s.fingerprint_exposure + (0.15 if realized else 0.0),
        )

    elif action is A.TIMING_PROBE:
        masked = prev is P.DELAY_PRESSURE
        s = replace(
            s,
            intel_value=s.intel_value + 0.05 * novelty,
            fingerprint_exposure=s.fingerprint_exposure + (0.0 if masked else 0.05),
        )

    elif action is A.PAYLOAD_TRANSFER:
        s = replace(
            s,
            intel_value=s.intel_value + 0.3 * novelty,
            ttp_value=s.ttp_value + 0.2,
        )

    elif action in (A.TOOL_ABUSE, A.PERSISTENCE_ATTEMPT):
        # Safety cost is charged to the defender line that permitted it —
        # that is what makes the unsafe_behavior weight actually steer
        # policy rather than being a constant everyone pays.
        permitted = prev in (P.PERSONA_LLM, P.DECOY_DISCLOSURE)
        s = replace(
            s,
            intel_value=s.intel_value + 0.2 * novelty,
            ttp_value=s.ttp_value + 0.3,
            unsafe_events=s.unsafe_events + (1 if permitted else 0),
        )

    else:
        s = replace(s, intel_value=s.intel_value + 0.1 * novelty)

    s = replace(s, objective=_infer_objective(s))
    return _clamp(s)


def _clamp(s: GameState) -> GameState:
    """Keep the bounded accumulators inside their ranges.

    Only the genuinely bounded quantities are clamped here. The cumulative
    ones (intel_value, ttp_value) are deliberately left unbounded and are
    softly saturated at evaluation time instead — a hard clamp would make
    every deep line score identically once it saturates, collapsing the
    search to enum order.
    """
    engagement = min(1.0, max(0.0, s.engagement))
    fingerprint = min(1.0, max(0.0, s.fingerprint_exposure))
    debt = max(0.0, s.consistency_debt)
    if (
        engagement == s.engagement
        and fingerprint == s.fingerprint_exposure
        and debt == s.consistency_debt
    ):
        return s
    return replace(
        s,
        engagement=engagement,
        fingerprint_exposure=fingerprint,
        consistency_debt=debt,
    )
