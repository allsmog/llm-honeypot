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

#: At or above this many unsafe signals, stop volunteering new material.
#:
#: `unsafe_events` counts things *we* did wrong — an SSRF-gate block, a
#: credential-shaped string in our own output, an observation leak — and is
#: supplied by the runtime at the root; no transition increments it. That
#: makes it constant within a single search, so as a utility term it cannot
#: change which policy wins. This legality gate is where it actually bites:
#: a session that has already leaked something stops being allowed to plant
#: decoys. It is also the guard against the planner learning to bait
#: dangerous behaviour now that attacker tool-abuse counts as a gain.
UNSAFE_SIGNALS_FOR_DISCLOSURE_HALT = 1


def _state_ok(state: GameState, policy: DefenderPolicy) -> bool:
    """Constraints the *runtime cannot know* — budgets and safety gates.

    Applied at every node, root included, because the protocol seam's
    legality probe answers "can this component render this command?" and
    has no view of the token budget or accumulated safety signals.
    """
    if policy is P.DELAY_PRESSURE:
        # Unconditionally legal. This is load-bearing: it guarantees the
        # legal set is never empty for a non-terminal state, which is what
        # keeps a +/-inf sentinel from escaping as a node value.
        return True

    if policy is P.TERMINATE_SESSION:
        return (
            state.last_attack is A.EXIT
            or state.unsafe_events >= UNSAFE_SIGNALS_FOR_TERMINATE
        )

    if policy is P.DECOY_DISCLOSURE:
        if state.unsafe_events >= UNSAFE_SIGNALS_FOR_DISCLOSURE_HALT:
            return False
        return _tokens_available(state)

    if policy in (P.PERSONA_LLM, P.DOWNLOADER_INTERCEPT):
        return _tokens_available(state)

    return True


def _shape_ok(state: GameState, policy: DefenderPolicy) -> bool:
    """Proxy for "could this component render the command?", by category.

    Interior nodes have no command string — MIN played an abstract
    category, and there is nothing to hand to responder.respond(). So away
    from the root we approximate availability from the attacker's action
    class.

    This is only ever a *proxy*, and at the root it must not be consulted:
    there we have the real command and the real probe result. Intersecting
    the runtime mask with this table used to strip DETERMINISTIC from
    `crontab -l` — the emulator answers it perfectly, but the command
    classifies as PERSISTENCE_ATTEMPT, which is not in the answerable set.
    """
    last = state.last_attack

    if policy is P.DOWNLOADER_INTERCEPT:
        return last is A.PAYLOAD_TRANSFER
    if policy is P.DECOY_DISCLOSURE:
        return last in _DECOY_RELEVANT
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

    At the root the mask is *authoritative* about which components can
    render the command, because it was built by actually calling them. It
    is intersected only with the state constraints the runtime cannot see
    (token budget, safety gates) — never with the category proxy in
    _shape_ok, which exists solely for nodes that have no command text.

    Guaranteed non-empty for any non-terminal state: if the mask and the
    state constraints leave nothing, we fall back to DELAY_PRESSURE rather
    than returning (), because an empty move set makes the max loop return
    its -inf seed as a real node value — and both search implementations
    would do that identically, so the equivalence test would pass while the
    planner is corrupt.
    """
    if state.terminated:
        return ()
    if allowed is not None:
        moves = [p for p in DefenderPolicy if p in allowed and _state_ok(state, p)]
    else:
        moves = [
            p for p in DefenderPolicy if _state_ok(state, p) and _shape_ok(state, p)
        ]
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


def _engage(engagement: float, k: float) -> float:
    """Move engagement toward 1.0 by a fraction of the remaining headroom.

    Deliberately not `min(1.0, e + k)`. A hard cap means that once a
    session is going well every engaging policy scores identically on this
    term, the search can no longer tell them apart, and whichever policy is
    cheapest wins by default. Diminishing returns keep the ordering strict
    at every level while staying inside (0, 1).
    """
    return engagement + (1.0 - engagement) * k


def _disengage(engagement: float, k: float) -> float:
    """Mirror of _engage: proportional loss, never below 0."""
    return engagement - engagement * k


def apply_defender(state: GameState, policy: DefenderPolicy) -> GameState:
    """MAX moves. Returns a new state; ``state`` is never mutated.

    NOTE ON INTEL: no defender policy grants intel_value. Intelligence is
    generated by what the *attacker* chose to do — their command is what
    reveals their capability and intent — not by which of our components
    rendered the reply. An earlier version credited DETERMINISTIC +0.05 and
    PERSONA_LLM +0.20, which asserts the model is four times more
    informative about the attacker than the emulator is. For any command
    both can answer that is simply false: the observable output is the
    same, and the LLM version is merely slower, costs tokens, and can
    contradict itself later. That mis-attribution made PERSONA_LLM beat
    DETERMINISTIC on every command, including `whoami`.

    So policies are separated only by what they genuinely differ on: cost,
    the attacker's willingness to keep going, fingerprint exposure,
    consistency risk, and what they capture.
    """
    s = state.with_policy(policy)
    last = state.last_attack
    fp = _repetition_penalty(state, policy)

    if policy is P.DETERMINISTIC:
        # Free, instant, and provably reproducible — so it *repays*
        # consistency debt rather than adding to it.
        s = replace(
            s,
            engagement=_engage(s.engagement, 0.06),
            consistency_debt=max(0.0, s.consistency_debt - 0.05),
            # HEURISTIC: a microsecond reply is slightly *too* instant, but
            # _show_prompt's jitter already covers most of that, so the
            # residual risk is small. Real bash answers in 1-10ms, which is
            # what this path most closely resembles.
            fingerprint_exposure=s.fingerprint_exposure
            + fp
            + (0.02 if last is A.TIMING_PROBE else 0.0),
        )

    elif policy is P.PLAUSIBLE_FAILURE:
        s = replace(
            s,
            engagement=_disengage(s.engagement, 0.10),
            # A refusal still tells us what they were reaching for.
            ttp_value=s.ttp_value + 0.05,
            fingerprint_exposure=s.fingerprint_exposure + 0.02 + fp,
        )

    elif policy is P.PERMISSION_DENIAL:
        # Denial is *expected* when probing privilege, and it teaches us
        # what the attacker was reaching for.
        expected = last is A.USER_PRIV_DISCOVERY
        s = replace(
            s,
            engagement=_disengage(s.engagement, 0.12),
            ttp_value=s.ttp_value + (0.10 if expected else 0.05),
            fingerprint_exposure=s.fingerprint_exposure
            + (0.0 if expected else 0.04)
            + fp,
        )

    elif policy is P.DELAY_PRESSURE:
        # Not a specialist for timing probes — a *fallback*. Stalling does
        # not mask the model's latency; it substitutes a third, equally
        # unnatural timing. Its real use is when nothing else is available:
        # the token budget is gone and the emulator cannot answer. On an
        # ordinary command it answers nothing at all, and a box that
        # randomly freezes is its own tell, so it is priced to lose
        # whenever a real answer is on the table.
        s = replace(
            s,
            latency_cost=s.latency_cost + 0.15,
            engagement=_disengage(s.engagement, 0.20),
            fingerprint_exposure=s.fingerprint_exposure + 0.06 + fp,
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
            engagement=_engage(s.engagement, 0.06),
            consistency_debt=s.consistency_debt + debt,
            n_obligations_llm=s.n_obligations_llm + 1,
            # HEURISTIC: the model round-trip takes 500-2000ms where real
            # bash takes 1-10ms. cowrie.cfg.dist calls this out as the
            # fork's headline timing tell, and it is exactly what someone
            # running a timing probe is measuring.
            fingerprint_exposure=s.fingerprint_exposure
            + fp
            + (0.08 if last is A.TIMING_PROBE else 0.0),
        )

    elif policy is P.DECOY_DISCLOSURE:
        s = replace(
            s,
            tokens_spent=s.tokens_spent + _llm_cost(s),
            engagement=_engage(s.engagement, 0.20),
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
            ttp_value=s.ttp_value + 0.4,
            engagement=_engage(s.engagement, 0.05),
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
        # An attacker deploying a miner or installing an authorized_keys
        # entry is the single most valuable thing that can happen here: it
        # is exactly the intelligence a honeypot exists to collect. Credit
        # it.
        #
        # A previous version charged this to whichever policy "permitted"
        # it — meaning an LLM reply was blamed for the attacker's next
        # command. That was both causally wrong (they run `chmod +x`
        # regardless of how we narrated the previous line) and
        # catastrophic in practice: at weight 10.0 the penalty exceeded
        # every possible gain, so MIN played tool abuse after every LLM
        # reply and PERSONA_LLM became unselectable. See
        # scripts/planner_diff.py.
        s = replace(
            s,
            intel_value=s.intel_value + 0.3 * novelty,
            ttp_value=s.ttp_value + 0.4,
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
