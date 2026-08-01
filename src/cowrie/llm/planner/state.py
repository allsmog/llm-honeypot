# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: The search's state, scoring weights, and result types. GameState
# ABOUTME: is frozen and the search copies-on-apply: at ~7k nodes of ~20
# ABOUTME: scalar fields that costs microseconds, and immutability is what
# ABOUTME: makes the minimax/alpha-beta equivalence proof airtight — neither
# ABOUTME: algorithm can observe the other's mutations.

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from cowrie.llm.planner.actions import (
    AttackerAction,
    AttackerObjective,
    DefenderPolicy,
)

#: Position of each action/policy in the count tuples below. Built once at
#: import; the tuples are positional so GameState stays hashable and cheap
#: to copy (a dict field would make replace() allocate a new mapping).
_ATTACK_INDEX: dict[AttackerAction, int] = {a: i for i, a in enumerate(AttackerAction)}
_POLICY_INDEX: dict[DefenderPolicy, int] = {p: i for i, p in enumerate(DefenderPolicy)}

_N_ATTACKS = len(AttackerAction)
_N_POLICIES = len(DefenderPolicy)

#: How many recent attacker actions the state remembers. Six is enough to
#: see a probe/re-probe pair with a few commands between them.
HISTORY_WINDOW = 6


def _bump(counts: tuple[int, ...], index: int) -> tuple[int, ...]:
    """Return ``counts`` with position ``index`` incremented."""
    out = list(counts)
    out[index] += 1
    return tuple(out)


@dataclass(frozen=True)
class GameState:
    """Everything the search reasons over. Frozen; transitions return copies.

    Note what is *not* here: no command strings, no persona object, no
    WorldState, no wall clock. Interior search nodes have no command text —
    MIN plays abstract categories — so anything derived from a literal
    command must be computed at the root and folded in before plan() runs.
    """

    # --- identity -------------------------------------------------------
    persona_slug: str = ""

    # --- history --------------------------------------------------------
    #: Most recent first-to-last; [-1] is the observed command at the root.
    recent_attacks: tuple[AttackerAction, ...] = ()
    #: Per-action occurrence counts, positional by AttackerAction order.
    #: Drives *marginal* novelty: the fifth `uname` is worth less than the
    #: first. Cumulative-then-clamped novelty saturates mid-session and
    #: collapses the search to enum order; marginal novelty does not.
    action_counts: tuple[int, ...] = field(default=(0,) * _N_ATTACKS)
    #: Per-policy counts, positional by DefenderPolicy order. Feeds the
    #: repetition penalty — a session where every answer is the same
    #: plausible failure is its own fingerprint.
    policy_counts: tuple[int, ...] = field(default=(0,) * _N_POLICIES)
    prev_defender: DefenderPolicy | None = None
    objective: AttackerObjective = AttackerObjective.UNKNOWN

    # --- consistency obligations, bucketed by risk ----------------------
    # Deliberately counts, not family names. Interior nodes have no command
    # text to name a family with, and a frozenset[str] cannot carry the
    # source-dependent risk that the whole contradiction term depends on.
    # The PlannerSessionState -> GameState projection is the only bridge.
    n_obligations_llm: int = 0
    n_obligations_det: int = 0
    n_obligations_world: int = 0
    #: Risk of the obligation the *observed* command re-probes, if any.
    #: Root-only; interior nodes inherit it unchanged.
    current_probe_risk: float = 0.0

    # --- budgets --------------------------------------------------------
    commands_remaining: int = 200
    tokens_spent: int = 0
    #: 0 means unlimited, matching [llm] max_tokens_per_session. When it is
    #: 0 the token-cost component is inert — say so rather than pretending
    #: the search is optimizing spend it cannot see.
    tokens_budget: int = 0
    #: Estimated cost of one LLM turn, from this session's observed usage.
    est_tokens_per_llm_turn: int = 900

    # --- accumulators ---------------------------------------------------
    intel_value: float = 0.0
    payloads_captured: int = 0
    ttp_value: float = 0.0
    engagement: float = 0.5
    fingerprint_exposure: float = 0.0
    consistency_debt: float = 0.0
    contradictions: int = 0
    unsafe_events: int = 0
    latency_cost: float = 0.0

    # --- termination ----------------------------------------------------
    #: Checked *before* move generation, so a terminal node never reaches an
    #: empty move loop and never returns a +/-inf sentinel as a node value.
    terminated: bool = False

    # -- derived helpers -------------------------------------------------

    def count_of(self, action: AttackerAction) -> int:
        return self.action_counts[_ATTACK_INDEX[action]]

    def policy_count_of(self, policy: DefenderPolicy) -> int:
        return self.policy_counts[_POLICY_INDEX[policy]]

    def with_attack(self, action: AttackerAction) -> GameState:
        """Append ``action`` to the history window and bump its count."""
        window = (*self.recent_attacks, action)[-HISTORY_WINDOW:]
        return replace(
            self,
            recent_attacks=window,
            action_counts=_bump(self.action_counts, _ATTACK_INDEX[action]),
        )

    def with_policy(self, policy: DefenderPolicy) -> GameState:
        return replace(
            self,
            prev_defender=policy,
            policy_counts=_bump(self.policy_counts, _POLICY_INDEX[policy]),
        )

    @property
    def last_attack(self) -> AttackerAction | None:
        return self.recent_attacks[-1] if self.recent_attacks else None

    @property
    def obligation_pressure(self) -> float:
        """Weighted count of facts we owe consistency on.

        LLM-sourced facts are the risky ones: we deliberately do not parse
        model output, so we cannot reproduce them on demand. Deterministic
        and persona-pinned facts are re-derivable and near-free.
        """
        return (
            1.0 * self.n_obligations_llm
            + 0.1 * self.n_obligations_world
            + 0.0 * self.n_obligations_det
        )


@dataclass(frozen=True)
class UtilityComponents:
    """The ten scored terms, each already normalized to [0, 1].

    Positive and negative contributions are stored as positive magnitudes;
    ``total`` applies the signs. That keeps the logged breakdown readable —
    a fingerprint_exposure of 0.4 means "40% of the way to maximally
    exposed", not "-0.4 of something".
    """

    intel_novelty: float
    engagement: float
    payload_capture: float
    ttp_value: float
    persona_consistency: float
    fingerprint_exposure: float
    contradictions: float
    unsafe_behavior: float
    latency: float
    token_cost: float

    def total(self, w: UtilityWeights) -> float:
        return (
            w.intel_novelty * self.intel_novelty
            + w.engagement * self.engagement
            + w.payload_capture * self.payload_capture
            + w.ttp_value * self.ttp_value
            + w.persona_consistency * self.persona_consistency
            - w.fingerprint_exposure * self.fingerprint_exposure
            - w.contradictions * self.contradictions
            - w.unsafe_behavior * self.unsafe_behavior
            - w.latency * self.latency
            - w.token_cost * self.token_cost
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "intel_novelty": self.intel_novelty,
            "engagement": self.engagement,
            "payload_capture": self.payload_capture,
            "ttp_value": self.ttp_value,
            "persona_consistency": self.persona_consistency,
            "fingerprint_exposure": self.fingerprint_exposure,
            "contradictions": self.contradictions,
            "unsafe_behavior": self.unsafe_behavior,
            "latency": self.latency,
            "token_cost": self.token_cost,
        }


_WEIGHT_NAMES = (
    "intel_novelty",
    "engagement",
    "payload_capture",
    "ttp_value",
    "persona_consistency",
    "fingerprint_exposure",
    "contradictions",
    "unsafe_behavior",
    "latency",
    "token_cost",
)


@dataclass(frozen=True)
class UtilityWeights:
    """Relative importance of each term. Hand-tuned, not learned.

    These numbers are the least defensible part of the planner and the
    docstring should stay honest about it: what the search *proves*
    (alpha-beta == minimax) is independent of them; what it *recommends* is
    only as good as they are.

    unsafe_behavior dominates deliberately — near-lexicographic, so no
    amount of intelligence yield buys an unsafe response.
    """

    intel_novelty: float = 3.0
    engagement: float = 2.0
    payload_capture: float = 4.0
    ttp_value: float = 2.0
    persona_consistency: float = 2.0
    fingerprint_exposure: float = 3.0
    contradictions: float = 3.0
    unsafe_behavior: float = 10.0
    latency: float = 0.5
    token_cost: float = 1.0

    def bound(self) -> float:
        """Max |utility| given every component lies in [0, 1].

        The +/-inf sentinels used to seed the search must strictly dominate
        this, or a real leaf could tie with a sentinel and be mistaken for
        "no move found".
        """
        return sum(abs(float(getattr(self, n))) for n in _WEIGHT_NAMES)

    @classmethod
    def from_override(cls, spec: str) -> UtilityWeights:
        """Parse ``"contradictions=-3.0,token_cost=0.8"``.

        Unknown names and non-finite values are errors, never silent
        no-ops. A typo'd weight that is quietly ignored makes an experiment
        unreproducible in the worst possible way: it looks like it worked.
        And a NaN weight makes every comparison False, so the search
        silently returns first-move garbage from *both* algorithms — the
        equivalence test would pass while the output is meaningless.
        """
        if not spec or not spec.strip():
            return cls()
        values: dict[str, float] = {}
        errors: list[str] = []
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            name, sep, raw = chunk.partition("=")
            name = name.strip()
            if not sep:
                errors.append(f"{chunk!r}: expected name=value")
                continue
            if name not in _WEIGHT_NAMES:
                errors.append(f"{name!r}: unknown weight (known: {', '.join(_WEIGHT_NAMES)})")
                continue
            try:
                value = float(raw.strip())
            except ValueError:
                errors.append(f"{name}: {raw.strip()!r} is not a number")
                continue
            if not math.isfinite(value):
                errors.append(f"{name}: {value} is not finite")
                continue
            values[name] = value
        if errors:
            raise ValueError("invalid planner_weights:\n  - " + "\n  - ".join(errors))
        return replace(cls(), **values)

    def fingerprint(self) -> str:
        """Stable short hash of the weight vector.

        Logged with every decision so a JSON line is self-describing: given
        the log you can reconstruct the exact weights and re-run the search.
        """
        import hashlib

        payload = ",".join(f"{n}={getattr(self, n):.6g}" for n in _WEIGHT_NAMES)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class SearchResult:
    """What plan() hands back — the decision plus its whole justification."""

    best_policy: DefenderPolicy | None
    value: float
    components: UtilityComponents
    #: Root-first, alternating defender/attacker.
    pv: tuple[object, ...]
    worst_case_reply: AttackerAction | None
    depth_searched: int
    nodes_expanded: int
    cutoff: object
    #: Alpha-beta cutoffs taken. Always 0 for the plain minimax oracle.
    prunes: int = 0
