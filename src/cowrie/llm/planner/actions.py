# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: The two players' move sets for the minimax response planner.
# ABOUTME: MIN (the attacker) picks an abstract probe category; MAX (the
# ABOUTME: honeypot) picks a response policy. Enum *definition order* is
# ABOUTME: load-bearing: it is the deterministic tie-break order that makes
# ABOUTME: minimax and alpha-beta provably select the same root move.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique


@unique
class AttackerAction(Enum):
    """What the attacker does on their turn.

    Definition order is the MIN tie-break order, arranged recon-first so
    that natural enum order is also a serviceable move ordering for
    alpha-beta before any heuristic sorting is applied.

    UNKNOWN is a *classifier output only* — it is what classify() returns
    for input it cannot categorize. It is deliberately excluded from
    legal_attacker_actions() so the MIN branching factor stays 10, not 11.
    Including it would grow the depth-4 tree from 7,129 to 8,545 nodes and
    silently invalidate every published node count.
    """

    OS_FINGERPRINT = "os_fingerprint"
    FS_CONSISTENCY_PROBE = "fs_consistency_probe"
    USER_PRIV_DISCOVERY = "user_priv_discovery"
    TIMING_PROBE = "timing_probe"
    NETWORK_DISCOVERY = "network_discovery"
    PAYLOAD_TRANSFER = "payload_transfer"
    PERSISTENCE_ATTEMPT = "persistence_attempt"
    REPEAT_CONSISTENCY_CHECK = "repeat_consistency_check"
    TOOL_ABUSE = "tool_abuse"
    EXIT = "exit"

    # Not a move. See the class docstring.
    UNKNOWN = "unknown"


#: The 10 actions MIN may actually play. Frozen at import so callers cannot
#: accidentally widen the branching factor.
ATTACKER_MOVES: tuple[AttackerAction, ...] = tuple(
    a for a in AttackerAction if a is not AttackerAction.UNKNOWN
)


@unique
class DefenderPolicy(Enum):
    """How the honeypot responds.

    Definition order is the MAX tie-break order, cheapest-and-safest first,
    so that equal-valued policies resolve toward the boring option. That is
    a deliberate bias: when the search cannot tell two responses apart, we
    would rather emit the deterministic one than invent new text.
    """

    DETERMINISTIC = "deterministic"
    PLAUSIBLE_FAILURE = "plausible_failure"
    PERMISSION_DENIAL = "permission_denial"
    DELAY_PRESSURE = "delay_pressure"
    PERSONA_LLM = "persona_llm"
    DECOY_DISCLOSURE = "decoy_disclosure"
    DOWNLOADER_INTERCEPT = "downloader_intercept"
    TERMINATE_SESSION = "terminate_session"


@unique
class CutoffStatus(Enum):
    """Why the principal variation stopped where it did."""

    TERMINAL = "terminal"
    DEPTH_LIMIT = "depth"
    NODE_BUDGET = "node_budget"
    STOPPED = "stopped"


@unique
class AttackerObjective(Enum):
    """Coarse inference of what the attacker is here for."""

    UNKNOWN = "unknown"
    RECON = "recon"
    PAYLOAD_DEPLOY = "payload_deploy"
    PERSISTENCE = "persistence"
    HONEYPOT_DETECTION = "honeypot_detection"


@dataclass(frozen=True)
class PolicyLegality:
    """Which policies are physically available for a concrete command.

    Computed once per turn at the protocol seam by probing the real
    subsystems (can the deterministic responder handle this? is this a
    download?) and handed to the search as a mask over the root's legal
    moves.

    This exists because an illegal move that scored well would poison the
    minimax value even if the runtime later refused to execute it — so
    masking has to happen at move *generation*, not at dispatch.

    An empty ``allowed`` set means the planner declines the turn entirely
    and the caller runs its legacy path. That is the correct response when
    another subsystem owns the command (e.g. a full-screen program).
    """

    allowed: frozenset[DefenderPolicy] = frozenset(DefenderPolicy)

    def __bool__(self) -> bool:
        return bool(self.allowed)
