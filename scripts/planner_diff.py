#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Compares what the minimax planner WOULD choose against what the
# ABOUTME: existing if-ladder actually does, command by command, over the
# ABOUTME: attacker_sim patterns. Needs no LLM, no honeypot and no corpus.
# ABOUTME: Exists because unit tests pin the sign of each individual effect
# ABOUTME: and cannot see that one policy is quietly winning every decision.

"""Ladder-vs-planner decision diff, and a gate against degenerate tuning.

Two real defects reached a fully green 805-test suite before this tool
existed, both invisible to unit tests and both caught here in one run:

  1. `unsafe_events` was charged to the defender whenever the attacker
     later ran a tool-abuse command. At weight 10.0 that exceeded the
     largest possible gain, so PERSONA_LLM was dominated and the planner
     answered every non-deterministic command by pausing and printing a
     prompt -- emitting nothing at all.

  2. Correcting (1) by crediting attacker tool-abuse flipped the
     domination the other way: PERSONA_LLM won 61 of 67 decisions,
     including `whoami`, which the local emulator answers instantly and
     for free.

Both are the same shape: one policy wins almost everything, so the search
is not discriminating on merits. That shape is what this tool fails on.

What a healthy result looks like: HIGH agreement with the ladder. On
recon-heavy traffic the ladder is close to optimal, so a correctly tuned
planner mostly agrees with it -- DETERMINISTIC when the emulator can
answer, PERSONA_LLM when it cannot. Treat a LOW agreement rate as a bug
signal, not as evidence the planner is adding value.

The corollary is worth stating plainly: at 100% agreement the planner is
choosing exactly what the ladder already chooses, so on this corpus it
adds nothing. Its value has to come from decisions this corpus does not
contain -- token-budget pressure, a re-probe needing replay steering, an
accumulated safety signal -- and none of those are exercised here.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ruff: noqa: E402 — these must follow the sys.path insert above so the
# script runs from a checkout without the package being installed.
from cowrie.llm import attack_map, downloader, interactive, persona, responder
from cowrie.llm.planner import (
    AttackerAction,
    DefenderPolicy,
    GameState,
    UtilityWeights,
    plan,
)
from cowrie.llm.worldstate import WorldState

A = AttackerAction
P = DefenderPolicy

#: Commands the shell fastpath answers before the planner is ever consulted.
_FASTPATH = ("exit", "logout", "quit", "clear", "pwd", "cd")

#: PROVISIONAL command -> attacker-action projection.
#:
#: Phase 3 ships the real cowrie.llm.planner.classify, built over the same
#: attack_map technique ids plus supplementary rules for the categories
#: ATT&CK does not express. This local copy keeps the tuning gate usable
#: before that lands; delete it when classify.py exists.
_TECHNIQUE_TO_ACTION = {
    "T1082": A.OS_FINGERPRINT,
    "T1518": A.OS_FINGERPRINT,
    "T1083": A.FS_CONSISTENCY_PROBE,
    "T1057": A.FS_CONSISTENCY_PROBE,
    "T1033": A.USER_PRIV_DISCOVERY,
    "T1087": A.USER_PRIV_DISCOVERY,
    "T1078": A.USER_PRIV_DISCOVERY,
    "T1548.003": A.USER_PRIV_DISCOVERY,
    "T1016": A.NETWORK_DISCOVERY,
    "T1049": A.NETWORK_DISCOVERY,
    "T1021.004": A.NETWORK_DISCOVERY,
    "T1105": A.PAYLOAD_TRANSFER,
    "T1071.001": A.PAYLOAD_TRANSFER,
    "T1053.003": A.PERSISTENCE_ATTEMPT,
    "T1098.004": A.PERSISTENCE_ATTEMPT,
    "T1136": A.PERSISTENCE_ATTEMPT,
    "T1546.004": A.PERSISTENCE_ATTEMPT,
    "T1543.002": A.PERSISTENCE_ATTEMPT,
    "T1496": A.TOOL_ABUSE,
    "T1489": A.TOOL_ABUSE,
    "T1110": A.TOOL_ABUSE,
    "T1048": A.TOOL_ABUSE,
    "T1070.002": A.TOOL_ABUSE,
    "T1070.003": A.TOOL_ABUSE,
    "T1059.004": A.TOOL_ABUSE,
    "T1059.006": A.TOOL_ABUSE,
}


def classify(command: str) -> A:
    parts = command.strip().split()
    if parts and parts[0] in ("exit", "logout", "quit"):
        return A.EXIT
    try:
        for tech in attack_map.classify(command):
            hit = _TECHNIQUE_TO_ACTION.get(tech.id)
            if hit is not None:
                return hit
    except Exception:
        return A.UNKNOWN
    return A.UNKNOWN


def load_patterns():
    """Import attacker_sim for its PATTERNS without needing paramiko.

    The module imports paramiko at the top for its live SSH driver, which
    we never call -- we only want the frozen scenario catalog.
    """
    if "paramiko" not in sys.modules:
        sys.modules["paramiko"] = types.ModuleType("paramiko")
    spec = importlib.util.spec_from_file_location(
        "attacker_sim", REPO_ROOT / "scripts" / "attacker_sim.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["attacker_sim"] = module
    spec.loader.exec_module(module)
    return module.PATTERNS


def ladder_decision(command: str, ctx) -> str:
    """Mirror of cowrie.llm.protocol.lineReceived's if-ladder.

    Deliberately re-derived from the same real probes the protocol calls,
    rather than imported, so this tool works without a reactor. Keep it in
    step with protocol.py:187-210.
    """
    cmd = command.strip()
    if not cmd or cmd.split()[0] in _FASTPATH:
        return "fastpath"
    if responder.respond(cmd, ctx) is not None:
        return "deterministic"
    try:
        if interactive.make_program(cmd, ctx) is not None:
            return "interactive"
    except Exception:
        pass
    if downloader.parse_download_command(cmd) is not None:
        return "download"
    return "llm"


def policy_legality(command: str, ctx) -> frozenset:
    """The runtime mask, built from the same probes the protocol seam uses."""
    allowed = {P.DELAY_PRESSURE, P.PERSONA_LLM}
    if responder.respond(command, ctx) is not None:
        allowed.add(P.DETERMINISTIC)
    if downloader.parse_download_command(command) is not None:
        allowed.add(P.DOWNLOADER_INTERCEPT)
    return frozenset(allowed)


#: Ladder outcome -> the policy that means "the planner agreed".
_EQUIVALENT = {
    "deterministic": P.DETERMINISTIC,
    "llm": P.PERSONA_LLM,
    "download": P.DOWNLOADER_INTERCEPT,
}


def run(depth: int, weights: UtilityWeights) -> dict:
    rows = []
    picks: Counter = Counter()
    legal_counts: Counter = Counter()
    transitions: Counter = Counter()
    latencies = []
    agree = 0

    for pattern in load_patterns():
        who = persona.pick_persona("203.0.113.45")
        ctx = responder.ShellContext(
            persona=who,
            boot_time=time.time() - 90_000,
            world=WorldState(),
            cwd="/root",
            login_user=pattern.username,
            hostname="h4-web01",
            seed="planner-diff",
        )
        state = GameState()

        for raw in pattern.commands:
            command = raw.strip()
            if not command:
                continue
            ladder = ladder_decision(command, ctx)
            if ladder in ("fastpath", "interactive"):
                rows.append((pattern.name, command, ladder, "-", True))
                continue

            action = classify(command)
            state = state.with_attack(
                action if action is not A.UNKNOWN else A.FS_CONSISTENCY_PROBE
            )
            allowed = policy_legality(command, ctx)
            for policy in allowed:
                legal_counts[policy] += 1
            started = time.perf_counter()
            result = plan(state, depth=depth, weights=weights, allowed=allowed)
            latencies.append((time.perf_counter() - started) * 1000)

            chosen = result.best_policy
            picks[chosen] += 1
            same = chosen is _EQUIVALENT[ladder]
            agree += same
            if not same:
                transitions[f"{ladder} -> {chosen.value}"] += 1
            rows.append((pattern.name, command, ladder, chosen.value, same))

    decided = sum(picks.values())
    return {
        "rows": rows,
        "picks": picks,
        "legal_counts": legal_counts,
        "transitions": transitions,
        "decided": decided,
        "agree": agree,
        "latencies": latencies,
    }


def render(report: dict, show_all: bool) -> str:
    out = ["", f"{'pattern':<22} {'command':<34} {'ladder':<14} planner", "-" * 92]
    for name, command, ladder, chosen, same in report["rows"]:
        if same and not show_all:
            continue
        flag = "" if same else "   <-- differs"
        out.append(f"{name:<22} {command[:33]:<34} {ladder:<14} {chosen}{flag}")
    if len(out) == 3:
        out.append("(planner agreed with the ladder on every decidable command)")

    decided = report["decided"]
    agree = report["agree"]
    out += ["", "-" * 92, ""]
    out.append(f"decidable commands : {decided}")
    out.append(f"agreement          : {agree}/{decided} = {agree / decided:.0%}")
    out.append("")
    out.append("policy distribution:")
    for policy, count in report["picks"].most_common():
        out.append(f"    {policy.value:<22} {count:>4}  ({count / decided:.0%})")
    if report["transitions"]:
        out.append("")
        out.append("disagreements by direction:")
        for direction, count in report["transitions"].most_common():
            out.append(f"    {direction:<40} {count:>4}")

    latencies = sorted(report["latencies"])
    if latencies:
        out += [
            "",
            f"planner latency    : mean {sum(latencies) / len(latencies):.2f} ms, "
            f"p95 {latencies[int(0.95 * (len(latencies) - 1))]:.2f} ms, "
            f"max {latencies[-1]:.2f} ms",
        ]
    return "\n".join(out) + "\n"


def check(report: dict, max_share: float) -> list[str]:
    """The degeneracy smells that green unit tests cannot see.

    The sharpest of the three is `never chosen while often legal`. Both
    real defects presented that way -- PERSONA_LLM was legal on all 67
    commands and won 0, then DETERMINISTIC was legal on 27 and won 0 --
    while the cruder share and direction heuristics stayed inside their
    limits. A policy the search is free to pick and never picks is
    dominated, and dominated means the tuning is wrong, not that the
    policy is unwanted.
    """
    failures = []
    decided = report["decided"]
    if not decided:
        return ["no decidable commands -- the harness is not exercising anything"]

    picks = report["picks"]
    for policy in _EQUIVALENT.values():
        legal = report["legal_counts"].get(policy, 0)
        if legal < 0.25 * decided:
            continue  # too rarely available to draw a conclusion from
        if picks.get(policy, 0) == 0:
            failures.append(
                f"{policy.value} was legal in {legal}/{decided} decisions and chosen "
                "in none. That policy is dominated -- some other option beats it "
                "everywhere, which is a tuning defect rather than a preference."
            )

    # Only the three policies the ladder can also express are checked above.
    # DELAY_PRESSURE and friends are specialists: DELAY is legal everywhere
    # but is only the right answer under a timing probe, so "never chosen"
    # over a corpus containing no timing probes is correct behaviour rather
    # than domination. Specialists are verified by targeted probes instead
    # -- see specialist_probes().

    top_policy, top_count = picks.most_common(1)[0]
    share = top_count / decided
    if share > max_share:
        failures.append(
            f"{top_policy.value} wins {share:.0%} of decisions (limit {max_share:.0%}). "
            "One policy dominating means the utility function is not "
            "discriminating -- retune the effect tables."
        )

    directions = report["transitions"]
    disagreements = sum(directions.values())
    if disagreements >= 5:
        _, top = directions.most_common(1)[0]
        if top / disagreements >= 0.8:
            worst = directions.most_common(1)[0][0]
            failures.append(
                f"{top}/{disagreements} disagreements are the same substitution "
                f"({worst}). A one-directional diff is a stuck policy, not judgment."
            )
    return failures


#: Specialist policies and the condition each exists to handle.
#:
#: The corpus check above can only judge policies the ladder also has. A
#: specialist that is legal everywhere but correct almost nowhere -- DELAY
#: under a timing probe -- would look "never chosen" on any corpus lacking
#: its trigger, which is right, not broken. So each specialist gets one
#: targeted state where it should win. If it loses there too, it is dead
#: code and the tuning has silently removed a capability.
_SPECIALIST_PROBES = (
    (
        "interception beats narration once a payload is being fetched",
        A.PAYLOAD_TRANSFER,
        frozenset({P.DOWNLOADER_INTERCEPT, P.PERSONA_LLM, P.DELAY_PRESSURE}),
        P.DOWNLOADER_INTERCEPT,
        {},
    ),
    (
        "denial is preferred to narration when privilege is probed",
        A.USER_PRIV_DISCOVERY,
        frozenset({P.PERMISSION_DENIAL, P.DELAY_PRESSURE}),
        P.PERMISSION_DENIAL,
        {},
    ),
    (
        "the fast local answer wins a timing probe (real bash is 1-10ms)",
        A.TIMING_PROBE,
        frozenset({P.DETERMINISTIC, P.PERSONA_LLM, P.DELAY_PRESSURE}),
        P.DETERMINISTIC,
        {},
    ),
    (
        "delay is the fallback once the token budget is spent",
        A.FS_CONSISTENCY_PROBE,
        frozenset({P.PERSONA_LLM, P.DELAY_PRESSURE}),
        P.DELAY_PRESSURE,
        {"tokens_budget": 1000, "tokens_spent": 900, "est_tokens_per_llm_turn": 900},
    ),
)


def specialist_probes(depth: int, weights: UtilityWeights) -> list[tuple]:
    results = []
    for label, trigger, allowed, expected, overrides in _SPECIALIST_PROBES:
        state = GameState(**overrides).with_attack(trigger)
        chosen = plan(state, depth=depth, weights=weights, allowed=allowed).best_policy
        results.append((label, expected, chosen, chosen is expected))
    return results


def render_specialists(results: list[tuple]) -> str:
    out = ["", "specialist probes (each policy under the condition it exists for):", ""]
    for label, expected, chosen, ok in results:
        mark = "ok  " if ok else "FAIL"
        detail = f"chose {chosen.value}" if ok else f"expected {expected.value}, chose {chosen.value}"
        out.append(f"    {mark}  {label}")
        out.append(f"          {detail}")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diff the minimax planner's choices against the current if-ladder."
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--weights", default="", help="e.g. 'token_cost=2.0,engagement=1.5'"
    )
    parser.add_argument(
        "--max-share",
        type=float,
        default=0.70,
        help="fail if any single policy exceeds this share of decisions",
    )
    parser.add_argument(
        "--all", action="store_true", help="show agreements as well as differences"
    )
    args = parser.parse_args()

    weights = UtilityWeights.from_override(args.weights)
    report = run(args.depth, weights)
    print(render(report, args.all))

    specialists = specialist_probes(args.depth, weights)
    print(render_specialists(specialists))

    failures = check(report, args.max_share)
    failures += [
        f"specialist probe failed: {label} (expected {expected.value}, "
        f"chose {chosen.value})"
        for label, expected, chosen, ok in specialists
        if not ok
    ]
    if failures:
        print("FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("OK: no single policy dominates and disagreements are not one-directional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
