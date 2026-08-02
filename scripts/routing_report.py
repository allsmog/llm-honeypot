#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Reports where each attacker command gets routed — local
# ABOUTME: emulator, download interceptor, full-screen program, or the
# ABOUTME: model — by replaying attacker_sim's patterns through the same
# ABOUTME: probes protocol.py's if-ladder uses. Fails when the model's
# ABOUTME: share of traffic regresses, which is the cheapest guard we have
# ABOUTME: on cost, latency and self-consistency all at once.

"""Command routing report, and a regression gate on model-bound traffic.

Every command the local emulator answers is instant, free, and identical
every time it is asked. Every command that reaches the model costs tokens,
adds 300-2000ms where real bash takes 1-10ms, and is free to contradict an
answer given earlier in the session. So the share of traffic reaching the
model is a direct proxy for cost, timing fingerprint and consistency risk
at once — worth watching as a single number.

This tool exists because that measurement already paid for itself. An
earlier version, written to compare a minimax planner against the ladder,
found that 19% of model-bound commands were piped forms of commands the
emulator already answered (`free -m | head -2` and friends) — including
two of the three repeat probes in FINGERPRINT_PROBE, the adversary written
specifically to catch the honeypot contradicting itself. Fixing that moved
model-bound traffic from 48% to 39%. The planner was retired (see the
minimax-planner-v1 tag); the measurement was worth keeping.

What it cannot tell you: whether 39% is *good*. These are our own
hand-written patterns, so the number describes attacker_sim, not real
traffic. It is a regression detector, never a claim about the wild.
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
from cowrie.llm import downloader, interactive, persona, responder
from cowrie.llm.worldstate import WorldState

#: Commands the shell fastpath answers before anything else is consulted.
_FASTPATH_CMDS = ("exit", "logout", "quit", "clear", "pwd", "cd")

#: Where a command can end up, in the ladder's own order.
FASTPATH = "fastpath"
LOCAL = "emulator"
INTERACTIVE = "fullscreen"
DOWNLOAD = "download"
MODEL = "model"

#: Routes that cost neither tokens nor a model round trip.
_FREE = (FASTPATH, LOCAL, INTERACTIVE)


def load_patterns():
    """Import attacker_sim for its PATTERNS without needing paramiko.

    The module imports paramiko at the top for its live SSH driver, which
    we never call — we only want the frozen scenario catalog.
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


def route(command: str, ctx) -> str:
    """Mirror of cowrie.llm.protocol.lineReceived's if-ladder.

    Deliberately re-derived from the same real probes the protocol calls,
    rather than imported, so this runs without a reactor. Keep it in step
    with protocol.py's lineReceived — if the ladder gains a stage and this
    does not, the report quietly misattributes that traffic.
    """
    cmd = command.strip()
    if not cmd or cmd.split()[0] in _FASTPATH_CMDS:
        return FASTPATH
    if responder.respond(cmd, ctx) is not None:
        return LOCAL
    try:
        if interactive.make_program(cmd, ctx) is not None:
            return INTERACTIVE
    except Exception:
        pass
    if downloader.parse_download_command(cmd) is not None:
        return DOWNLOAD
    return MODEL


def run() -> dict:
    rows = []
    totals: Counter = Counter()
    per_pattern: dict[str, Counter] = {}

    for pattern in load_patterns():
        who = persona.pick_persona("203.0.113.45")
        ctx = responder.ShellContext(
            persona=who,
            boot_time=time.time() - 90_000,
            world=WorldState(),
            cwd="/root",
            login_user=pattern.username,
            hostname="h4-web01",
            seed="routing-report",
        )
        counts: Counter = Counter()
        for raw in pattern.commands:
            command = raw.strip()
            if not command:
                continue
            where = route(command, ctx)
            counts[where] += 1
            totals[where] += 1
            rows.append((pattern.name, command, where))
        per_pattern[pattern.name] = counts

    total = sum(totals.values())
    return {
        "rows": rows,
        "totals": totals,
        "per_pattern": per_pattern,
        "total": total,
        "model_share": totals[MODEL] / total if total else 0.0,
    }


def render(report: dict, show_all: bool) -> str:
    out = ["", f"{'pattern':<24} {'command':<42} route", "-" * 82]
    for name, command, where in report["rows"]:
        if where in _FREE and not show_all:
            continue
        out.append(f"{name:<24} {command[:41]:<42} {where}")
    if len(out) == 3:
        out.append("(every command is answered without the model)")

    total = report["total"]
    out += ["", "-" * 82, "", f"commands: {total}", "", "routing:"]
    for where, count in report["totals"].most_common():
        out.append(f"    {where:<12} {count:>4}  ({count / total:.0%})")

    out += ["", "by pattern:"]
    for name, counts in report["per_pattern"].items():
        n = sum(counts.values())
        out.append(f"    {name:<24} {counts.get(MODEL, 0)}/{n} to the model")

    model_cmds = sorted({c for _, c, w in report["rows"] if w == MODEL})
    if model_cmds:
        out += ["", "still reaching the model — the worklist for emulator coverage:"]
        out += [f"    {c}" for c in model_cmds]
    return "\n".join(out) + "\n"


def check(report: dict, max_share: float) -> list[str]:
    """Fail when model-bound traffic regresses.

    One number, three risks: tokens spent, response latency against real
    bash's 1-10ms, and the chance of contradicting an earlier answer. A
    rise means the emulator lost coverage — which is exactly how the
    piped-command gap went unnoticed until someone measured it.
    """
    if not report["total"]:
        return ["no commands routed — the harness is not exercising anything"]
    share = report["model_share"]
    if share > max_share:
        return [
            f"{share:.0%} of commands reach the model (limit {max_share:.0%}). "
            "The emulator has lost coverage: every command it stops answering "
            "costs tokens, adds latency, and becomes a fresh chance to "
            "contradict an earlier answer."
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report where attacker commands get routed, and gate on "
        "the share reaching the model."
    )
    parser.add_argument(
        "--max-llm-share",
        type=float,
        default=0.45,
        help="fail above this fraction (default 0.45; currently ~0.39, and "
        "0.48 before pipeline support landed)",
    )
    parser.add_argument(
        "--all", action="store_true", help="list locally-answered commands too"
    )
    args = parser.parse_args()

    report = run()
    print(render(report, args.all))

    failures = check(report, args.max_llm_share)
    if failures:
        print("FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(
        f"OK: {report['model_share']:.0%} of commands reach the model "
        f"(limit {args.max_llm_share:.0%})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
