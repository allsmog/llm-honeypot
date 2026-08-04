#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Ask the trained network which subsystem broke, given whichever
# ABOUTME: harness signals are in hand, and which unobserved signal would
# ABOUTME: settle it fastest.

"""Diagnose a fidelity failure.

    python scripts/diagnose.py --collect                 # run the batteries now
    python scripts/diagnose.py --evidence audit_append=doubled
    python scripts/diagnose.py --from-report evidence.jsonl
    python scripts/diagnose.py --self-test               # CI: inject each fault

This does not gate anything on its own. The fidelity and probe harnesses
decide whether the build fails; this only says where to look when one does.
A posterior is a hypothesis ranking, not a verdict — which is why
`--self-test` measures it against injected faults whose answer is known.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cowrie.llm.intelligence import structure  # noqa: E402
from cowrie.llm.intelligence.diagnostics import (  # noqa: E402
    diagnose,
    load_model,
)


def _parse_evidence(pairs: list[str]) -> dict[str, str]:
    evidence: dict[str, str] = {}
    by_name = {v.name: v for v in structure.STRUCTURE}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"evidence must be key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        var = by_name.get(key)
        if var is None or key == "fault":
            raise SystemExit(
                f"unknown evidence node {key!r}; known: "
                f"{', '.join(structure.EVIDENCE_NODES)}, persona"
            )
        if value not in var.domain:
            raise SystemExit(
                f"{key}={value!r} is not one of {', '.join(var.domain)}"
            )
        evidence[key] = value
    return evidence


def _collect(persona_slug: str, fault: str = "no_fault") -> dict[str, str]:
    import diagnoser_train

    record = diagnoser_train.collect(fault, persona_slug)
    return {k: v for k, v in record.items() if k != "fault"}


def _self_test(net) -> int:
    """Inject each fault in turn and check the diagnosis names it.

    Reported per fault with the posterior mass, because "identified it, but
    only just" and "identified it decisively" are different results and
    averaging them into one accuracy number hides which is which.
    """

    from cowrie.llm.intelligence.faults import FAULTS

    print("self-test: inject each fault, collect full evidence, diagnose")
    failures: list[str] = []
    for fault in FAULTS:
        for slug in ("ubuntu_22_04", "alpine_3_19"):
            evidence = _collect(slug, fault)
            result = diagnose(net, evidence)
            hit = result.top == fault
            mark = "ok  " if hit else "MISS"
            print(f"  {mark} {fault:<18} {slug:<14} "
                  f"-> {result.top:<18} p={result.top_p:.3f}")
            if not hit:
                failures.append(f"{fault}/{slug} diagnosed as {result.top}")
    if failures:
        print(f"FAIL: {len(failures)} misdiagnosed:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("OK: every injected fault was identified.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--from-report", type=Path, help="JSONL of evidence records")
    parser.add_argument(
        "--collect", action="store_true", help="run the batteries now"
    )
    parser.add_argument("--persona", default="ubuntu_22_04")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=Path, help="model artifact to load")
    args = parser.parse_args()

    net = load_model(args.model)

    if args.self_test:
        return _self_test(net)

    evidence: dict[str, str] = {}
    if args.collect:
        evidence.update(_collect(args.persona))
    if args.from_report:
        for line in args.from_report.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                evidence.update(
                    {k: v for k, v in record.items() if k != "fault"}
                )
    evidence.update(_parse_evidence(args.evidence))

    if not evidence:
        parser.error("give --evidence, --from-report, --collect, or --self-test")

    print(diagnose(net, evidence).render())
    # Only worth saying when an inert node is actually reporting trouble:
    # a signal no modelled fault ever tripped is now tripping, so the cause
    # is very likely outside the eight families and the posterior above is
    # ranking the wrong hypotheses.
    inert = set(net.meta.get("inert_nodes") or [])
    firing = sorted(
        node
        for node, value in evidence.items()
        if node in inert and value != structure.CLEAN_EVIDENCE.get(node)
    )
    if firing:
        print()
        subject = "is" if len(firing) == 1 else "are"
        print(
            f"note: {', '.join(firing)} {subject} reporting a failure that no "
            "modelled fault produces. The posterior above cannot explain it — "
            "treat this as a fault family the diagnoser does not know."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
