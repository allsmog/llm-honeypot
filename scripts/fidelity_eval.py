#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Fidelity evaluation harness for the LLM honeypot's deterministic responder.

Scores two believability axes the honeypot literature (SoK 2025, shelLM,
LLMHoney) uses:

  CONSISTENCY  — cross-command and against-persona invariants that must hold
                 (uname -r ⊂ uname -a, nproc == /proc/cpuinfo blocks,
                 id www-data == /etc/passwd, hostname == /etc/hostname, ...).
                 Pure; needs no network or host. Doubles as a CI regression
                 gate: any failure exits non-zero.

  COVERAGE     — what fraction of a recon corpus the deterministic layer
                 answers locally (instant + consistent + free) vs. defers to
                 the LLM.

  REFERENCE    — (opt-in, --reference local) structural similarity of the
                 honeypot's output to the REAL host shell after masking
                 volatile tokens. Only runs a hardcoded allowlist of
                 read-only commands; never an attacker payload.

Usage:
    PYTHONPATH=src python scripts/fidelity_eval.py
    PYTHONPATH=src python scripts/fidelity_eval.py --persona ubuntu_22_04
    PYTHONPATH=src python scripts/fidelity_eval.py --reference local
    PYTHONPATH=src python scripts/fidelity_eval.py --all-personas
"""
from __future__ import annotations

import argparse
import sys

from cowrie.llm import fidelity
from cowrie.llm.persona import PERSONAS


def _eval_persona(slug: str, *, reference: bool, min_similarity: float | None) -> bool:
    ctx = fidelity.build_context(slug)
    print(f"\n=== persona: {slug}  ({ctx.persona.distro}) ===")
    ok = True

    # Consistency — the hard, host-independent gate.
    checks = fidelity.run_consistency(ctx)
    failed = [c for c in checks if not c.passed]
    passed = len(checks) - len(failed)
    print(f"consistency: {passed}/{len(checks)} invariants hold")
    for c in failed:
        print(f"  FAIL  {c.name}: {c.detail}")
    ok = ok and not failed

    # Coverage
    cov = fidelity.coverage(ctx)
    print(f"coverage:    {cov.handled}/{cov.total} recon commands deterministic "
          f"({cov.rate * 100:.0f}%)")
    for cat, (h, t) in sorted(cov.by_category.items()):
        print(f"  {cat:<10} {h}/{t}")
    if cov.deferred_commands:
        print(f"  deferred to LLM: {', '.join(cov.deferred_commands)}")

    # Reference — INFORMATIONAL by default. Structural similarity to whatever
    # host CI runs on is inherently noisy: short identity outputs (whoami,
    # groups) and host-specific data (cpuinfo flags, cores) differ because the
    # runner's identity != our persona, not because the render is wrong. So a
    # low score is a readout, not a failure — the real structural guarantees
    # live in the consistency invariants (e.g. the meminfo field-count check).
    # Opt into hard enforcement with --enforce-similarity for a local A/B.
    if reference:
        refs = fidelity.reference_compare(ctx)
        scored = [r for r in refs if r.ran_on_host]
        if scored:
            mean = sum(r.similarity for r in scored) / len(scored)
            print(f"reference:   mean structural similarity {mean * 100:.1f}% "
                  f"over {len(scored)} commands run on host (informational)")
            for r in sorted(scored, key=lambda x: x.similarity):
                flag = "  " if r.similarity >= 0.80 else "low "
                print(f"  {flag} {r.similarity * 100:5.1f}%  {r.command}")
            if min_similarity is not None:
                below = [r for r in scored if r.similarity < min_similarity]
                if below:
                    ok = False
                    print(f"  ENFORCED: {len(below)} command(s) below the "
                          f"{min_similarity * 100:.0f}% floor")
        skipped = [r for r in refs if not r.ran_on_host]
        for r in skipped:
            print(f"  skip  {r.command}  ({r.note})")

    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persona", default="ubuntu_22_04",
                    help="persona slug to evaluate (default: ubuntu_22_04)")
    ap.add_argument("--all-personas", action="store_true",
                    help="evaluate every built-in persona")
    ap.add_argument("--reference", choices=["none", "local"], default="none",
                    help="'local' prints structural similarity to the real "
                         "host shell (informational unless --enforce-similarity)")
    ap.add_argument("--enforce-similarity", type=float, default=None,
                    metavar="FLOOR",
                    help="opt-in: fail if any host-run command scores below "
                         "FLOOR (0-1). Off by default — reference is a readout, "
                         "the consistency invariants are the gate.")
    args = ap.parse_args(argv)

    slugs = [p.slug for p in PERSONAS] if args.all_personas else [args.persona]
    all_ok = True
    for slug in slugs:
        ok = _eval_persona(slug, reference=(args.reference == "local"),
                           min_similarity=args.enforce_similarity)
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("RESULT: all consistency invariants hold.")
        return 0
    print("RESULT: FAILURES detected (see above).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
