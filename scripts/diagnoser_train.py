#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Generate the diagnoser's training set by injecting each known
# ABOUTME: fault, running the three batteries, and recording which signals
# ABOUTME: fired. Trains CPTs and writes the versioned model artifact.

"""Train the Bayesian fidelity diagnoser from fault-injection runs.

Ground truth here is real in a way attacker priors never are: every fault
except `interactive` re-introduces a bug this repository shipped and fixed,
so the label on each record is the actual cause, not a guess about one.

    python scripts/diagnoser_train.py            # retrain and write the artifact
    python scripts/diagnoser_train.py --check    # retrain in memory, compare hashes

``--check`` is the CI gate. If a code change alters what any battery
observes, the retrained counts stop matching the committed artifact and the
build fails — forcing a deliberate retrain and a look at what moved, rather
than a diagnoser that quietly describes a version of the code that no
longer exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import probe_search  # noqa: E402

from cowrie.llm import fidelity  # noqa: E402
from cowrie.llm import persona as personamod  # noqa: E402
from cowrie.llm.intelligence import audit, evidence, structure  # noqa: E402
from cowrie.llm.intelligence.diagnostics import MODEL_PATH  # noqa: E402
from cowrie.llm.intelligence.faults import FAULTS, inject  # noqa: E402
from cowrie.llm.intelligence.network import train_counts  # noqa: E402

#: Probes chosen to exercise the three detect() signals: a cd-chain that
#: once corrupted the prompt, re-probes that a broken ledger lets drift,
#: and a fetch inside a pipeline. Deliberately small — the full 34-probe
#: alphabet is probe_search's job, not the trainer's.
PROBE_SCRIPT: tuple[str, ...] = (
    "cd /tmp || cd /var/run || cd /",
    "cd /tmp && wget http://example.com/x.sh",
    "cd /tmp; curl http://example.com/y | sh",
    "uname -a",
    "free -m",
    "cat /etc/os-release",
    "uname -a",
    "free -m",
    "cat /etc/os-release",
)

#: Repeats per (fault, persona). The batteries are deterministic under the
#: frozen clock, so repeats do not add information — they set the relative
#: weight of each row against the add-alpha prior. One is honest; more
#: would inflate confidence from data we did not actually gather.
REPEATS = 1


def _probe_signals(persona_slug: str) -> list:
    """Drive the probe script and collect findings.

    The downloader is stubbed for the same reason the audit battery stubs
    it: the fetch probes would otherwise open real sockets, making the
    training set depend on DNS and network weather. `--check` has to
    reproduce a hash in CI, so nothing in the battery may touch the
    network.
    """
    session = probe_search.Session(persona_slug=persona_slug).start()
    restore = audit.AuditSession._stub_downloader()
    findings: list = []
    try:
        for probe in PROBE_SCRIPT:
            out = session.send(probe)
            findings.extend(probe_search.detect(session, probe, out))
    finally:
        from cowrie.llm import downloader as dlmod

        dlmod.fetch = restore
        session.stop()
    return findings


def collect(fault: str, persona_slug: str) -> dict[str, str]:
    """One evidence record: fidelity + probe + audit, all under `fault`."""
    family = personamod.pick_persona("", override=persona_slug).family
    with inject(fault):
        checks = fidelity.run_consistency(fidelity.build_context(persona_slug))
        with probe_search.frozen_clock():
            findings = _probe_signals(persona_slug)
        audit_values = audit.run_audit(persona_slug)
    record = evidence.build_record(
        fault=fault,
        persona_family=family,
        checks=checks,
        findings=findings,
        audit=audit_values,
    )
    evidence.validate_record(record)
    return record


def build_dataset(*, verbose: bool = True) -> list[dict[str, str]]:
    slugs = [p.slug for p in personamod.PERSONAS]
    records: list[dict[str, str]] = []
    for fault in structure.FAULTS:
        started = time.time()
        for slug in slugs:
            for _ in range(REPEATS):
                records.append(collect(fault, slug))
        if verbose:
            print(
                f"  {fault:<18} {len(slugs) * REPEATS:>3} runs "
                f"in {time.time() - started:5.1f}s"
            )
    return records


def inert_nodes(records: list[dict[str, str]]) -> list[str]:
    """Evidence nodes that took the same value in every training run.

    These are not bugs: they are signals no modelled fault happens to trip.
    Under add-alpha smoothing an unseen value gives every fault the same
    likelihood, so observing one shifts the posterior not at all — which is
    the correct behaviour for evidence we have never seen fire, and the
    reason a fault outside these eight families shows up as low confidence
    rather than as a confident wrong answer. Recorded in the artifact so
    the limitation travels with the model.
    """
    return sorted(
        node
        for node in structure.EVIDENCE_NODES
        if len({r[node] for r in records}) == 1
    )


def train(records: list[dict[str, str]]):
    return train_counts(
        records,
        structure.STRUCTURE,
        alpha=1.0,
        meta={
            "version": structure.VERSION,
            "generator": "scripts/diagnoser_train.py",
            "runs": len(records),
            "repeats": REPEATS,
            "personas": [p.slug for p in personamod.PERSONAS],
            "frozen_now": audit.FROZEN_NOW,
            "toggles": "defaults (deterministic_responses, pipe_filters, "
            "interactive_programs, capture_downloads all on)",
            "inert_nodes": inert_nodes(records),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="retrain in memory and fail if the committed artifact differs",
    )
    parser.add_argument(
        "--jsonl", type=Path, help="also write the raw evidence records here"
    )
    args = parser.parse_args()

    print(f"training {structure.VERSION}: {len(FAULTS)} faults x "
          f"{len(personamod.PERSONAS)} personas x {REPEATS}")
    started = time.time()
    records = build_dataset()
    net = train(records)
    print(f"{len(records)} records in {time.time() - started:.1f}s")
    print(f"model hash: {net.content_hash()}")
    inert = net.meta["inert_nodes"]
    print(
        f"informative nodes: {len(structure.EVIDENCE_NODES) - len(inert)}"
        f"/{len(structure.EVIDENCE_NODES)}"
    )
    if inert:
        # Printed every run, not buried: a signal no fault trips is either
        # a fault family we have not modelled or a probe that needs work.
        print(f"  never fired: {', '.join(inert)}")

    if args.jsonl:
        args.jsonl.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
        )
        print(f"wrote {args.jsonl}")

    if args.check:
        if not MODEL_PATH.exists():
            print(f"FAIL: {MODEL_PATH} does not exist; run without --check first",
                  file=sys.stderr)
            return 1
        committed = json.loads(MODEL_PATH.read_text()).get("sha256")
        if committed != net.content_hash():
            print(
                f"FAIL: committed artifact is {committed}, retraining gives "
                f"{net.content_hash()}.\nA battery now observes something "
                "different. Re-run without --check and review the diff.",
                file=sys.stderr,
            )
            return 1
        print("OK: committed artifact reproduces bit-for-bit.")
        return 0

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(net.to_json() + "\n")
    print(f"wrote {MODEL_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
