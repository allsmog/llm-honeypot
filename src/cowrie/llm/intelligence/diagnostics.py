# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Posterior over fault families plus which unobserved signal would
# ABOUTME: most reduce the remaining uncertainty, rendered as plain text in
# ABOUTME: the same house style as the other harness reports.

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cowrie.llm.intelligence import structure
from cowrie.llm.intelligence.exact import variable_elimination
from cowrie.llm.intelligence.network import BayesNet

MODEL_PATH = Path(__file__).parent / "models" / "fidelity-v1.json"

#: Below this, the top family is not a diagnosis, it is a coin toss between
#: several. Saying so is more useful than naming a winner.
CONFIDENCE_FLOOR = 0.5


def load_model(path: Path | None = None) -> BayesNet:
    return BayesNet.from_json((path or MODEL_PATH).read_text())


@dataclass
class Diagnosis:
    posterior: dict[str, float]
    top: str
    top_p: float
    next_evidence: list[tuple[str, float]]
    evidence: dict[str, str]

    @property
    def confident(self) -> bool:
        return self.top_p >= CONFIDENCE_FLOOR

    def render(self) -> str:
        observed = ", ".join(f"{k}={v}" for k, v in sorted(self.evidence.items()))
        lines = [f"posterior over fault family (evidence: {observed or 'none'}):"]
        ranked = sorted(self.posterior.items(), key=lambda kv: (-kv[1], kv[0]))
        for name, p in ranked:
            if p < 0.001:
                continue
            lines.append(f"  {name:<18} {p:.3f}")
        lines.append("")
        if self.confident:
            lines.append(f"likely fault: {self.top} (p={self.top_p:.3f})")
        else:
            joint = ", ".join(n for n, p in ranked[:3] if p > 0.05)
            lines.append(
                f"no confident diagnosis (top {self.top} at p={self.top_p:.3f}); "
                f"candidates: {joint}"
            )
        if self.next_evidence:
            lines.append("most informative next evidence:")
            for node, gain in self.next_evidence:
                lines.append(f"  {node:<22} {gain:.3f} bits")
        return "\n".join(lines)


def _entropy(dist: Mapping[str, float]) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0.0)


def expected_information_gain(
    net: BayesNet, target: str, candidate: str, evidence: Mapping[str, str]
) -> float:
    """H(target | e) - E_v[ H(target | e, candidate=v) ], in bits.

    Non-negative by construction (conditioning never increases expected
    entropy), and exactly zero for an already-observed candidate.
    """
    if candidate in evidence:
        return 0.0
    prior = variable_elimination(net, target, evidence)
    base = _entropy(prior)
    candidate_dist = variable_elimination(net, candidate, evidence)
    expected = 0.0
    for value, weight in candidate_dist.items():
        if weight <= 0.0:
            continue
        posterior = variable_elimination(
            net, target, {**evidence, candidate: value}
        )
        expected += weight * _entropy(posterior)
    return max(0.0, base - expected)


def diagnose(
    net: BayesNet, evidence: Mapping[str, str], *, top_k: int = 3
) -> Diagnosis:
    posterior = variable_elimination(net, "fault", evidence)
    top, top_p = max(posterior.items(), key=lambda kv: (kv[1], kv[0]))
    gains = [
        (node, expected_information_gain(net, "fault", node, evidence))
        for node in structure.EVIDENCE_NODES
        if node not in evidence
    ]
    ranked = sorted(gains, key=lambda kv: (-kv[1], kv[0]))
    next_evidence = [(n, g) for n, g in ranked if g > 1e-9][:top_k]
    return Diagnosis(
        posterior=posterior,
        top=top,
        top_p=top_p,
        next_evidence=next_evidence,
        evidence=dict(evidence),
    )
