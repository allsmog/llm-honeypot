# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Approximate inference — likelihood weighting, Gibbs, and
# ABOUTME: Metropolis-Hastings. Every entry point takes an explicit string
# ABOUTME: seed; there is no module-level RNG state, matching _rng_floats.

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping

from cowrie.llm.intelligence.network import BayesNet


def _seeded_rng(seed: str) -> random.Random:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _sample_value(
    net: BayesNet, name: str, assignment: Mapping[str, str], rng: random.Random
) -> str:
    roll = rng.random()
    cumulative = 0.0
    domain = net.variables[name].domain
    for value in domain:
        cumulative += net.prob(name, value, assignment)
        if roll < cumulative:
            return value
    return domain[-1]


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0.0:
        return weights
    return {v: w / total for v, w in weights.items()}


def likelihood_weighting(
    net: BayesNet,
    query: str,
    evidence: Mapping[str, str],
    *,
    n: int,
    seed: str,
) -> dict[str, float]:
    rng = _seeded_rng(seed)
    order = net.topological_order()
    weights = {v: 0.0 for v in net.variables[query].domain}
    for _ in range(n):
        sample: dict[str, str] = {}
        weight = 1.0
        for name in order:
            if name in evidence:
                sample[name] = evidence[name]
                weight *= net.prob(name, evidence[name], sample)
            else:
                sample[name] = _sample_value(net, name, sample, rng)
        weights[sample[query]] += weight
    return _normalize(weights)


def _blanket_distribution(
    net: BayesNet, name: str, state: dict[str, str]
) -> list[float]:
    """P(name | markov blanket) up to normalization: the variable's own CPT
    times each child's CPT evaluated under the current state."""
    domain = net.variables[name].domain
    children = net.children(name)
    scores: list[float] = []
    original = state[name]
    for value in domain:
        state[name] = value
        p = net.prob(name, value, state)
        for child in children:
            p *= net.prob(child, state[child], state)
        scores.append(p)
    state[name] = original
    return scores


def _initial_state(
    net: BayesNet, evidence: Mapping[str, str], rng: random.Random
) -> dict[str, str]:
    state: dict[str, str] = {}
    for name in net.topological_order():
        if name in evidence:
            state[name] = evidence[name]
        else:
            state[name] = _sample_value(net, name, state, rng)
    return state


def gibbs(
    net: BayesNet,
    query: str,
    evidence: Mapping[str, str],
    *,
    n: int,
    burn: int,
    seed: str,
) -> dict[str, float]:
    """``n`` sweeps (each resampling every non-evidence variable once),
    counting the query value after each post-burn sweep."""
    rng = _seeded_rng(seed)
    state = _initial_state(net, evidence, rng)
    free = [v for v in net.topological_order() if v not in evidence]
    counts = {v: 0 for v in net.variables[query].domain}
    for sweep in range(n):
        for name in free:
            scores = _blanket_distribution(net, name, state)
            total = sum(scores)
            domain = net.variables[name].domain
            if total <= 0.0:
                state[name] = domain[rng.randrange(len(domain))]
                continue
            roll = rng.random() * total
            cumulative = 0.0
            for value, score in zip(domain, scores, strict=True):
                cumulative += score
                if roll < cumulative:
                    state[name] = value
                    break
            else:
                state[name] = domain[-1]
        if sweep >= burn:
            counts[state[query]] += 1
    return _normalize({v: float(c) for v, c in counts.items()})


def metropolis_hastings(
    net: BayesNet,
    query: str,
    evidence: Mapping[str, str],
    *,
    n: int,
    burn: int,
    seed: str,
) -> dict[str, float]:
    """Single-variable uniform proposal; the acceptance ratio only needs the
    proposed variable's factor and its children's factors, since everything
    else cancels in the joint ratio."""
    rng = _seeded_rng(seed)
    state = _initial_state(net, evidence, rng)
    free = [v for v in net.topological_order() if v not in evidence]
    counts = {v: 0 for v in net.variables[query].domain}

    def local_score(name: str) -> float:
        p = net.prob(name, state[name], state)
        for child in net.children(name):
            p *= net.prob(child, state[child], state)
        return p

    for step in range(n):
        name = free[rng.randrange(len(free))]
        domain = net.variables[name].domain
        proposed = domain[rng.randrange(len(domain))]
        current = state[name]
        if proposed != current:
            before = local_score(name)
            state[name] = proposed
            after = local_score(name)
            if before <= 0.0:
                accept = after > 0.0
            else:
                accept = rng.random() < min(1.0, after / before)
            if not accept:
                state[name] = current
        if step >= burn:
            counts[state[query]] += 1
    return _normalize({v: float(c) for v, c in counts.items()})
