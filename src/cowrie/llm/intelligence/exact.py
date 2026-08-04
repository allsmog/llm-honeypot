# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Exact inference — enumeration and variable elimination. These
# ABOUTME: are the oracle every sampler is validated against in tests.

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cowrie.llm.intelligence.factors import (
    Factor,
    multiply,
    normalize,
    restrict,
    sum_out,
)
from cowrie.llm.intelligence.network import BayesNet, factor_for


def enumeration_ask(
    net: BayesNet, query: str, evidence: Mapping[str, str]
) -> dict[str, float]:
    """P(query | evidence) by summing the full joint. Exponential; kept as
    the simplest-possible reference implementation."""
    order = net.topological_order()
    dist: dict[str, float] = {}
    for value in net.variables[query].domain:
        extended = dict(evidence)
        extended[query] = value
        dist[value] = _enumerate_all(net, order, extended)
    total = sum(dist.values())
    if total <= 0.0:
        return dist
    return {v: p / total for v, p in dist.items()}


def _enumerate_all(
    net: BayesNet, order: Sequence[str], assignment: dict[str, str]
) -> float:
    if not order:
        return 1.0
    first, rest = order[0], order[1:]
    if first in assignment:
        return net.prob(first, assignment[first], assignment) * _enumerate_all(
            net, rest, assignment
        )
    total = 0.0
    for value in net.variables[first].domain:
        assignment[first] = value
        total += net.prob(first, value, assignment) * _enumerate_all(
            net, rest, assignment
        )
    del assignment[first]
    return total


def variable_elimination(
    net: BayesNet,
    query: str,
    evidence: Mapping[str, str],
    order: Sequence[str] | None = None,
) -> dict[str, float]:
    """P(query | evidence) by factor elimination.

    ``order`` fixes the elimination order of the hidden variables; by
    default a min-degree greedy order over the factor interaction graph.
    """
    factors: list[Factor] = []
    for name in net.variables:
        f = factor_for(net, name)
        for var, value in evidence.items():
            if var in f.variables:
                f = restrict(f, var, value)
        factors.append(f)

    hidden = [
        name for name in net.variables if name != query and name not in evidence
    ]
    if order is None:
        order = _min_degree_order(factors, hidden)
    else:
        missing = set(hidden) - set(order)
        if missing:
            raise ValueError(f"elimination order omits hidden variables {missing}")

    for var in order:
        involved = [f for f in factors if var in f.variables]
        if not involved:
            continue
        product = involved[0]
        for f in involved[1:]:
            product = multiply(product, f)
        factors = [f for f in factors if var not in f.variables]
        factors.append(sum_out(product, var))

    result = factors[0]
    for f in factors[1:]:
        result = multiply(result, f)
    normalized = normalize(result)
    idx = normalized.variables.index(query)
    dist: dict[str, float] = {}
    for key, p in normalized.table.items():
        dist[key[idx]] = dist.get(key[idx], 0.0) + p
    return dist


def _min_degree_order(factors: Sequence[Factor], hidden: Sequence[str]) -> list[str]:
    """Greedy min-degree: eliminate the variable connected to the fewest
    others in the current interaction graph, updating as elimination fuses
    factor scopes."""
    def degree(var: str, scopes: list[set[str]]) -> int:
        neighbors: set[str] = set()
        for scope in scopes:
            if var in scope:
                neighbors.update(scope)
        neighbors.discard(var)
        return len(neighbors)

    scopes = [set(f.variables) for f in factors]
    remaining = list(hidden)
    order: list[str] = []
    while remaining:
        best = min(remaining, key=lambda v: (degree(v, scopes), v))
        order.append(best)
        remaining.remove(best)
        fused: set[str] = set()
        kept: list[set[str]] = []
        for scope in scopes:
            if best in scope:
                fused.update(scope)
            else:
                kept.append(scope)
        fused.discard(best)
        if fused:
            kept.append(fused)
        scopes = kept
    return order
