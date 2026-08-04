# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Discrete factor algebra — the four operations every exact
# ABOUTME: inference algorithm is built from. Pure stdlib, no cowrie imports.

from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class Factor:
    """A table over a tuple of named discrete variables.

    ``table`` maps a full assignment (one value per variable, in
    ``variables`` order) to a non-negative float. Assignments absent from
    the table are treated as zero by construction — every constructor here
    enumerates the full cross product, so lookups never miss.
    """

    variables: tuple[str, ...]
    domains: tuple[tuple[str, ...], ...]
    table: dict[tuple[str, ...], float]

    def domain_of(self, var: str) -> tuple[str, ...]:
        return self.domains[self.variables.index(var)]


def restrict(f: Factor, var: str, value: str) -> Factor:
    """Fix ``var`` to ``value``, dropping it from the factor's scope."""
    idx = f.variables.index(var)
    variables = f.variables[:idx] + f.variables[idx + 1 :]
    domains = f.domains[:idx] + f.domains[idx + 1 :]
    table = {
        key[:idx] + key[idx + 1 :]: p
        for key, p in f.table.items()
        if key[idx] == value
    }
    return Factor(variables, domains, table)


def multiply(a: Factor, b: Factor) -> Factor:
    """Pointwise product over the union of the two scopes."""
    variables = a.variables + tuple(v for v in b.variables if v not in a.variables)
    domains = a.domains + tuple(
        b.domains[b.variables.index(v)]
        for v in b.variables
        if v not in a.variables
    )
    a_idx = [variables.index(v) for v in a.variables]
    b_idx = [variables.index(v) for v in b.variables]
    table: dict[tuple[str, ...], float] = {}
    for key in itertools.product(*domains):
        a_key = tuple(key[i] for i in a_idx)
        b_key = tuple(key[i] for i in b_idx)
        table[key] = a.table.get(a_key, 0.0) * b.table.get(b_key, 0.0)
    return Factor(variables, domains, table)


def sum_out(f: Factor, var: str) -> Factor:
    """Marginalize ``var`` out of the factor."""
    idx = f.variables.index(var)
    variables = f.variables[:idx] + f.variables[idx + 1 :]
    domains = f.domains[:idx] + f.domains[idx + 1 :]
    table: dict[tuple[str, ...], float] = {}
    for key, p in f.table.items():
        reduced = key[:idx] + key[idx + 1 :]
        table[reduced] = table.get(reduced, 0.0) + p
    return Factor(variables, domains, table)


def normalize(f: Factor) -> Factor:
    """Scale the table to sum to 1. A zero factor stays zero rather than
    raising: callers treat that as "evidence has probability zero" and
    surface it themselves with more context than a ZeroDivisionError."""
    total = sum(f.table.values())
    if total <= 0.0:
        return f
    return Factor(f.variables, f.domains, {k: p / total for k, p in f.table.items()})
