# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Discrete Bayesian network stored as integer counts + add-alpha
# ABOUTME: smoothing, so the committed artifact hashes identically across
# ABOUTME: platforms — CPTs are derived at load, never serialized as floats.

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from cowrie.llm.intelligence.factors import Factor


@dataclass(frozen=True)
class Variable:
    name: str
    domain: tuple[str, ...]
    parents: tuple[str, ...] = ()


# counts[node][parent_assignment][value] -> observation count. Parent
# assignments follow the order of Variable.parents.
Counts = dict[str, dict[tuple[str, ...], dict[str, float]]]


@dataclass
class BayesNet:
    variables: dict[str, Variable]
    counts: Counts = field(default_factory=dict)
    alpha: float = 1.0
    meta: dict = field(default_factory=dict)

    # -- construction ---------------------------------------------------

    @classmethod
    def from_structure(
        cls, structure: Iterable[Variable], *, alpha: float = 1.0, meta: dict | None = None
    ) -> BayesNet:
        variables = {v.name: v for v in structure}
        for v in variables.values():
            for p in v.parents:
                if p not in variables:
                    raise ValueError(f"{v.name} has undeclared parent {p!r}")
        return cls(variables=variables, alpha=alpha, meta=dict(meta or {}))

    @classmethod
    def from_cpts(
        cls,
        structure: Iterable[Variable],
        cpts: Mapping[str, Mapping[tuple[str, ...], Mapping[str, float]]],
    ) -> BayesNet:
        """Build a net from explicit probabilities (tests and textbook nets).

        Probabilities are stored directly as "counts" with alpha=0, under
        which prob() reduces to count/total — i.e. the given CPT exactly.
        """
        net = cls.from_structure(structure, alpha=0.0)
        net.counts = {
            name: {tuple(pk): dict(row) for pk, row in table.items()}
            for name, table in cpts.items()
        }
        return net

    def add_record(self, record: Mapping[str, str]) -> None:
        """Count one fully-observed training record."""
        for name, var in self.variables.items():
            value = record[name]
            if value not in var.domain:
                raise ValueError(f"{name}={value!r} outside domain {var.domain}")
            parent_key = tuple(record[p] for p in var.parents)
            row = self.counts.setdefault(name, {}).setdefault(parent_key, {})
            row[value] = row.get(value, 0) + 1

    # -- queries --------------------------------------------------------

    def prob(self, name: str, value: str, assignment: Mapping[str, str]) -> float:
        var = self.variables[name]
        parent_key = tuple(assignment[p] for p in var.parents)
        row = self.counts.get(name, {}).get(parent_key, {})
        total = sum(row.values())
        denom = total + self.alpha * len(var.domain)
        if denom <= 0.0:
            # No observations and no smoothing: the honest answer is
            # "we know nothing about this row", which is uniform.
            return 1.0 / len(var.domain)
        return (row.get(value, 0) + self.alpha) / denom

    def topological_order(self) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()

        def visit(name: str, trail: tuple[str, ...]) -> None:
            if name in seen:
                return
            if name in trail:
                raise ValueError(f"cycle through {name!r}")
            for p in self.variables[name].parents:
                visit(p, (*trail, name))
            seen.add(name)
            order.append(name)

        for name in self.variables:
            visit(name, ())
        return order

    def children(self, name: str) -> list[str]:
        return [v.name for v in self.variables.values() if name in v.parents]

    def markov_blanket(self, name: str) -> set[str]:
        blanket: set[str] = set(self.variables[name].parents)
        for child in self.children(name):
            blanket.add(child)
            blanket.update(self.variables[child].parents)
        blanket.discard(name)
        return blanket

    # -- serialization --------------------------------------------------

    _KEY_SEP = "\x1f"  # unit separator: cannot appear in domain values

    def _canonical(self) -> dict:
        return {
            "version": self.meta.get("version", ""),
            "alpha": self.alpha,
            "variables": [
                {"name": v.name, "domain": list(v.domain), "parents": list(v.parents)}
                for v in self.variables.values()
            ],
            "counts": {
                name: {
                    self._KEY_SEP.join(pk): dict(sorted(row.items()))
                    for pk, row in sorted(table.items())
                }
                for name, table in sorted(self.counts.items())
            },
        }

    def content_hash(self) -> str:
        blob = json.dumps(self._canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        doc = self._canonical()
        doc["sha256"] = self.content_hash()
        doc["meta"] = self.meta
        return json.dumps(doc, sort_keys=True, indent=1)

    @classmethod
    def from_json(cls, text: str) -> BayesNet:
        doc = json.loads(text)
        structure = [
            Variable(v["name"], tuple(v["domain"]), tuple(v["parents"]))
            for v in doc["variables"]
        ]
        net = cls.from_structure(structure, alpha=doc["alpha"], meta=doc.get("meta", {}))
        net.counts = {
            name: {
                tuple(pk.split(cls._KEY_SEP)) if pk else (): dict(row)
                for pk, row in table.items()
            }
            for name, table in doc["counts"].items()
        }
        embedded = doc.get("sha256")
        if embedded and embedded != net.content_hash():
            raise ValueError(
                "model artifact hash mismatch: file says "
                f"{embedded[:12]}…, content is {net.content_hash()[:12]}…"
            )
        return net


def factor_for(net: BayesNet, name: str) -> Factor:
    """The CPT of ``name`` as a factor over (parents…, name)."""
    var = net.variables[name]
    scope = (*var.parents, name)
    domains = tuple(net.variables[v].domain for v in scope)
    table: dict[tuple[str, ...], float] = {}
    import itertools

    for key in itertools.product(*domains):
        assignment = dict(zip(scope, key, strict=True))
        table[key] = net.prob(name, key[-1], assignment)
    return Factor(scope, domains, table)


def train_counts(
    records: Iterable[Mapping[str, str]],
    structure: Iterable[Variable],
    *,
    alpha: float = 1.0,
    meta: dict | None = None,
) -> BayesNet:
    net = BayesNet.from_structure(structure, alpha=alpha, meta=meta)
    for record in records:
        net.add_record(record)
    return net
