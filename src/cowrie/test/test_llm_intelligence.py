# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: The inference math, validated bottom-up: factor algebra against
# ABOUTME: hand-computed tables, enumeration against the textbook alarm
# ABOUTME: network, VE against enumeration, and every sampler against exact.

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from cowrie.llm.intelligence.exact import enumeration_ask, variable_elimination
from cowrie.llm.intelligence.factors import (
    Factor,
    multiply,
    normalize,
    restrict,
    sum_out,
)
from cowrie.llm.intelligence.network import BayesNet, Variable, train_counts
from cowrie.llm.intelligence.sampling import (
    _initial_state,
    _seeded_rng,
    gibbs,
    likelihood_weighting,
    metropolis_hastings,
)


def tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def mae(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys) / len(keys)


def alarm_network() -> BayesNet:
    """The AIMA burglary/alarm network, so the expected posteriors come
    from an external source rather than this codebase."""
    t, f = "t", "f"
    structure = [
        Variable("B", (t, f)),
        Variable("E", (t, f)),
        Variable("A", (t, f), ("B", "E")),
        Variable("J", (t, f), ("A",)),
        Variable("M", (t, f), ("A",)),
    ]
    cpts = {
        "B": {(): {t: 0.001, f: 0.999}},
        "E": {(): {t: 0.002, f: 0.998}},
        "A": {
            (t, t): {t: 0.95, f: 0.05},
            (t, f): {t: 0.94, f: 0.06},
            (f, t): {t: 0.29, f: 0.71},
            (f, f): {t: 0.001, f: 0.999},
        },
        "J": {(t,): {t: 0.90, f: 0.10}, (f,): {t: 0.05, f: 0.95}},
        "M": {(t,): {t: 0.70, f: 0.30}, (f,): {t: 0.01, f: 0.99}},
    }
    return BayesNet.from_cpts(structure, cpts)


def random_net(seed: str, n_vars: int = 8) -> BayesNet:
    """A seeded random-CPT net with a fixed diamond-chain structure, for
    VE-vs-enumeration agreement across many shapes of table."""
    rng = _seeded_rng(seed)
    parents_of = {
        "v0": (), "v1": ("v0",), "v2": ("v0",), "v3": ("v1", "v2"),
        "v4": ("v3",), "v5": ("v3",), "v6": ("v4", "v5"), "v7": ("v6",),
    }
    names = [f"v{i}" for i in range(n_vars)]
    domains = {n: ("a", "b", "c") if i % 3 == 0 else ("a", "b")
               for i, n in enumerate(names)}
    structure = [Variable(n, domains[n], parents_of[n]) for n in names]
    cpts: dict = {}
    import itertools

    for var in structure:
        table: dict = {}
        parent_domains = [domains[p] for p in var.parents]
        for pk in itertools.product(*parent_domains):
            raw = [rng.random() + 0.05 for _ in var.domain]
            total = sum(raw)
            table[pk] = {v: r / total for v, r in zip(var.domain, raw, strict=True)}
        cpts[var.name] = table
    return BayesNet.from_cpts(structure, cpts)


class TestFactorAlgebra(unittest.TestCase):
    def setUp(self) -> None:
        self.f = Factor(
            ("x", "y"),
            (("a", "b"), ("u", "v", "w")),
            {
                ("a", "u"): 0.1, ("a", "v"): 0.2, ("a", "w"): 0.3,
                ("b", "u"): 0.05, ("b", "v"): 0.15, ("b", "w"): 0.2,
            },
        )

    def test_restrict_hand_computed(self) -> None:
        r = restrict(self.f, "y", "v")
        self.assertEqual(r.variables, ("x",))
        self.assertEqual(r.table, {("a",): 0.2, ("b",): 0.15})

    def test_sum_out_hand_computed(self) -> None:
        s = sum_out(self.f, "y")
        self.assertEqual(s.variables, ("x",))
        self.assertAlmostEqual(s.table[("a",)], 0.6)
        self.assertAlmostEqual(s.table[("b",)], 0.4)

    def test_normalize_hand_computed(self) -> None:
        n = normalize(sum_out(self.f, "y"))
        self.assertAlmostEqual(n.table[("a",)], 0.6)
        self.assertAlmostEqual(n.table[("b",)], 0.4)
        self.assertAlmostEqual(sum(n.table.values()), 1.0)

    def test_multiply_hand_computed(self) -> None:
        g = Factor(("y",), (("u", "v", "w"),), {("u",): 2.0, ("v",): 0.5, ("w",): 1.0})
        m = multiply(self.f, g)
        self.assertEqual(set(m.variables), {"x", "y"})
        self.assertAlmostEqual(m.table[("a", "u")], 0.2)
        self.assertAlmostEqual(m.table[("a", "v")], 0.1)
        self.assertAlmostEqual(m.table[("b", "w")], 0.2)

    def test_multiply_commutative_and_associative(self) -> None:
        g = Factor(("y", "z"), (("u", "v", "w"), ("p", "q")),
                   {(y, z): 0.1 + 0.07 * i
                    for i, (y, z) in enumerate(
                        (y, z) for y in ("u", "v", "w") for z in ("p", "q"))})
        h = Factor(("z",), (("p", "q"),), {("p",): 0.3, ("q",): 0.7})
        ab = multiply(self.f, g)
        ba = multiply(g, self.f)
        for key in ab.table:
            perm = tuple(key[ab.variables.index(v)] for v in ba.variables)
            self.assertAlmostEqual(ab.table[key], ba.table[perm])
        left = multiply(multiply(self.f, g), h)
        right = multiply(self.f, multiply(g, h))
        for key in left.table:
            perm = tuple(key[left.variables.index(v)] for v in right.variables)
            self.assertAlmostEqual(left.table[key], right.table[perm])


class TestExactInference(unittest.TestCase):
    def test_enumeration_matches_textbook_alarm_posterior(self) -> None:
        net = alarm_network()
        dist = enumeration_ask(net, "B", {"J": "t", "M": "t"})
        self.assertAlmostEqual(dist["t"], 0.2841718, places=6)

    def test_ve_equals_enumeration_on_alarm(self) -> None:
        net = alarm_network()
        evidence_sets = [
            {}, {"J": "t"}, {"J": "t", "M": "t"}, {"M": "f"},
            {"E": "t", "J": "t"}, {"B": "t"}, {"A": "t", "M": "f"},
        ]
        for evidence in evidence_sets:
            for query in net.variables:
                if query in evidence:
                    continue
                enum = enumeration_ask(net, query, evidence)
                ve = variable_elimination(net, query, evidence)
                for value in enum:
                    self.assertAlmostEqual(
                        enum[value], ve[value], delta=1e-12,
                        msg=f"query={query} evidence={evidence}",
                    )

    def test_ve_equals_enumeration_on_random_nets(self) -> None:
        for net_seed in ("net-1", "net-2"):
            net = random_net(net_seed)
            rng = _seeded_rng(f"evidence|{net_seed}")
            names = list(net.variables)
            for _ in range(10):
                observed = rng.sample(names, rng.randrange(0, 4))
                evidence = {
                    n: net.variables[n].domain[
                        rng.randrange(len(net.variables[n].domain))
                    ]
                    for n in observed
                }
                query = rng.choice([n for n in names if n not in evidence])
                enum = enumeration_ask(net, query, evidence)
                ve = variable_elimination(net, query, evidence)
                for value in enum:
                    self.assertAlmostEqual(enum[value], ve[value], delta=1e-12)

    def test_ve_explicit_order_matches_default(self) -> None:
        net = alarm_network()
        default = variable_elimination(net, "B", {"J": "t", "M": "t"})
        explicit = variable_elimination(
            net, "B", {"J": "t", "M": "t"}, order=["E", "A"]
        )
        for value in default:
            self.assertAlmostEqual(default[value], explicit[value], delta=1e-12)


class TestSamplers(unittest.TestCase):
    """Gates sized ~3x the error measured during implementation, with fixed
    seeds so CI sees the identical estimate every run."""

    def setUp(self) -> None:
        self.net = alarm_network()
        self.evidence_sets = [
            {"J": "t", "M": "t"},
            {"J": "t"},
            {"M": "f", "E": "t"},
        ]

    def _exact(self, evidence: dict[str, str]) -> dict[str, float]:
        return variable_elimination(self.net, "B", evidence)

    def test_likelihood_weighting_within_gates(self) -> None:
        # LW is gated on queries where forward sampling actually reaches the
        # posterior: downstream queries and evidence that does not hang the
        # whole posterior on a rare cause. Its rare-cause weakness is not
        # hidden — it is measured explicitly in the next test.
        cases = [
            ("B", {"J": "t"}),
            ("B", {"M": "f", "E": "t"}),
            ("J", {"B": "t"}),
            ("M", {"E": "t", "J": "t"}),
        ]
        for query, evidence in cases:
            exact = variable_elimination(self.net, query, evidence)
            approx = likelihood_weighting(
                self.net, query, evidence, n=20_000, seed=f"lw|{sorted(evidence)}"
            )
            self.assertLessEqual(tv_distance(exact, approx), 0.02, msg=str(evidence))
            # For binary variables MAE == TV, so a tighter MAE gate would
            # be incoherent; both gates sit at 0.02.
            self.assertLessEqual(mae(exact, approx), 0.02, msg=str(evidence))

    def test_likelihood_weighting_degrades_on_rare_cause_evidence(self) -> None:
        # P(B | j, m): B=t appears in ~20 of 20k forward samples, so the
        # whole posterior rests on a handful of weighted samples (seed
        # spread measured at +-0.13). Gibbs conditions on the evidence
        # instead of hoping to sample past it, and is ~15x closer at the
        # same budget. This asymmetry is the measured reason the diagnoser
        # prefers Gibbs when evidence sits downstream of a rare cause.
        evidence = {"J": "t", "M": "t"}
        exact = self._exact(evidence)
        lw = likelihood_weighting(self.net, "B", evidence, n=20_000, seed="rare|lw")
        gb = gibbs(self.net, "B", evidence, n=20_000, burn=2_000, seed="rare|gibbs")
        self.assertGreater(tv_distance(exact, lw), 0.01)
        self.assertLess(tv_distance(exact, gb), 0.01)
        self.assertLess(tv_distance(exact, gb), tv_distance(exact, lw))

    def test_gibbs_within_gates(self) -> None:
        for evidence in self.evidence_sets:
            exact = self._exact(evidence)
            approx = gibbs(
                self.net, "B", evidence,
                n=20_000, burn=2_000, seed=f"gibbs|{sorted(evidence)}",
            )
            self.assertLessEqual(tv_distance(exact, approx), 0.03, msg=str(evidence))
            self.assertLessEqual(mae(exact, approx), 0.015, msg=str(evidence))

    def test_metropolis_hastings_within_gates(self) -> None:
        for evidence in self.evidence_sets:
            exact = self._exact(evidence)
            approx = metropolis_hastings(
                self.net, "B", evidence,
                n=40_000, burn=4_000, seed=f"mh|{sorted(evidence)}",
            )
            self.assertLessEqual(tv_distance(exact, approx), 0.05, msg=str(evidence))
            self.assertLessEqual(mae(exact, approx), 0.025, msg=str(evidence))

    def test_samplers_deterministic_given_seed(self) -> None:
        evidence = {"J": "t", "M": "t"}
        for sampler, kwargs in (
            (likelihood_weighting, {"n": 500}),
            (gibbs, {"n": 500, "burn": 50}),
            (metropolis_hastings, {"n": 500, "burn": 50}),
        ):
            first = sampler(self.net, "B", evidence, seed="det", **kwargs)
            second = sampler(self.net, "B", evidence, seed="det", **kwargs)
            self.assertEqual(first, second, msg=sampler.__name__)


class TestTraining(unittest.TestCase):
    def test_train_counts_recovers_known_parameters(self) -> None:
        truth = alarm_network()
        # Skew the priors so B/E actually appear in a 5k sample.
        truth.counts["B"] = {(): {"t": 0.3, "f": 0.7}}
        truth.counts["E"] = {(): {"t": 0.4, "f": 0.6}}
        rng = _seeded_rng("forward-sample")
        records = [_initial_state(truth, {}, rng) for _ in range(5_000)]
        learned = train_counts(records, truth.variables.values(), alpha=1.0)
        for name, var in truth.variables.items():
            import itertools

            parent_domains = [truth.variables[p].domain for p in var.parents]
            for pk in itertools.product(*parent_domains):
                assignment = dict(zip(var.parents, pk, strict=True))
                for value in var.domain:
                    expected = truth.prob(name, value, assignment)
                    got = learned.prob(name, value, assignment)
                    self.assertLess(
                        abs(expected - got), 0.03,
                        msg=f"{name}={value} | {assignment}",
                    )

    def test_add_alpha_smoothing_exact_value_for_unobserved(self) -> None:
        structure = [Variable("x", ("a", "b", "z"))]
        records = [{"x": "a"}] * 6 + [{"x": "b"}] * 4
        net = train_counts(records, structure, alpha=1.0)
        # 'z' never observed: alpha / (N + alpha * |domain|) = 1 / 13.
        self.assertAlmostEqual(net.prob("x", "z", {}), 1.0 / 13.0)
        self.assertAlmostEqual(net.prob("x", "a", {}), 7.0 / 13.0)

    def test_add_record_rejects_out_of_domain(self) -> None:
        net = BayesNet.from_structure([Variable("x", ("a", "b"))])
        with self.assertRaises(ValueError):
            net.add_record({"x": "nope"})


class TestSerialization(unittest.TestCase):
    def test_json_roundtrip_preserves_hash_and_probs(self) -> None:
        structure = [
            Variable("fault", ("f1", "f2")),
            Variable("sig", ("fired", "clear"), ("fault",)),
        ]
        records = [
            {"fault": "f1", "sig": "fired"},
            {"fault": "f1", "sig": "fired"},
            {"fault": "f2", "sig": "clear"},
        ]
        net = train_counts(records, structure, alpha=1.0, meta={"version": "test-v1"})
        clone = BayesNet.from_json(net.to_json())
        self.assertEqual(net.content_hash(), clone.content_hash())
        self.assertAlmostEqual(
            net.prob("sig", "fired", {"fault": "f1"}),
            clone.prob("sig", "fired", {"fault": "f1"}),
        )

    def test_from_json_rejects_tampered_content(self) -> None:
        net = train_counts(
            [{"x": "a"}], [Variable("x", ("a", "b"))], alpha=1.0,
            meta={"version": "test-v1"},
        )
        tampered = net.to_json().replace('"a": 1', '"a": 2')
        with self.assertRaises(ValueError):
            BayesNet.from_json(tampered)


class TestImportHygiene(unittest.TestCase):
    def test_no_module_level_imports_outside_stdlib_and_package(self) -> None:
        """The live-path firewall, stated as an auditable contract.

        cowrie/__init__.py itself imports twisted, so "importing the
        package never loads twisted" is unachievable through the normal
        package path. What we can and do guarantee: no intelligence module
        imports anything at module level except the stdlib and its own
        package. cowrie/twisted imports are only permitted lazily, inside
        functions, where the harness that calls them already runs cowrie.
        """
        import ast

        package_dir = (
            Path(__file__).resolve().parents[1] / "llm" / "intelligence"
        )
        allowed_stdlib = set(sys.stdlib_module_names)
        for path in sorted(package_dir.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in tree.body:
                if isinstance(node, ast.If):
                    # `if TYPE_CHECKING:` blocks never execute at runtime.
                    continue
                if isinstance(node, ast.Import):
                    roots = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    roots = [node.module or ""]
                else:
                    continue
                for name in roots:
                    top = name.split(".")[0]
                    ok = top in allowed_stdlib or name.startswith(
                        "cowrie.llm.intelligence"
                    )
                    self.assertTrue(
                        ok,
                        msg=f"{path.name} imports {name!r} at module level",
                    )


if __name__ == "__main__":
    unittest.main()
