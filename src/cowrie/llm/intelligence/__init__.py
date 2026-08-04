# SPDX-License-Identifier: BSD-3-Clause

"""Offline Bayesian fidelity diagnoser.

Nothing in this package is imported by the live SSH response path — it is
consumed only by scripts/ and tests, the same contract as fidelity.py and
probe_search.py. A test asserts that importing it never pulls in twisted.

The package maps observed harness signals (fidelity invariant groups, probe
search findings, and a state-audit battery) to a posterior over which
subsystem is at fault, using CPTs trained from fault-injection runs against
the real protocol.
"""
