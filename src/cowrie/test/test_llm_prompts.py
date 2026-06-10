# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cowrie.llm.prompts — the hardened behavioral contract.

These are guardrail tests: the system prompt is the highest-leverage
believability lever, so we assert the load-bearing directives are present
and that the template formats cleanly with the variables the protocol
supplies (a stray brace would crash every session)."""

from __future__ import annotations

from twisted.trial import unittest

from cowrie.llm import prompts


class TestInteractivePrompt(unittest.TestCase):
    def test_formats_with_protocol_variables(self):
        # Must not raise on the exact key set the protocol passes.
        out = prompts.INTERACTIVE_SYSTEM_PROMPT.format_map(
            {
                "hostname": "web01", "username": "root", "ip": "1.2.3.4",
                "ip6": "", "client_ip": "5.6.7.8", "cwd": "/root",
            }
        )
        self.assertIn("web01", out)
        self.assertIn("root", out)

    def test_contains_output_discipline(self):
        p = prompts.INTERACTIVE_SYSTEM_PROMPT
        self.assertIn("ONLY", p)
        self.assertIn("markdown", p.lower())
        self.assertIn("prompt", p.lower())

    def test_contains_anti_break_character(self):
        p = prompts.INTERACTIVE_SYSTEM_PROMPT.lower()
        self.assertIn("never reveal", p)
        self.assertTrue("ai" in p or "model" in p or "simulation" in p)

    def test_contains_consistency_and_error_fidelity(self):
        p = prompts.INTERACTIVE_SYSTEM_PROMPT
        self.assertIn("ground truth", p)
        self.assertIn("command not found", p)
        self.assertIn("Permission denied", p)

    def test_mentions_interactive_programs(self):
        p = prompts.INTERACTIVE_SYSTEM_PROMPT.lower()
        for prog in ("top", "vim", "tail -f"):
            self.assertIn(prog, p)


class TestExecPrompt(unittest.TestCase):
    def test_exec_prompt_is_terse_and_strict(self):
        p = prompts.EXEC_SYSTEM_PROMPT
        self.assertIn("ONLY", p)
        self.assertIn("command not found", p)

    def test_exec_prompt_formats(self):
        # The exec prompt currently uses no variables, but must remain
        # format-safe (no stray braces).
        try:
            prompts.EXEC_SYSTEM_PROMPT.format_map({})
        except Exception as e:  # pragma: no cover
            self.fail(f"exec prompt failed to format: {e}")
