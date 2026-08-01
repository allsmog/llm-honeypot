# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests every registered provider must satisfy.

Unlike test_llm_providers.py, which asserts each backend's specific wire
format, this file loops ProviderRegistry.available() — so a provider added
later is covered the moment it registers, without anyone remembering to
write tests for it.

It exists because the traps that bite a new provider are all silent. None
of them raise, none log, and all four produce a honeypot that looks like it
works:

  * Reading ``request.system`` instead of ``request.system_text()`` sends
    no system prompt at all. The interactive protocol only populates
    ``system_blocks``. The model answers anyway — as a generic assistant
    with no persona, no world state and no instructions.
  * Returning a failure rather than "" on an upstream error propagates into
    the SSH session instead of degrading to an empty prompt.
  * Never populating ``request.usage`` makes every turn look free, so cost
    telemetry logs zeros and a per-session token cap can never fire.
  * Opting into streaming with a foreign SSE shape yields a blank response.

What this file deliberately does NOT check: that a provider talks to its
real upstream correctly. Everything here is stubbed. A provider can pass
all of this and still be pointed at the wrong endpoint.
"""

from __future__ import annotations

import configparser

from twisted.trial import unittest

from cowrie.llm.providers import ProviderRegistry
from cowrie.llm.providers.base import LLMMessage, LLMRequest
from cowrie.test.test_llm_providers import StubAgent

#: Credentials for every provider, so construction succeeds and the test
#: exercises the request path rather than a config error. Extend when a
#: provider with a new credential key registers.
_CREDENTIALS = {
    "anthropic_api_key": "sk-ant-test",
    "openai_api_key": "sk-test",
    "api_key": "sk-test",
    "langchain_model": "fake:model",
}

#: Canned successful responses, one per wire shape we speak. A provider
#: understands one of these and ignores the others, so the test tries each
#: rather than needing to know which backend expects what.
#:
#: The JSON blob deliberately merges the Anthropic and OpenAI shapes into a
#: single document; the second is buffered SSE, which the Responses-API
#: provider parses instead of plain JSON.
_CANNED_BODIES = (
    (
        b"{"
        b'"content":[{"type":"text","text":"hi"}],'
        b'"choices":[{"message":{"content":"hi"}}],'
        b'"output":[{"content":[{"type":"output_text","text":"hi"}]}],'
        b'"usage":{"input_tokens":11,"output_tokens":2,'
        b'"prompt_tokens":11,"completion_tokens":2,"total_tokens":13}'
        b"}"
    ),
    (
        b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
        b'data: {"type":"response.completed","response":{"output_text":"hi",'
        b'"usage":{"input_tokens":11,"output_tokens":2,"total_tokens":13}}}\n\n'
    ),
)


def _config(**overrides) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg["llm"] = {"debug": "false", **_CREDENTIALS, **overrides}
    return cfg


def _request() -> LLMRequest:
    return LLMRequest(
        system_blocks=[("STABLEHEAD", True), ("MUTABLETAIL", False)],
        messages=[LLMMessage(role="user", content="whoami")],
    )


def _providers():
    """Every registered provider that can be constructed for testing.

    A provider whose construction needs something we cannot fake (a real
    OAuth token file, a live keychain) is skipped rather than failed — the
    point is to cover what can be covered automatically, not to demand that
    every backend be stubbable.
    """
    out = []
    for name in ProviderRegistry.available():
        try:
            out.append((name, ProviderRegistry.create(name, _config())))
        except Exception:
            continue
    return out


class TestProviderConformance(unittest.TestCase):
    def setUp(self):
        self.providers = _providers()
        self.assertTrue(self.providers, "no providers could be constructed")

    def test_system_prompt_survives_into_the_request_body(self):
        """The trap that makes a honeypot silently persona-less.

        protocol.py sets system_blocks and leaves `system` empty, so a
        provider reading request.system directly ships no system prompt.
        Both block texts must reach the wire, however the provider chooses
        to arrange them.
        """
        for name, provider in self.providers:
            if not hasattr(provider, "_format_body"):
                continue
            try:
                body = repr(provider._format_body(_request()))
            except NotImplementedError:
                continue  # provider bypasses the HTTP body path entirely
            self.assertIn("STABLEHEAD", body, f"{name} dropped the stable head")
            self.assertIn("MUTABLETAIL", body, f"{name} dropped the mutable tail")

    def test_upstream_error_yields_empty_string_not_a_failure(self):
        """Provider trouble must degrade to an empty prompt, never break
        the SSH session — an abrupt disconnect is a louder tell than a
        command that produced no output."""
        for name, provider in self.providers:
            provider.agent = StubAgent(status=500, body=b'{"error":"boom"}')
            result = self.successResultOf(provider.generate(_request()))
            self.assertEqual(result, "", f"{name} did not degrade to empty string")
        self.flushLoggedErrors()

    def test_successful_response_populates_usage(self):
        """Without this, cost telemetry logs zeros forever and a token cap
        cannot fire — and nothing distinguishes 'free' from 'unmeasured'."""
        for name, provider in self.providers:
            reported = False
            for body in _CANNED_BODIES:
                provider.agent = StubAgent(status=200, body=body)
                request = _request()
                self.successResultOf(provider.generate(request))
                reported = reported or bool(request.usage.get("total_tokens"))
            self.assertTrue(
                reported,
                f"{name} reported no token usage for any known response shape",
            )

    def test_streaming_providers_parse_their_own_wire_format(self):
        """A provider that claims streaming must actually decode a stream.

        Opting in with the wrong parser is the blank-shell failure; this
        catches it at the parser level using each provider's own hook.
        """
        from cowrie.llm.providers.streaming import make_streaming_consumer

        anthropic_sse = (
            b'data: {"type":"content_block_delta",'
            b'"delta":{"type":"text_delta","text":"hi"}}\n\n'
        )
        openai_sse = (
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        )
        for name, provider in self.providers:
            if not provider._supports_streaming():
                continue
            parsed_any = False
            for sse in (anthropic_sse, openai_sse):
                chunks: list[str] = []
                consumer, completion = make_streaming_consumer(
                    200, chunks.append, provider._parse_stream_event
                )
                consumer.dataReceived(sse)
                consumer.connectionLost(None)
                text, _usage, _diag = self.successResultOf(completion)
                parsed_any = parsed_any or bool(text)
            self.assertTrue(
                parsed_any,
                f"{name} claims streaming but its parser decoded no known SSE shape",
            )

    def test_validate_config_reports_missing_credentials(self):
        """Fail at startup with a readable message, not at the first
        attacker command with an HTTP 401 and an empty shell."""
        bare = configparser.ConfigParser()
        bare["llm"] = {"debug": "false"}
        for name in ProviderRegistry.available():
            errors = ProviderRegistry.validate(name, bare)
            self.assertIsInstance(errors, list, f"{name}.validate_config")
            for message in errors:
                self.assertIsInstance(message, str)
                self.assertTrue(message.strip(), f"{name} returned a blank error")

    def test_every_provider_declares_a_name(self):
        for name, provider in self.providers:
            self.assertEqual(provider.name, name)


class TestRegistryIsOpen(unittest.TestCase):
    def test_registry_lists_at_least_the_builtin_providers(self):
        """Superset, not equality.

        An exact-set assertion turns the suite red the moment anyone adds
        a provider, which punishes exactly the extension the abstraction
        exists to support. The README recipe never mentioned updating it.
        """
        self.assertLessEqual(
            {"anthropic_apikey", "anthropic_oauth", "codex_apikey", "codex_oauth"},
            set(ProviderRegistry.available()),
        )
