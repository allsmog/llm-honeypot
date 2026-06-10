# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for cowrie.llm.providers — registration, body framing per
provider, response parsing, validate_config, and 401-retry semantics.

HTTP is stubbed end-to-end. The provider.agent attribute is replaced
post-construction with a StubAgent that captures request bodies and
returns canned responses, so the deferred chain executes synchronously
under twisted.trial without any real I/O.
"""

from __future__ import annotations

import configparser
import json

from twisted.internet import defer
from twisted.trial import unittest
from twisted.web.http_headers import Headers

from cowrie.llm.providers import ProviderRegistry
from cowrie.llm.providers.anthropic_apikey import AnthropicAPIKeyProvider
from cowrie.llm.providers.anthropic_oauth import AnthropicOAuthProvider
from cowrie.llm.providers.base import LLMMessage, LLMRequest
from cowrie.llm.providers.codex_apikey import CodexAPIKeyProvider
from cowrie.llm.providers.codex_oauth import CodexOAuthProvider

# ----------------------------------------------------------------------
# Stub HTTP plumbing


class _BytesConsumer:
    def __init__(self):
        self.bytes = b""

    def write(self, data):
        self.bytes += data

    def registerProducer(self, *a, **kw):
        pass

    def unregisterProducer(self):
        pass


class _StubResponse:
    def __init__(self, code, body):
        self.code = code
        self._body = body
        self.headers = Headers()

    def deliverBody(self, protocol):
        protocol.dataReceived(self._body)
        protocol.connectionLost(None)


class StubAgent:
    """Captures the outgoing request body + returns canned responses.

    By default replies with status=200 and body. Pass a list of
    (status, body) tuples to ``responses`` to script a sequence
    (used for the 401-retry test).
    """

    def __init__(self, status=200, body=b'{"content":[{"type":"text","text":"hi"}]}',
                 responses=None):
        self.requests = []
        if responses is None:
            self._responses = [(status, body)]
        else:
            self._responses = list(responses)

    def request(self, method, uri, headers=None, bodyProducer=None):
        captured = {"method": method, "uri": uri, "headers": headers, "body": b""}
        if bodyProducer is not None:
            consumer = _BytesConsumer()
            bodyProducer.startProducing(consumer)
            captured["body"] = consumer.bytes
        self.requests.append(captured)
        # If sequence exhausted, repeat the last one.
        status, body = self._responses.pop(0) if self._responses else (200, b"{}")
        return defer.succeed(_StubResponse(status, body))


def _config(overrides: dict[str, str]) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg["llm"] = {"debug": "false", **overrides}
    return cfg


# ----------------------------------------------------------------------
# Registry


class TestRegistry(unittest.TestCase):
    def test_all_four_providers_registered(self):
        self.assertEqual(
            set(ProviderRegistry.available()),
            {"anthropic_apikey", "anthropic_oauth", "codex_apikey", "codex_oauth"},
        )

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            ProviderRegistry.create("not_a_provider", _config({}))


# ----------------------------------------------------------------------
# validate_config


class TestValidateConfig(unittest.TestCase):
    def test_anthropic_apikey_missing_key(self):
        errors = ProviderRegistry.validate("anthropic_apikey", _config({}))
        self.assertEqual(len(errors), 1)
        self.assertIn("anthropic_api_key", errors[0])

    def test_anthropic_apikey_ok_with_key(self):
        errors = ProviderRegistry.validate(
            "anthropic_apikey", _config({"anthropic_api_key": "sk-ant-xxx"}),
        )
        self.assertEqual(errors, [])

    def test_anthropic_apikey_accepts_legacy_api_key(self):
        # Migration path from upstream Cowrie's single-provider [llm].
        errors = ProviderRegistry.validate(
            "anthropic_apikey", _config({"api_key": "sk-ant-xxx"}),
        )
        self.assertEqual(errors, [])

    def test_codex_apikey_missing_key(self):
        errors = ProviderRegistry.validate("codex_apikey", _config({}))
        self.assertEqual(len(errors), 1)
        self.assertIn("openai_api_key", errors[0])

    def test_codex_oauth_missing_token_file(self):
        errors = ProviderRegistry.validate(
            "codex_oauth", _config({"codex_oauth_token_file": "/nonexistent/path"}),
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("/nonexistent/path", errors[0])


# ----------------------------------------------------------------------
# AnthropicAPIKeyProvider body framing


class TestAnthropicAPIKeyBody(unittest.TestCase):
    def _make(self, **kw):
        cfg = _config({"anthropic_api_key": "sk-test", **kw})
        provider = AnthropicAPIKeyProvider(cfg)
        provider.agent = StubAgent(
            body=b'{"content":[{"type":"text","text":"hello"}]}'
        )
        return provider

    def test_includes_cache_control_by_default(self):
        provider = self._make()
        request = LLMRequest(
            system="persona block",
            messages=[LLMMessage(role="user", content="echo hi")],
        )
        d = provider.generate(request)
        result = self.successResultOf(d)
        self.assertEqual(result, "hello")
        body = json.loads(provider.agent.requests[0]["body"])
        self.assertIsInstance(body["system"], list)
        self.assertEqual(body["system"][0]["cache_control"], {"type": "ephemeral"})

    def test_omits_cache_control_when_disabled(self):
        provider = self._make(anthropic_cache_system="false")
        request = LLMRequest(system="x", messages=[LLMMessage(role="user", content="hi")])
        provider.generate(request)
        body = json.loads(provider.agent.requests[0]["body"])
        # When caching is disabled, system is a plain string.
        self.assertIsInstance(body["system"], str)

    def test_system_blocks_emit_per_block_cache_control(self):
        provider = self._make()
        request = LLMRequest(
            system_blocks=[("stable head", True), ("mutable tail", False)],
            messages=[LLMMessage(role="user", content="hi")],
        )
        provider.generate(request)
        body = json.loads(provider.agent.requests[0]["body"])
        self.assertEqual(len(body["system"]), 2)
        self.assertEqual(body["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", body["system"][1])

    def test_parse_response_returns_text_block(self):
        provider = self._make()
        request = LLMRequest(system="x", messages=[LLMMessage(role="user", content="hi")])
        result = self.successResultOf(provider.generate(request))
        self.assertEqual(result, "hello")

    def test_parse_response_empty_on_malformed_payload(self):
        provider = self._make()
        provider.agent = StubAgent(body=b'{"unexpected":"shape"}')
        request = LLMRequest(system="x", messages=[LLMMessage(role="user", content="hi")])
        result = self.successResultOf(provider.generate(request))
        self.assertEqual(result, "")


# ----------------------------------------------------------------------
# CodexAPIKeyProvider body framing


class TestCodexAPIKeyBody(unittest.TestCase):
    def test_uses_chat_completions_shape(self):
        cfg = _config({"openai_api_key": "sk-openai"})
        provider = CodexAPIKeyProvider(cfg)
        provider.agent = StubAgent(
            body=b'{"choices":[{"message":{"content":"ok"}}]}'
        )
        request = LLMRequest(
            system="sys",
            messages=[LLMMessage(role="user", content="cmd")],
        )
        result = self.successResultOf(provider.generate(request))
        self.assertEqual(result, "ok")
        body = json.loads(provider.agent.requests[0]["body"])
        self.assertEqual(body["model"], "gpt-4o-mini")
        # Chat-completions shape: system as a message, not a top-level field.
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][0]["content"], "sys")
        self.assertEqual(body["messages"][1]["role"], "user")

    def test_system_blocks_concatenated(self):
        cfg = _config({"openai_api_key": "sk-openai"})
        provider = CodexAPIKeyProvider(cfg)
        provider.agent = StubAgent(body=b'{"choices":[{"message":{"content":"x"}}]}')
        request = LLMRequest(
            system_blocks=[("HEAD", True), ("TAIL", False)],
            messages=[LLMMessage(role="user", content="cmd")],
        )
        provider.generate(request)
        body = json.loads(provider.agent.requests[0]["body"])
        # Two blocks join into one system message; no cache_control field
        # (chat-completions doesn't model per-block caching).
        self.assertIn("HEAD", body["messages"][0]["content"])
        self.assertIn("TAIL", body["messages"][0]["content"])


# ----------------------------------------------------------------------
# CodexOAuthProvider body framing


class TestCodexOAuthBody(unittest.TestCase):
    def _make(self, tmp_path_token: str | None = None):
        # Synthesize a Codex CLI auth.json on disk for the provider to read.
        import os
        import tempfile

        path = tmp_path_token or tempfile.mkstemp(suffix=".json")[1]
        with open(path, "w") as f:
            json.dump({"tokens": {"access_token": "fake-codex-token"}}, f)
        self.addCleanup(lambda: os.unlink(path) if os.path.exists(path) else None)
        cfg = _config({"codex_oauth_token_file": path})
        provider = CodexOAuthProvider(cfg)
        # SSE body — one delta plus completed event.
        sse_body = (
            b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
            b'data: {"type":"response.completed","response":{"output_text":"hi"}}\n\n'
        )
        provider.agent = StubAgent(body=sse_body)  # type: ignore[assignment]
        return provider

    def test_uses_responses_api_shape(self):
        provider = self._make()
        request = LLMRequest(
            system="instructions text",
            messages=[LLMMessage(role="user", content="cmd")],
        )
        result = self.successResultOf(provider.generate(request))
        self.assertEqual(result, "hi")
        body = json.loads(provider.agent.requests[0]["body"])
        # Responses API uses `instructions` + `input`, not `messages`.
        self.assertEqual(body["instructions"], "instructions text")
        self.assertEqual(body["input"][0]["content"], "cmd")
        # Both required flags must be set.
        self.assertEqual(body["store"], False)
        self.assertEqual(body["stream"], True)
        # max_output_tokens / temperature must NOT be sent (Codex rejects them).
        self.assertNotIn("max_output_tokens", body)
        self.assertNotIn("temperature", body)


# ----------------------------------------------------------------------
# 401-retry mechanics


class TestAnthropicOAuthHeaderConfig(unittest.TestCase):
    """Verify the anthropic-beta header is config-overridable.

    Anthropic bumps the beta name periodically; operators need to update
    without a code change. Default is exercised in
    TestAuthReload.test_anthropic_oauth_reloads_*; here we just verify
    the override is picked up.
    """

    def _make_with_token(self, **extra_config) -> tuple[AnthropicOAuthProvider, str]:
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump(
                {"claudeAiOauth": {"accessToken": "tok-1", "expiresAt": 0}}, f,
            )
        self.addCleanup(lambda: os.unlink(path) if os.path.exists(path) else None)

        cfg = _config({"anthropic_oauth_token_file": path, **extra_config})
        return AnthropicOAuthProvider(cfg), path

    def test_default_beta_header(self):
        provider, _ = self._make_with_token()
        provider.agent = StubAgent(
            body=b'{"content":[{"type":"text","text":"ok"}]}'
        )
        provider.generate(LLMRequest(
            system="x", messages=[LLMMessage(role="user", content="hi")],
        ))
        headers = provider.agent.requests[0]["headers"]
        beta = headers.getRawHeaders(b"anthropic-beta")
        self.assertEqual(beta, [b"oauth-2025-04-20"])

    def test_overridden_beta_header(self):
        provider, _ = self._make_with_token(
            anthropic_oauth_beta="oauth-2099-12-31",
        )
        provider.agent = StubAgent(
            body=b'{"content":[{"type":"text","text":"ok"}]}'
        )
        provider.generate(LLMRequest(
            system="x", messages=[LLMMessage(role="user", content="hi")],
        ))
        headers = provider.agent.requests[0]["headers"]
        beta = headers.getRawHeaders(b"anthropic-beta")
        self.assertEqual(beta, [b"oauth-2099-12-31"])


class TestAuthReload(unittest.TestCase):
    def test_anthropic_apikey_no_retry_default(self):
        # The base class's _on_auth_failure returns False — API-key
        # providers don't retry. Verify a 401 produces an empty string
        # and only one request was sent.
        cfg = _config({"anthropic_api_key": "sk-bad"})
        provider = AnthropicAPIKeyProvider(cfg)
        provider.agent = StubAgent(status=401, body=b'{"error":"unauthorized"}')
        request = LLMRequest(system="x", messages=[LLMMessage(role="user", content="hi")])
        result = self.successResultOf(provider.generate(request))
        self.assertEqual(result, "")
        self.assertEqual(len(provider.agent.requests), 1)

    def test_anthropic_oauth_reloads_and_retries_when_token_changes(self):
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        # Write initial token.
        with open(path, "w") as f:
            json.dump({"claudeAiOauth": {"accessToken": "tok-old", "expiresAt": 0}}, f)
        self.addCleanup(lambda: os.unlink(path) if os.path.exists(path) else None)

        cfg = _config({"anthropic_oauth_token_file": path})
        provider = AnthropicOAuthProvider(cfg)

        # Stage 401 (triggers reload) then 200 (success on retry).
        success_body = b'{"content":[{"type":"text","text":"ok"}]}'
        provider.agent = StubAgent(
            responses=[(401, b'{"error":"bad token"}'), (200, success_body)],
        )

        # While the first request is being made, rotate the token on disk
        # so the reload picks up a *different* value (this is the contract:
        # same-token reloads don't retry).
        original_request = provider.agent.request

        def request_then_rotate(*args, **kwargs):
            d = original_request(*args, **kwargs)
            if len(provider.agent.requests) == 1:
                with open(path, "w") as f:
                    json.dump(
                        {"claudeAiOauth": {"accessToken": "tok-new", "expiresAt": 0}},
                        f,
                    )
            return d

        provider.agent.request = request_then_rotate

        request = LLMRequest(system="x", messages=[LLMMessage(role="user", content="hi")])
        result = self.successResultOf(provider.generate(request))
        self.assertEqual(result, "ok")
        # Two requests went out: one 401, one retry with the fresh token.
        self.assertEqual(len(provider.agent.requests), 2)

    def test_anthropic_oauth_no_retry_when_token_unchanged(self):
        # When _load_token returns the same value the provider already
        # has cached, _on_auth_failure must return False — otherwise a
        # truly-bad token causes an infinite retry loop.
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump({"claudeAiOauth": {"accessToken": "tok-x", "expiresAt": 0}}, f)
        self.addCleanup(lambda: os.unlink(path) if os.path.exists(path) else None)

        cfg = _config({"anthropic_oauth_token_file": path})
        provider = AnthropicOAuthProvider(cfg)
        # 401 on first request; if the retry mechanism kicked in we'd see
        # a second request (and run out of canned responses).
        provider.agent = StubAgent(
            responses=[(401, b'{"error":"bad token"}')],
        )

        request = LLMRequest(system="x", messages=[LLMMessage(role="user", content="hi")])
        result = self.successResultOf(provider.generate(request))
        self.assertEqual(result, "")
        self.assertEqual(len(provider.agent.requests), 1)
