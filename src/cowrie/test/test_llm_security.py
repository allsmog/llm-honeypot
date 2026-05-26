# SPDX-License-Identifier: BSD-3-Clause

"""Audit tests for credential leakage in provider debug logging.

When [llm] debug = true the providers log full request/response bodies
via twisted.python.log.msg. Bodies do not carry auth (headers do), so
no credential should ever land in the log — but this is fragile to
refactors. These tests pin the invariant.

We attach a Twisted log observer, run a stub request, and assert the
captured log text contains neither the literal api-key value nor the
literal OAuth bearer token. If a future change adds e.g. header
debug-dumping, these tests must fail to remind the author to think
twice.
"""

from __future__ import annotations

import configparser
import json
import os
import tempfile
from typing import Any

from twisted.internet import defer
from twisted.python import log as tw_log
from twisted.trial import unittest
from twisted.web.http_headers import Headers

from cowrie.llm.providers.anthropic_apikey import AnthropicAPIKeyProvider
from cowrie.llm.providers.anthropic_oauth import AnthropicOAuthProvider
from cowrie.llm.providers.base import LLMMessage, LLMRequest


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
    def __init__(self, body):
        self.code = 200
        self._body = body
        self.headers = Headers()

    def deliverBody(self, protocol):
        protocol.dataReceived(self._body)
        protocol.connectionLost(None)


class _CapturingAgent:
    """Captures the outgoing request body so we can assert it stays
    credential-free (the body shouldn't contain auth — auth is in
    headers — but this test catches the day someone "helpfully" dumps
    full request state into a debug log)."""

    def __init__(self, body):
        self._body = body
        self.captured_bodies: list[bytes] = []

    def request(self, method, uri, headers=None, bodyProducer=None):
        if bodyProducer is not None:
            consumer = _BytesConsumer()
            bodyProducer.startProducing(consumer)
            self.captured_bodies.append(consumer.bytes)
        return defer.succeed(_StubResponse(self._body))


def _capture_log() -> tuple[list[str], Any]:
    """Returns (captured_strings, observer_fn). The observer formats
    each log event back to text the way the file logger does — that's
    what would actually hit disk if someone scraped cowrie.log."""
    captured: list[str] = []

    def observer(event: dict) -> None:
        # Twisted log events carry either a 'format' template + kwargs
        # or a 'message' tuple. We format both shapes the same way
        # twisted.python.log.textFromEventDict does for files.
        if "format" in event:
            try:
                captured.append(event["format"] % event)
                return
            except Exception:
                pass
        msg = event.get("message")
        if msg:
            captured.append(" ".join(str(m) for m in msg))
        elif "log_format" in event:
            captured.append(event["log_format"])

    return captured, observer


class TestNoCredentialLeak(unittest.TestCase):
    """Assert provider debug logs never contain Bearer tokens or api keys."""

    SENTINEL_APIKEY = "sk-ant-PRIVATE-SENTINEL-9f8e7d6c5b4a3210"
    SENTINEL_OAUTH = "OAUTH-PRIVATE-SENTINEL-9f8e7d6c5b4a3210"

    def _run_with_observer(self, fn):
        captured, observer = _capture_log()
        tw_log.addObserver(observer)
        try:
            fn()
        finally:
            tw_log.removeObserver(observer)
        return "\n".join(captured)

    def test_anthropic_apikey_debug_does_not_leak_key(self):
        cfg = configparser.ConfigParser()
        cfg["llm"] = {
            "debug": "true",
            "anthropic_api_key": self.SENTINEL_APIKEY,
        }
        provider = AnthropicAPIKeyProvider(cfg)
        provider.agent = _CapturingAgent(
            body=b'{"content":[{"type":"text","text":"ok"}]}'
        )

        def fire():
            d = provider.generate(LLMRequest(
                system="x",
                messages=[LLMMessage(role="user", content="hi")],
            ))
            self.successResultOf(d)

        text = self._run_with_observer(fire)
        self.assertNotIn(self.SENTINEL_APIKEY, text,
                         "debug log leaked the literal api key")
        self.assertNotIn("Bearer ", text,
                         "debug log contains a Bearer token marker")
        self.assertNotIn("x-api-key:", text.lower().replace("\n", " "),
                         "debug log contains x-api-key header")
        # And confirm: the body capture also doesn't carry the key.
        body_bytes = b"".join(provider.agent.captured_bodies)
        self.assertNotIn(self.SENTINEL_APIKEY.encode(), body_bytes,
                         "request body contains the api key — should be in header only")

    def test_anthropic_oauth_debug_does_not_leak_token(self):
        # Stage an oauth token file the provider will load.
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump({
                "claudeAiOauth": {
                    "accessToken": self.SENTINEL_OAUTH,
                    "expiresAt": 0,
                },
            }, f)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        cfg = configparser.ConfigParser()
        cfg["llm"] = {
            "debug": "true",
            "anthropic_oauth_token_file": path,
        }
        provider = AnthropicOAuthProvider(cfg)
        provider.agent = _CapturingAgent(
            body=b'{"content":[{"type":"text","text":"ok"}]}'
        )

        def fire():
            d = provider.generate(LLMRequest(
                system="x",
                messages=[LLMMessage(role="user", content="hi")],
            ))
            self.successResultOf(d)

        text = self._run_with_observer(fire)
        self.assertNotIn(self.SENTINEL_OAUTH, text,
                         "debug log leaked the OAuth bearer token")
        self.assertNotIn("Bearer ", text,
                         "debug log contains a Bearer prefix")
        body_bytes = b"".join(provider.agent.captured_bodies)
        self.assertNotIn(self.SENTINEL_OAUTH.encode(), body_bytes,
                         "request body contains the oauth token — should be header only")
