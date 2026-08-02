# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the LangChain-backed provider.

The chat model is always a local fake — nothing here touches a network or
a real backend. What is being tested is the *bridge*: that the system
prompt survives into the message list, that LangChain's normalized
usage_metadata reaches request.usage (which is what lets a token cap work
on any backend), that a raising model degrades to an empty response rather
than breaking the SSH session, and that streaming chunks are marshalled
back onto the reactor thread.

Skipped when langchain-core is absent, since it is an optional extra and
CI covers interpreters LangChain does not support. The message classes
come from langchain-core; only _chat_model() needs the heavier `langchain`
package, and these tests inject past it.
"""

from __future__ import annotations

import configparser

from twisted.internet import defer
from twisted.trial import unittest

from cowrie.llm.providers.base import LLMMessage, LLMRequest
from cowrie.llm.providers.langchain_provider import LangChainProvider

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    HAVE_CORE = True
except Exception:  # pragma: no cover - exercised only on installs without it
    HAVE_CORE = False

#: trial spells conditional skipping as a class attribute, not a decorator.
_NEEDS_CORE = None if HAVE_CORE else "langchain-core is not installed"


class _FakeMessage:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage_metadata = usage


class _FakeChatModel:
    """Stands in for whatever init_chat_model would have built."""

    def __init__(self, content="hi", usage=None, raises=None, chunks=None):
        self.content = content
        self.usage = usage
        self.raises = raises
        self.chunks = chunks or []
        self.seen: list = []

    def invoke(self, messages):
        self.seen = messages
        if self.raises:
            raise self.raises
        return _FakeMessage(self.content, self.usage)

    def stream(self, messages):
        self.seen = messages
        if self.raises:
            raise self.raises
        for i, text in enumerate(self.chunks):
            last = i == len(self.chunks) - 1
            yield _FakeMessage(text, self.usage if last else None)


def _provider(chat=None, **overrides) -> LangChainProvider:
    cfg = configparser.ConfigParser()
    cfg["llm"] = {"debug": "false", "langchain_model": "fake:model", **overrides}
    provider = LangChainProvider(cfg)
    if chat is not None:
        provider._chat = chat
    return provider


def _request() -> LLMRequest:
    return LLMRequest(
        system_blocks=[("PERSONA HEAD", True), ("WORLD TAIL", False)],
        messages=[LLMMessage(role="user", content="whoami")],
    )


class TestConfig(unittest.TestCase):
    def test_missing_model_is_a_startup_error(self):
        cfg = configparser.ConfigParser()
        cfg["llm"] = {}
        errors = LangChainProvider.validate_config(cfg)
        self.assertTrue(any("langchain_model" in e for e in errors))

    def test_error_names_the_extra_to_install(self):
        cfg = configparser.ConfigParser()
        cfg["llm"] = {"langchain_model": "ollama:llama3.1"}
        errors = LangChainProvider.validate_config(cfg)
        # Only meaningful when the full package is genuinely absent; when
        # it is installed this correctly reports no errors.
        for message in errors:
            self.assertIn("pip install", message)

    def test_falls_back_to_the_generic_model_key(self):
        cfg = configparser.ConfigParser()
        cfg["llm"] = {"model": "ollama:llama3.1"}
        self.assertEqual(LangChainProvider(cfg).model, "ollama:llama3.1")


class TestMessageTranslation(unittest.TestCase):
    skip = _NEEDS_CORE
    def test_system_blocks_become_a_system_message(self):
        """The trap that would otherwise ship a persona-less honeypot:
        protocol.py never sets request.system, only system_blocks."""
        provider = _provider(_FakeChatModel())
        messages = provider._to_messages(_request())
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIn("PERSONA HEAD", messages[0].content)
        self.assertIn("WORLD TAIL", messages[0].content)

    def test_roles_map_to_langchain_classes(self):
        provider = _provider(_FakeChatModel())
        request = LLMRequest(
            messages=[
                LLMMessage(role="user", content="a"),
                LLMMessage(role="assistant", content="b"),
            ]
        )
        messages = provider._to_messages(request)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)

    def test_no_system_message_when_there_is_no_system_text(self):
        provider = _provider(_FakeChatModel())
        messages = provider._to_messages(
            LLMRequest(messages=[LLMMessage(role="user", content="a")])
        )
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], HumanMessage)


class TestGenerate(unittest.TestCase):
    skip = _NEEDS_CORE
    def test_returns_the_model_text(self):
        provider = _provider(_FakeChatModel(content="root\n"))
        d = provider.generate(_request())
        return d.addCallback(lambda r: self.assertEqual(r, "root\n"))

    def test_usage_metadata_reaches_request_usage(self):
        """This is what makes a per-session token cap work on any backend.

        LangChain normalizes every vendor's counts into usage_metadata, so
        mapping it once here covers backends whose raw shapes the base
        class would not recognize at all.
        """
        chat = _FakeChatModel(
            usage={
                "input_tokens": 120,
                "output_tokens": 8,
                "total_tokens": 128,
                "input_token_details": {"cache_read": 100},
            }
        )
        provider = _provider(chat)
        request = _request()

        def check(_):
            self.assertEqual(request.usage["input_tokens"], 120)
            self.assertEqual(request.usage["output_tokens"], 8)
            self.assertEqual(request.usage["total_tokens"], 128)
            self.assertEqual(request.usage["cached_tokens"], 100)

        return provider.generate(request).addCallback(check)

    def test_a_raising_model_degrades_to_empty_string(self):
        """Provider trouble must never propagate into the SSH session — an
        abrupt disconnect is a louder tell than an empty response."""
        provider = _provider(_FakeChatModel(raises=RuntimeError("backend down")))

        def check(result):
            self.assertEqual(result, "")
            self.flushLoggedErrors()

        return provider.generate(_request()).addCallback(check)

    def test_block_list_content_is_flattened(self):
        """Some backends return content as a list of typed blocks rather
        than a bare string."""
        provider = _provider(_FakeChatModel())
        message = _FakeMessage(
            [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]
        )
        self.assertEqual(provider._text_of(message), "hello")


class TestStreaming(unittest.TestCase):
    skip = _NEEDS_CORE
    def test_chunks_arrive_in_order_and_text_accumulates(self):
        chat = _FakeChatModel(
            chunks=["hel", "lo ", "world"], usage={"total_tokens": 9}
        )
        provider = _provider(chat)
        request = _request()
        chunks: list[str] = []

        def check(result):
            self.assertEqual(result, "hello world")
            self.assertEqual(chunks, ["hel", "lo ", "world"])
            self.assertEqual(request.usage["total_tokens"], 9)

        return provider.generate_streaming(request, chunks.append).addCallback(check)

    def test_streaming_is_advertised(self):
        self.assertTrue(_provider(_FakeChatModel())._supports_streaming())

    def test_a_raising_stream_degrades_to_empty_string(self):
        provider = _provider(_FakeChatModel(raises=RuntimeError("stream died")))

        def check(result):
            self.assertEqual(result, "")
            self.flushLoggedErrors()

        return provider.generate_streaming(_request(), lambda _t: None).addCallback(check)


class TestUsageNormalization(unittest.TestCase):
    """Pure — no langchain import needed."""

    def test_langchain_shape(self):
        out = _provider()._normalize_usage(
            {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        )
        self.assertEqual(out["total_tokens"], 7)

    def test_total_is_derived_when_absent(self):
        out = _provider()._normalize_usage({"input_tokens": 5, "output_tokens": 2})
        self.assertEqual(out["total_tokens"], 7)

    def test_falls_back_to_vendor_shapes(self):
        """A backend LangChain passes through unnormalized still works."""
        out = _provider()._normalize_usage(
            {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
        )
        self.assertEqual(out["total_tokens"], 5)

    def test_empty_usage_yields_nothing_rather_than_a_zero(self):
        """{} must not become {"total_tokens": 0}.

        A backend that reports no usage has to stay distinguishable from
        one that genuinely cost nothing, or a token cap silently treats
        every unmeasured turn as free.
        """
        self.assertEqual(_provider()._normalize_usage({}), {})

    def test_unrecognized_keys_yield_nothing(self):
        self.assertEqual(_provider()._normalize_usage({"whatever": 1}), {})


class TestDeferredContract(unittest.TestCase):
    def test_generate_returns_a_deferred(self):
        provider = _provider(_FakeChatModel())
        d = provider.generate(_request())
        self.assertIsInstance(d, defer.Deferred)
        return d
