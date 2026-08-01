# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cowrie.llm.providers.streaming — the SSE consumer that
fires per-delta callbacks instead of buffering the full body."""

from __future__ import annotations

from twisted.trial import unittest

from cowrie.llm.providers.streaming import (
    make_streaming_consumer,
    parse_anthropic_event,
    parse_openai_event,
)


class TestStreamingBodyConsumer(unittest.TestCase):
    def test_fires_callback_per_text_delta(self):
        chunks: list[str] = []
        consumer, completion = make_streaming_consumer(200, chunks.append)

        # Anthropic stream wire format. Feed the SSE byte stream in
        # chunks to simulate network arrival.
        sse = (
            b'event: message_start\n'
            b'data: {"type":"message_start","message":{"usage":{"input_tokens":42,"output_tokens":0}}}\n'
            b'\n'
            b'event: content_block_delta\n'
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hello "}}\n'
            b'\n'
            b'event: content_block_delta\n'
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"world"}}\n'
            b'\n'
            b'event: message_delta\n'
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}\n'
            b'\n'
            b'event: message_stop\n'
            b'data: {"type":"message_stop"}\n'
            b'\n'
        )
        # Deliver in 32-byte chunks to exercise the buffering across
        # network-boundary splits.
        for i in range(0, len(sse), 32):
            consumer.dataReceived(sse[i:i + 32])
        consumer.connectionLost(None)

        # Callback fired once per delta.
        self.assertEqual(chunks, ["hello ", "world"])
        # Completion deferred fires with (full_text, usage_dict, diagnosis).
        full_text, usage, diagnosis = self.successResultOf(completion)
        self.assertEqual(full_text, "hello world")
        self.assertEqual(diagnosis, "", "healthy stream must report no diagnosis")
        # The merged usage contains both the message_start input count
        # and the message_delta output count.
        self.assertEqual(usage.get("input_tokens"), 42)
        self.assertEqual(usage.get("output_tokens"), 7)

    def test_ignores_non_data_lines(self):
        chunks: list[str] = []
        consumer, _completion = make_streaming_consumer(200, chunks.append)
        consumer.dataReceived(
            b'event: ping\n\n'
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n'
        )
        consumer.connectionLost(None)
        self.assertEqual(chunks, ["hi"])

    def test_malformed_json_in_data_line_is_skipped(self):
        chunks: list[str] = []
        consumer, _completion = make_streaming_consumer(200, chunks.append)
        consumer.dataReceived(
            b'data: {invalid json}\n\n'
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}\n\n'
        )
        consumer.connectionLost(None)
        self.assertEqual(chunks, ["ok"])

    def test_empty_stream_completes_with_empty_text(self):
        chunks: list[str] = []
        consumer, completion = make_streaming_consumer(200, chunks.append)
        consumer.connectionLost(None)
        full_text, usage, diagnosis = self.successResultOf(completion)
        self.assertEqual(full_text, "")
        self.assertEqual(usage, {})
        # An empty stream must be reported, not silently served.
        self.assertEqual(diagnosis, "no SSE events decoded")


class TestWireFormatMismatch(unittest.TestCase):
    """The failure this whole hook exists to make visible.

    Before parse_event was pluggable, a provider whose SSE shape was not
    Anthropic's produced a perfectly successful stream containing no text.
    The attacker saw a blank line and nothing was logged anywhere — a
    honeypot that is simultaneously broken and invisible. These tests pin
    that such a stream is now *reported*, so base.generate_streaming can
    fall back to a buffered call rather than serve the blank.
    """

    #: One OpenAI chat-completions chunk. Note there is no "type" key at
    #: all, which is exactly why the Anthropic parser matches nothing.
    OPENAI_SSE = (
        b'data: {"id":"x","object":"chat.completion.chunk",'
        b'"choices":[{"delta":{"content":"hello"}}]}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk",'
        b'"choices":[{"delta":{"content":" world"}}],'
        b'"usage":{"prompt_tokens":11,"completion_tokens":2,"total_tokens":13}}\n\n'
        b"data: [DONE]\n\n"
    )

    def test_openai_stream_through_anthropic_parser_is_diagnosed(self):
        chunks: list[str] = []
        consumer, completion = make_streaming_consumer(200, chunks.append)
        consumer.dataReceived(self.OPENAI_SSE)
        consumer.connectionLost(None)

        full_text, _usage, diagnosis = self.successResultOf(completion)
        self.assertEqual(full_text, "")
        self.assertEqual(chunks, [])
        self.assertIn("wire format mismatch", diagnosis)
        self.assertIn("2 SSE events decoded", diagnosis)

    def test_openai_stream_through_openai_parser_works(self):
        chunks: list[str] = []
        consumer, completion = make_streaming_consumer(
            200, chunks.append, parse_openai_event
        )
        consumer.dataReceived(self.OPENAI_SSE)
        consumer.connectionLost(None)

        full_text, usage, diagnosis = self.successResultOf(completion)
        self.assertEqual(chunks, ["hello", " world"])
        self.assertEqual(full_text, "hello world")
        self.assertEqual(usage.get("total_tokens"), 13)
        self.assertEqual(diagnosis, "")

    def test_anthropic_stream_through_openai_parser_is_diagnosed(self):
        """The mismatch is symmetric — neither parser silently succeeds."""
        chunks: list[str] = []
        consumer, completion = make_streaming_consumer(
            200, chunks.append, parse_openai_event
        )
        consumer.dataReceived(
            b'data: {"type":"content_block_delta",'
            b'"delta":{"type":"text_delta","text":"hi"}}\n\n'
        )
        consumer.connectionLost(None)

        full_text, _usage, diagnosis = self.successResultOf(completion)
        self.assertEqual(full_text, "")
        self.assertIn("wire format mismatch", diagnosis)

    def test_a_raising_parser_does_not_kill_the_stream(self):
        def explode(event):
            raise RuntimeError("bad parser")

        chunks: list[str] = []
        consumer, completion = make_streaming_consumer(200, chunks.append, explode)
        consumer.dataReceived(self.OPENAI_SSE)
        consumer.connectionLost(None)

        full_text, _usage, diagnosis = self.successResultOf(completion)
        self.assertEqual(full_text, "")
        self.assertTrue(diagnosis)
        self.flushLoggedErrors()


class TestEventParsers(unittest.TestCase):
    def test_anthropic_parser_shapes(self):
        text, usage = parse_anthropic_event(
            {"type": "content_block_delta", "delta": {"text": "hi"}}
        )
        self.assertEqual(text, "hi")
        self.assertIsNone(usage)

        text, usage = parse_anthropic_event(
            {"type": "message_start", "message": {"usage": {"input_tokens": 5}}}
        )
        self.assertIsNone(text)
        self.assertEqual(usage, {"input_tokens": 5})

        self.assertEqual(parse_anthropic_event({"type": "message_stop"}), (None, None))
        self.assertEqual(parse_anthropic_event({}), (None, None))

    def test_openai_parser_shapes(self):
        text, usage = parse_openai_event({"choices": [{"delta": {"content": "hi"}}]})
        self.assertEqual(text, "hi")
        self.assertIsNone(usage)

        # Final chunk: empty delta, usage present.
        text, usage = parse_openai_event(
            {"choices": [{"delta": {}}], "usage": {"total_tokens": 3}}
        )
        self.assertIsNone(text)
        self.assertEqual(usage, {"total_tokens": 3})

        self.assertEqual(parse_openai_event({}), (None, None))
        self.assertEqual(parse_openai_event({"choices": []}), (None, None))
