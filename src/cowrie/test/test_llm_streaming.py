# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cowrie.llm.providers.streaming — the SSE consumer that
fires per-delta callbacks instead of buffering the full body."""

from __future__ import annotations

from twisted.trial import unittest

from cowrie.llm.providers.streaming import make_streaming_consumer


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
        # Completion deferred fires with (full_text, usage_dict).
        full_text, usage = self.successResultOf(completion)
        self.assertEqual(full_text, "hello world")
        # The merged usage contains both the message_start input count
        # and the message_delta output count.
        self.assertEqual(usage.get("input_tokens"), 42)
        self.assertEqual(usage.get("output_tokens"), 7)

    def test_ignores_non_data_lines(self):
        chunks: list[str] = []
        consumer, completion = make_streaming_consumer(200, chunks.append)
        consumer.dataReceived(
            b'event: ping\n\n'
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n'
        )
        consumer.connectionLost(None)
        self.assertEqual(chunks, ["hi"])

    def test_malformed_json_in_data_line_is_skipped(self):
        chunks: list[str] = []
        consumer, completion = make_streaming_consumer(200, chunks.append)
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
        full_text, usage = self.successResultOf(completion)
        self.assertEqual(full_text, "")
        self.assertEqual(usage, {})
