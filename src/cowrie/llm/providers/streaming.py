# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Streaming SSE consumer for LLM providers that support
# ABOUTME: incremental responses (Anthropic Messages with stream:true).
# ABOUTME: Fires a callback per text delta so the protocol can write
# ABOUTME: chunks to the attacker's terminal as they arrive, instead
# ABOUTME: of buffering the whole response. Codex OAuth already buffers
# ABOUTME: SSE in its _parse_body — this is the opposite direction:
# ABOUTME: fire chunks immediately, not after the stream closes.

from __future__ import annotations

import json
from typing import Callable, Optional

from twisted.internet import defer, protocol
from twisted.python import failure as tw_failure
from twisted.python import log


OnChunk = Callable[[str], None]


class StreamingBodyConsumer(protocol.Protocol):
    """Consume Anthropic SSE response bytes, fire on_chunk(text) per delta.

    The Anthropic Messages streaming wire format:
        event: content_block_delta
        data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

        event: message_delta
        data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{...}}

        event: message_stop
        data: {"type":"message_stop"}

    We parse line-by-line, fire on_chunk for each text_delta, and
    callback the deferred with (full_text, usage_dict) when the
    stream completes.
    """

    def __init__(
        self,
        status_code: int,
        on_chunk: OnChunk,
        completion: defer.Deferred,
    ) -> None:
        self.status_code = status_code
        self.on_chunk = on_chunk
        self.completion = completion
        self._buf = b""
        self._accumulated_text: list[str] = []
        self._usage: dict = {}

    def dataReceived(self, data: bytes) -> None:
        self._buf += data
        # SSE events are separated by blank lines. Process complete
        # events; keep the trailing partial line in the buffer.
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._process_line(line)

    def _process_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[len(b"data:"):].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return
        etype = event.get("type", "")
        if etype == "content_block_delta":
            delta = event.get("delta") or {}
            text = delta.get("text") or ""
            if text:
                self._accumulated_text.append(text)
                try:
                    self.on_chunk(text)
                except Exception as e:
                    log.err(f"streaming on_chunk callback raised: {e}")
        elif etype == "message_delta":
            usage = event.get("usage") or {}
            if isinstance(usage, dict):
                self._usage.update(usage)
        elif etype == "message_start":
            # Initial event carries an empty message; the usage block
            # here records the input_tokens (Anthropic provides them
            # up front so the operator knows the bill before output
            # streams in).
            msg = event.get("message") or {}
            usage = msg.get("usage") or {}
            if isinstance(usage, dict):
                self._usage.update(usage)

    def connectionLost(self, reason: tw_failure.Failure = protocol.connectionDone) -> None:
        # End of stream — drain any trailing partial line and fire the
        # completion deferred with the assembled text + usage.
        if self._buf:
            self._process_line(self._buf)
            self._buf = b""
        full_text = "".join(self._accumulated_text)
        self.completion.callback((full_text, self._usage))


def make_streaming_consumer(
    status_code: int,
    on_chunk: OnChunk,
) -> tuple[StreamingBodyConsumer, defer.Deferred]:
    """Build a consumer + the deferred that fires (text, usage) on
    stream completion. Used by base.LLMProvider's streaming path."""
    completion: defer.Deferred = defer.Deferred()
    consumer = StreamingBodyConsumer(status_code, on_chunk, completion)
    return consumer, completion
