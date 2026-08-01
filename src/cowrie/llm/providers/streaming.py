# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Streaming SSE consumer for LLM providers that support
# ABOUTME: incremental responses. The line framing and buffering are
# ABOUTME: vendor-neutral; the per-event shape is supplied by the provider
# ABOUTME: via parse_event, so a backend whose wire format is not
# ABOUTME: Anthropic's can stream without rewriting this file. Codex OAuth
# ABOUTME: buffers SSE in its own _parse_body — this is the opposite
# ABOUTME: direction: fire chunks immediately, not after the stream closes.

from __future__ import annotations

import json
from collections.abc import Callable

from twisted.internet import defer, protocol
from twisted.python import failure as tw_failure
from twisted.python import log

OnChunk = Callable[[str], None]

#: (text_delta_or_None, usage_dict_or_None) for one decoded SSE event.
ParseEvent = Callable[[dict], "tuple[str | None, dict | None]"]


def parse_anthropic_event(event: dict) -> tuple[str | None, dict | None]:
    """Anthropic Messages streaming wire format.

        event: content_block_delta
        data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}

        event: message_delta
        data: {"type":"message_delta","delta":{...},"usage":{...}}

        event: message_stop
        data: {"type":"message_stop"}

    ``message_start`` carries input_tokens up front, so the operator knows
    the bill before any output streams in.
    """
    etype = event.get("type", "")
    if etype == "content_block_delta":
        delta = event.get("delta") or {}
        return (delta.get("text") or None), None
    if etype == "message_delta":
        usage = event.get("usage")
        return None, (usage if isinstance(usage, dict) else None)
    if etype == "message_start":
        usage = (event.get("message") or {}).get("usage")
        return None, (usage if isinstance(usage, dict) else None)
    return None, None


def parse_openai_event(event: dict) -> tuple[str | None, dict | None]:
    """OpenAI chat-completions streaming chunks.

    Shape: ``{"choices":[{"delta":{"content":"hi"}}], "usage": {...}}``.
    Note there is no ``type`` discriminator at all, which is why feeding
    these to the Anthropic parser silently yields nothing.
    """
    choices = event.get("choices") or []
    text = None
    if choices:
        text = (choices[0].get("delta") or {}).get("content") or None
    usage = event.get("usage")
    return text, (usage if isinstance(usage, dict) else None)


class StreamingBodyConsumer(protocol.Protocol):
    """Frame SSE bytes into events; delegate event shape to parse_event.

    Tracks whether *any* event was understood. A stream that decodes as
    JSON but matches no branch of the parser produces empty output with no
    error — the honeypot shows the attacker a blank response and nothing
    is logged. That was the failure mode when a non-Anthropic provider
    opted into streaming, so `matched_events` exists to let the caller
    detect it and fall back rather than serve the blank.
    """

    def __init__(
        self,
        status_code: int,
        on_chunk: OnChunk,
        completion: defer.Deferred,
        parse_event: ParseEvent = parse_anthropic_event,
    ) -> None:
        self.status_code = status_code
        self.on_chunk = on_chunk
        self.completion = completion
        self.parse_event = parse_event
        self._buf = b""
        self._accumulated_text: list[str] = []
        self._usage: dict = {}
        self.decoded_events = 0
        self.matched_events = 0

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
        payload = line[len(b"data:") :].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return
        self.decoded_events += 1
        try:
            text, usage = self.parse_event(event)
        except Exception as e:
            log.err(f"streaming parse_event raised: {e}")
            return
        if text:
            self.matched_events += 1
            self._accumulated_text.append(text)
            try:
                self.on_chunk(text)
            except Exception as e:
                log.err(f"streaming on_chunk callback raised: {e}")
        if usage:
            self.matched_events += 1
            self._usage.update(usage)

    def connectionLost(
        self, reason: tw_failure.Failure = protocol.connectionDone
    ) -> None:
        # End of stream — drain any trailing partial line and fire the
        # completion deferred with the assembled text + usage.
        if self._buf:
            self._process_line(self._buf)
            self._buf = b""
        full_text = "".join(self._accumulated_text)
        self.completion.callback((full_text, self._usage, self._diagnosis()))

    def _diagnosis(self) -> str:
        """'' when healthy, else why the stream produced nothing usable."""
        if self._accumulated_text:
            return ""
        if self.decoded_events == 0:
            return "no SSE events decoded"
        if self.matched_events == 0:
            return (
                f"{self.decoded_events} SSE events decoded but none matched the "
                "provider's parse_event — wire format mismatch"
            )
        return "stream produced no text"


def make_streaming_consumer(
    status_code: int,
    on_chunk: OnChunk,
    parse_event: ParseEvent = parse_anthropic_event,
) -> tuple[StreamingBodyConsumer, defer.Deferred]:
    """Build a consumer plus the deferred that fires
    ``(text, usage, diagnosis)`` on stream completion. ``diagnosis`` is the
    empty string when the stream produced text; otherwise it says why not,
    so the caller can log and fall back instead of serving a blank shell."""
    completion: defer.Deferred = defer.Deferred()
    consumer = StreamingBodyConsumer(status_code, on_chunk, completion, parse_event)
    return consumer, completion
