# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Provider-agnostic interface and shared HTTP plumbing for LLM backends.
# ABOUTME: Concrete providers (Anthropic API key, Anthropic OAuth, Codex API key,
# ABOUTME: Codex OAuth, ...) subclass LLMProvider and implement generate().

from __future__ import annotations

import json
import os
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from twisted.internet import defer, protocol, reactor
from twisted.internet.defer import Deferred
from twisted.internet.endpoints import HostnameEndpoint
from twisted.python import failure as tw_failure
from twisted.python import log
from twisted.web.client import (
    Agent,
    HTTPConnectionPool,
    ProxyAgent,
    _HTTP11ClientFactory,
)
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from zope.interface import implementer

if TYPE_CHECKING:
    from collections.abc import Callable
    from configparser import ConfigParser


Role = Literal["user", "assistant"]


@dataclass
class LLMMessage:
    role: Role
    content: str


@dataclass
class LLMRequest:
    """Provider-agnostic request shape.

    Providers translate this into their own wire format.

    ``system_blocks`` is an optional structured replacement for
    ``system``: a list of ``(text, cacheable)`` pairs. Anthropic
    providers honor it to split the system prompt across cache
    breakpoints (stable persona = cached; mutable WorldState = not
    cached, cheap to bust). Other providers concatenate the texts and
    fall back to single-block behavior. If ``system_blocks`` is None
    the provider uses the legacy ``system`` field.

    ``usage`` is mutated by the provider after the response lands —
    holds the upstream's token counts, normalized to common keys
    (input_tokens, output_tokens, cached_tokens, total_tokens). The
    caller (LLMClient → protocol) reads this for cost telemetry. We
    use a request-scoped attribute (not a provider attribute) so
    overlapping sessions don't race on shared state.
    """

    system: str = ""
    messages: list[LLMMessage] = field(default_factory=list)
    max_tokens: int = 500
    temperature: float = 0.7
    system_blocks: list[tuple[str, bool]] | None = None
    usage: dict[str, int] = field(default_factory=dict)


def _normalize_anthropic_usage(payload_usage: dict) -> dict[str, int]:
    """Map Anthropic Messages API usage shape to our common keys."""
    if not isinstance(payload_usage, dict):
        return {}
    out: dict[str, int] = {}
    if "input_tokens" in payload_usage:
        out["input_tokens"] = int(payload_usage["input_tokens"])
    if "output_tokens" in payload_usage:
        out["output_tokens"] = int(payload_usage["output_tokens"])
    # Anthropic prompt caching exposes these two separately.
    cached = 0
    if "cache_read_input_tokens" in payload_usage:
        cached += int(payload_usage["cache_read_input_tokens"])
    if "cache_creation_input_tokens" in payload_usage:
        # Tokens written into the cache (counted toward input). Track
        # separately so operators can see cache miss/hit ratio.
        out["cache_creation_tokens"] = int(payload_usage["cache_creation_input_tokens"])
    if cached:
        out["cached_tokens"] = cached
    out["total_tokens"] = out.get("input_tokens", 0) + out.get("output_tokens", 0)
    return out


def _normalize_openai_usage(payload_usage: dict) -> dict[str, int]:
    """Map OpenAI chat-completions / Responses API usage to our keys."""
    if not isinstance(payload_usage, dict):
        return {}
    out: dict[str, int] = {}
    # chat-completions: prompt_tokens / completion_tokens / total_tokens
    if "prompt_tokens" in payload_usage:
        out["input_tokens"] = int(payload_usage["prompt_tokens"])
    if "completion_tokens" in payload_usage:
        out["output_tokens"] = int(payload_usage["completion_tokens"])
    # Responses API: input_tokens / output_tokens
    if "input_tokens" in payload_usage and "input_tokens" not in out:
        out["input_tokens"] = int(payload_usage["input_tokens"])
    if "output_tokens" in payload_usage and "output_tokens" not in out:
        out["output_tokens"] = int(payload_usage["output_tokens"])
    if "total_tokens" in payload_usage:
        out["total_tokens"] = int(payload_usage["total_tokens"])
    else:
        out["total_tokens"] = out.get("input_tokens", 0) + out.get("output_tokens", 0)
    return out


@implementer(IBodyProducer)
class _StringProducer:
    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")
        self.length = len(self.body)

    def startProducing(self, consumer):
        consumer.write(self.body)
        return defer.succeed(None)

    def pauseProducing(self) -> None:
        pass

    def resumeProducing(self) -> None:
        pass

    def stopProducing(self) -> None:
        pass


class _BodyCollector(protocol.Protocol):
    def __init__(self, status_code: int, d: Deferred) -> None:
        self.status_code = status_code
        self.buf = b""
        self.d = d

    def dataReceived(self, data: bytes) -> None:
        self.buf += data

    def connectionLost(self, reason: tw_failure.Failure = protocol.connectionDone) -> None:
        self.d.callback((self.status_code, self.buf))


class _QuietHTTP11ClientFactory(_HTTP11ClientFactory):
    noisy = False


class LLMProvider(ABC):
    """Base class for all LLM providers.

    Subclasses must set the ``name`` class attribute and implement
    :meth:`_build_headers`, :meth:`_format_body`, :meth:`_parse_response`,
    and the ``endpoint`` / ``model`` properties (or override
    :meth:`generate` outright).
    """

    name: str = ""

    def __init__(self, config: ConfigParser) -> None:
        self.config = config
        self.debug = config.getboolean("llm", "debug", fallback=False)
        self.max_tokens = config.getint("llm", "max_tokens", fallback=500)
        self.temperature = config.getfloat("llm", "temperature", fallback=0.7)
        self._pool = HTTPConnectionPool(reactor)
        self._pool._factory = _QuietHTTP11ClientFactory

        proxy_url = (
            os.environ.get("https_proxy")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("http_proxy")
            or os.environ.get("HTTP_PROXY")
        )
        if proxy_url:
            parsed = urllib.parse.urlparse(proxy_url)
            endpoint = HostnameEndpoint(
                reactor, parsed.hostname or "localhost", parsed.port or 8080
            )
            self.agent: Agent | ProxyAgent = ProxyAgent(
                endpoint, reactor, pool=self._pool
            )
            log.msg(f"LLM[{self.name}] using proxy {parsed.hostname}:{parsed.port}")
        else:
            self.agent = Agent(reactor, pool=self._pool)

    # ------------------------------------------------------------------
    # Subclass contract

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """Full URL the request POSTs to."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model name passed in the request body."""

    @abstractmethod
    def _build_headers(self) -> Headers:
        """Build authenticated HTTP headers."""

    @abstractmethod
    def _format_body(self, request: LLMRequest) -> dict[str, Any]:
        """Convert an LLMRequest into this provider's wire format."""

    @abstractmethod
    def _parse_response(self, payload: dict[str, Any]) -> str:
        """Extract the assistant text from the provider's response JSON."""

    # ------------------------------------------------------------------
    # Public entry point

    def generate(self, request: LLMRequest) -> Deferred:
        """Send a request and return a Deferred[str] of the generated text.

        On any error, the Deferred fires with an empty string and a log
        line. Errors are not propagated as failures because the SSH
        session has to keep going regardless — we'd rather show an empty
        prompt than crash the attacker's session and tip them off.
        """
        return self._generate(request, retried=False)

    # Hook: providers that support streaming override _format_streaming_body
    # to set stream:true on the wire and override _supports_streaming to
    # return True. The protocol layer opts in via [llm] stream = true.
    def _supports_streaming(self) -> bool:
        return False

    def generate_streaming(
        self, request: LLMRequest, on_chunk: Callable[[str], None],
    ) -> Deferred:
        """Stream the response, calling on_chunk(text) for each delta.

        Returns Deferred[str] of the full accumulated text — same
        contract as generate() so callers can treat it interchangeably.
        Providers that don't support streaming raise via the deferred;
        the protocol layer falls back to generate() in that case.
        """
        from cowrie.llm.providers.streaming import make_streaming_consumer

        if not self._supports_streaming():
            return defer.fail(RuntimeError(
                f"provider {self.name!r} does not support streaming"
            ))

        body = self._format_streaming_body(request)
        if self.debug:
            log.msg(f"LLM[{self.name}] stream request: {json.dumps(body, indent=2)}")

        d: Deferred = self.agent.request(
            b"POST",
            self.endpoint.encode("utf-8"),
            headers=self._build_headers(),
            bodyProducer=_StringProducer(json.dumps(body)),
        )

        def on_response(resp):
            if resp.code != 200:
                # Non-200: collect the error body the regular way then
                # return empty string so the session keeps going.
                d_body: Deferred = defer.Deferred()
                resp.deliverBody(_BodyCollector(resp.code, d_body))

                def on_body(result):
                    status, body = result
                    log.err(
                        f"LLM[{self.name}] stream HTTP {status}: "
                        f"{body[:300].decode('utf-8', errors='replace')}"
                    )
                    return ""

                d_body.addCallback(on_body)
                return d_body
            consumer, completion = make_streaming_consumer(resp.code, on_chunk)
            resp.deliverBody(consumer)

            def on_stream_done(result):
                text, usage = result
                if isinstance(usage, dict):
                    request.usage.update(_normalize_anthropic_usage(usage))
                return text

            completion.addCallback(on_stream_done)
            return completion

        def on_request_failure(failure):
            log.err(f"LLM[{self.name}] streaming request failed: {failure.getErrorMessage()}")
            return ""

        d.addCallbacks(on_response, on_request_failure)
        return d

    def _format_streaming_body(self, request: LLMRequest) -> dict:
        """Default: same as _format_body. Anthropic providers override
        to also set ``stream: true``."""
        return self._format_body(request)

    def _generate(self, request: LLMRequest, retried: bool) -> Deferred:
        body = self._format_body(request)
        if self.debug:
            log.msg(f"LLM[{self.name}] request: {json.dumps(body, indent=2)}")

        d: Deferred = self.agent.request(
            b"POST",
            self.endpoint.encode("utf-8"),
            headers=self._build_headers(),
            bodyProducer=_StringProducer(json.dumps(body)),
        )
        d.addCallbacks(self._read_body, self._connection_error)
        # Pass request + retried via callback args so _handle_status can
        # decide whether to reload-and-retry on 401 without stashing state
        # on self (which would race across overlapping sessions).
        d.addCallback(self._handle_status, request, retried)
        return d

    def _on_auth_failure(self) -> bool:
        """Hook for providers backed by refreshable credentials.

        Called when the upstream returns HTTP 401. Override to reload the
        credential (e.g. re-read the OAuth token file or keychain entry).
        Return True iff the reload produced a *different* token and a
        retry is worth attempting; False stops the retry chain.
        """
        return False

    @classmethod
    def validate_config(cls, config: ConfigParser) -> list[str]:
        """Return a list of human-readable errors for this provider's config.

        Empty list means the config is structurally valid (i.e. all
        required credential fields are present). Does NOT validate that
        the credential is live / accepted by the upstream — that's a
        network round trip we don't want at startup.

        Override in subclasses.
        """
        return []

    # ------------------------------------------------------------------
    # Internal HTTP helpers

    def _read_body(self, response: IResponse) -> Deferred:
        d: Deferred = defer.Deferred()
        response.deliverBody(_BodyCollector(response.code, d))
        return d

    def _connection_error(self, err: tw_failure.Failure) -> tuple[int, bytes]:
        err.trap(Exception)
        log.err(f"LLM[{self.name}] connection error: {err.getErrorMessage()}")
        return (599, err.getErrorMessage().encode("utf-8"))

    def _handle_status(
        self, result: tuple[int, bytes], request: LLMRequest, retried: bool
    ):
        status, body = result
        if status == 401 and not retried and self._on_auth_failure():
            log.msg(
                eventid="cowrie.llm.token_reloaded",
                provider=self.name,
                format="LLM[%(provider)s] credential reloaded after 401; retrying once",
            )
            return self._generate(request, retried=True)
        if status != 200:
            log.err(
                f"LLM[{self.name}] HTTP {status}: {body.decode('utf-8', errors='replace')}"
            )
            return ""
        try:
            return self._parse_body(body, request)
        except Exception as e:
            log.err(
                f"LLM[{self.name}] body parse failed ({e}): "
                f"{body[:300].decode('utf-8', errors='replace')!r}"
            )
            return ""

    def _parse_body(self, body: bytes, request: LLMRequest) -> str:
        """Convert the raw HTTP response body into assistant text.

        Default: treat as a single JSON document and delegate to
        :meth:`_parse_response`. Providers that speak SSE (server-sent
        events) override this directly.

        ``request`` is passed so subclasses can populate ``request.usage``
        from the response payload — request-scoped attribute, no
        cross-session race.
        """
        payload = json.loads(body)
        if self.debug:
            log.msg(f"LLM[{self.name}] response: {json.dumps(payload, indent=2)}")
        self._capture_usage(payload, request)
        return self._parse_response(payload)

    def _capture_usage(self, payload: dict, request: LLMRequest) -> None:
        """Default: try both Anthropic and OpenAI shapes; first match wins.

        Override in providers that need different shapes (Codex OAuth's
        SSE-delivered usage lives inside the response.completed event).
        """
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return
        # Try Anthropic first (input_tokens/output_tokens with optional
        # cache_*). If that yields no keys, fall back to OpenAI shape.
        norm = _normalize_anthropic_usage(usage)
        if norm.get("total_tokens"):
            request.usage.update(norm)
            return
        norm = _normalize_openai_usage(usage)
        if norm:
            request.usage.update(norm)
