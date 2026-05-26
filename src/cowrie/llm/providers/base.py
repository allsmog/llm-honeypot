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
    """

    system: str
    messages: list[LLMMessage] = field(default_factory=list)
    max_tokens: int = 500
    temperature: float = 0.7


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
        d.addCallback(self._handle_status)
        return d

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

    def _handle_status(self, result: tuple[int, bytes]) -> str:
        status, body = result
        if status != 200:
            log.err(
                f"LLM[{self.name}] HTTP {status}: {body.decode('utf-8', errors='replace')}"
            )
            return ""
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            log.err(f"LLM[{self.name}] invalid JSON: {e}")
            return ""
        if self.debug:
            log.msg(f"LLM[{self.name}] response: {json.dumps(payload, indent=2)}")
        try:
            return self._parse_response(payload)
        except (KeyError, IndexError, TypeError) as e:
            log.err(f"LLM[{self.name}] response parse failed ({e}): {payload!r}")
            return ""
