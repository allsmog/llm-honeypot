# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Provider that reaches any backend LangChain speaks to — local
# ABOUTME: Ollama, hosted OpenAI, Gemini, Bedrock, vLLM — through one
# ABOUTME: config line, instead of one hand-written adapter per vendor.
# ABOUTME: LangChain is sync and Cowrie is a Twisted reactor, so calls are
# ABOUTME: bridged with deferToThread; langchain is an optional install so
# ABOUTME: the honeypot's dependency footprint stays small by default.

"""LangChain-backed provider.

Why this is a provider rather than a replacement for the abstraction —
three constraints, each checkable rather than a matter of taste:

* **Dependency weight.** The honeypot's runtime deps are eleven lean
  pinned packages. LangChain brings a large transitive tree onto a machine
  that is deliberately exposed to the internet, so it must be opt-in.
* **Interpreter support.** CI covers CPython 3.10-3.15 including
  free-threaded builds and PyPy. LangChain does not support all of those,
  so it cannot be a hard dependency without breaking the matrix.
* **Concurrency model.** Every LLM path here returns a Twisted
  ``Deferred``; LangChain's chat models are sync (with an asyncio variant).
  The bridge belongs in one file rather than smeared through the codebase.

The tradeoff, stated plainly: ``deferToThread`` consumes one worker from
Twisted's default pool of ten per in-flight call. The native HTTP providers
are fully async and have no such ceiling. Under heavy concurrent load this
path queues where they would not — measure before pointing a busy sensor at
it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.threads import deferToThread
from twisted.python import log
from twisted.web.http_headers import Headers

from cowrie.llm.providers.base import LLMProvider, LLMRequest
from cowrie.llm.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from configparser import ConfigParser

#: pip extra that supplies the import.
_EXTRA = "llm-honeypot[langchain]"


def _import_messages():
    """Message classes only — these live in the lighter langchain-core."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    return SystemMessage, HumanMessage, AIMessage


def _import_init_chat_model():
    """Model factory, which lives in the full langchain package.

    Split from _import_messages so a test that injects its own chat model
    needs only langchain-core, and so a missing integration package
    produces a precise error rather than one blaming the whole install.
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model


def langchain_available() -> bool:
    """True when both imports resolve.

    Deliberately lazy everywhere: providers/__init__ imports every provider
    at startup to register it, so a hard top-level import would make
    langchain mandatory for everyone.
    """
    try:
        _import_messages()
        _import_init_chat_model()
    except Exception:
        return False
    return True


@ProviderRegistry.register("langchain")
class LangChainProvider(LLMProvider):
    """Any backend LangChain supports, selected by ``[llm] langchain_model``.

    Model strings are LangChain's ``provider:model`` form::

        langchain_model = ollama:llama3.1
        langchain_model = openai:gpt-4o-mini
        langchain_model = google_genai:gemini-2.0-flash

    Overrides ``generate`` outright: the base class's endpoint/headers/body
    machinery is for direct HTTP, and LangChain owns the wire format. The
    five abstract members are satisfied so the class is concrete, but only
    ``model`` carries meaning.
    """

    DEFAULT_MODEL = ""

    def __init__(self, config: ConfigParser) -> None:
        super().__init__(config)
        self._model = config.get("llm", "langchain_model", fallback="") or config.get(
            "llm", "model", fallback=""
        )
        self._chat: Any | None = None

    # -- abstract surface -------------------------------------------------
    # Satisfied for concreteness; LangChain owns transport, so there is no
    # endpoint or header for us to build.

    @property
    def endpoint(self) -> str:
        return f"langchain://{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _build_headers(self) -> Headers:
        return Headers({})

    def _format_body(self, request: LLMRequest) -> dict[str, Any]:
        """Not a wire body — a debug-visible view of what we send.

        The conformance suite reads this to check the system prompt is not
        being dropped, so it must reflect the real message list built in
        _to_messages.
        """
        return {
            "model": self._model,
            "system": request.system_text(),
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

    def _parse_response(self, payload: dict[str, Any]) -> str:
        return str(payload.get("content", ""))

    # -- the actual path --------------------------------------------------

    def _chat_model(self):
        """Build the chat model once, lazily.

        Deferred until first use so a misconfigured model string surfaces
        as an empty response plus a log line rather than taking down the
        listener at startup — matching how every other provider behaves
        when its upstream is unreachable.
        """
        if self._chat is None:
            init_chat_model = _import_init_chat_model()
            self._chat = init_chat_model(
                self._model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        return self._chat

    def _to_messages(self, request: LLMRequest) -> list:
        SystemMessage, HumanMessage, AIMessage = _import_messages()
        messages: list = []
        # system_text() rather than request.system: the interactive
        # protocol only ever fills system_blocks, so reading `system`
        # directly would send no persona at all.
        system = request.system_text()
        if system:
            messages.append(SystemMessage(content=system))
        for m in request.messages:
            cls = HumanMessage if m.role == "user" else AIMessage
            messages.append(cls(content=m.content))
        return messages

    def _normalize_usage(self, usage: dict) -> dict[str, int]:
        """LangChain normalizes across backends into usage_metadata.

        Shape: ``{"input_tokens", "output_tokens", "total_tokens",
        "input_token_details": {"cache_read": N}}``. Mapping it here is
        what lets a per-session token cap work against any backend rather
        than only the two whose raw shapes we happen to recognize.
        """
        out: dict[str, int] = {}
        for src, dst in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            if src in usage:
                out[dst] = int(usage[src] or 0)
        details = usage.get("input_token_details")
        if isinstance(details, dict) and details.get("cache_read"):
            out["cached_tokens"] = int(details["cache_read"])
        if out and "total_tokens" not in out:
            out["total_tokens"] = out.get("input_tokens", 0) + out.get("output_tokens", 0)
        if out:
            return out
        # Not LangChain's shape after all — let the base class try the raw
        # vendor shapes it knows.
        return super()._normalize_usage(usage)

    def _capture(self, message: Any, request: LLMRequest) -> None:
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict) and usage:
            request.usage.update(self._normalize_usage(usage))

    @staticmethod
    def _text_of(message: Any) -> str:
        """Flatten LangChain content, which may be a str or a block list."""
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return str(content or "")

    def _invoke(self, request: LLMRequest) -> str:
        message = self._chat_model().invoke(self._to_messages(request))
        self._capture(message, request)
        return self._text_of(message)

    def _to_empty(self, failure) -> str:
        """Provider trouble yields "" and a log line, never a failure.

        Same contract as the HTTP providers: the SSH session has to keep
        going regardless, and an abrupt disconnect is a louder tell than a
        command that produced no output.
        """
        log.msg(
            eventid="cowrie.llm.error",
            provider=self.name,
            error=failure.getErrorMessage()[:300],
            format="LLM[%(provider)s] %(error)s",
        )
        return ""

    def generate(self, request: LLMRequest) -> Deferred:
        d = deferToThread(self._invoke, request)
        d.addErrback(self._to_empty)
        return d

    def _supports_streaming(self) -> bool:
        return True

    def _stream(self, request: LLMRequest, on_chunk) -> str:
        """Runs on a worker thread — every reactor touch goes through
        callFromThread."""
        parts: list[str] = []
        last = None
        for chunk in self._chat_model().stream(self._to_messages(request)):
            last = chunk
            text = self._text_of(chunk)
            if text:
                parts.append(text)
                # The only safe way to touch the reactor from a worker
                # thread. on_chunk writes to the attacker's terminal, which
                # must happen on the reactor thread.
                reactor.callFromThread(on_chunk, text)  # type: ignore[attr-defined]
        if last is not None:
            self._capture(last, request)
        return "".join(parts)

    def generate_streaming(self, request: LLMRequest, on_chunk) -> Deferred:
        d = deferToThread(self._stream, request, on_chunk)
        d.addErrback(self._to_empty)
        return d

    @classmethod
    def validate_config(cls, config: ConfigParser) -> list[str]:
        errors: list[str] = []
        model = config.get("llm", "langchain_model", fallback="") or config.get(
            "llm", "model", fallback=""
        )
        if not model:
            errors.append(
                "langchain: missing [llm] langchain_model — use LangChain's "
                "'provider:model' form, e.g. 'ollama:llama3.1' or "
                "'openai:gpt-4o-mini'"
            )
        if not langchain_available():
            # Caught at startup rather than as an ImportError traceback on
            # the attacker's first command.
            errors.append(
                f"langchain: the langchain package is not installed — "
                f"`pip install '{_EXTRA}'`, plus the integration package for "
                f"your backend (e.g. langchain-ollama, langchain-openai)"
            )
        return errors
