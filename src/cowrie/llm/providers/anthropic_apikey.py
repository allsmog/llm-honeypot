# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Anthropic Messages API provider authenticated with an
# ABOUTME: ANTHROPIC_API_KEY. Uses ephemeral prompt caching on the
# ABOUTME: system prompt — the persona + world state is mostly stable
# ABOUTME: per session, so caching it is a large latency/cost win.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from twisted.python import log
from twisted.web.http_headers import Headers

from cowrie.llm.providers.base import LLMProvider, LLMRequest
from cowrie.llm.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from configparser import ConfigParser


@ProviderRegistry.register("anthropic_apikey")
class AnthropicAPIKeyProvider(LLMProvider):
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"
    DEFAULT_HOST = "https://api.anthropic.com"
    API_PATH = "/v1/messages"

    def __init__(self, config: ConfigParser) -> None:
        super().__init__(config)
        self._api_key = config.get("llm", "anthropic_api_key", fallback="")
        if not self._api_key:
            # Fallback for users migrating from the upstream single-provider
            # config that used a generic api_key.
            self._api_key = config.get("llm", "api_key", fallback="")
        if not self._api_key:
            log.msg(
                "WARNING: anthropic_apikey provider selected but no "
                "[llm] anthropic_api_key configured"
            )
        self._model = config.get("llm", "model", fallback=self.DEFAULT_MODEL)
        self._host = config.get("llm", "host", fallback=self.DEFAULT_HOST)
        self._cache_system = config.getboolean(
            "llm", "anthropic_cache_system", fallback=True
        )

    @property
    def endpoint(self) -> str:
        return f"{self._host}{self.API_PATH}"

    @property
    def model(self) -> str:
        return self._model

    def _build_headers(self) -> Headers:
        return Headers(
            {
                b"Content-Type": [b"application/json"],
                b"x-api-key": [self._api_key.encode()],
                b"anthropic-version": [b"2023-06-01"],
            }
        )

    def _format_body(self, request: LLMRequest) -> dict[str, Any]:
        if self._cache_system and request.system:
            system: Any = [
                {
                    "type": "text",
                    "text": request.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system = request.system

        messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ] or [{"role": "user", "content": ""}]

        return {
            "model": self._model,
            "system": system,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

    def _parse_response(self, payload: dict[str, Any]) -> str:
        content = payload.get("content") or []
        for block in content:
            if block.get("type") == "text":
                return block.get("text", "")
        return ""
