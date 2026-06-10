# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: OpenAI/Codex provider authenticated with an OPENAI_API_KEY.
# ABOUTME: Hits the standard /v1/chat/completions endpoint — works with
# ABOUTME: any OpenAI-compatible API (Azure, Together, vLLM, etc.) by
# ABOUTME: overriding the host/path config keys.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from twisted.python import log
from twisted.web.http_headers import Headers

from cowrie.llm.providers.base import LLMProvider, LLMRequest
from cowrie.llm.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from configparser import ConfigParser


@ProviderRegistry.register("codex_apikey")
class CodexAPIKeyProvider(LLMProvider):
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_HOST = "https://api.openai.com"
    DEFAULT_PATH = "/v1/chat/completions"

    def __init__(self, config: ConfigParser) -> None:
        super().__init__(config)
        self._api_key = config.get("llm", "openai_api_key", fallback="")
        if not self._api_key:
            self._api_key = config.get("llm", "api_key", fallback="")
        if not self._api_key:
            log.msg(
                "WARNING: codex_apikey provider selected but no "
                "[llm] openai_api_key configured"
            )
        self._model = config.get("llm", "model", fallback=self.DEFAULT_MODEL)
        self._host = config.get("llm", "host", fallback=self.DEFAULT_HOST)
        self._path = config.get("llm", "path", fallback=self.DEFAULT_PATH)

    @property
    def endpoint(self) -> str:
        return f"{self._host}{self._path}"

    @property
    def model(self) -> str:
        return self._model

    def _build_headers(self) -> Headers:
        return Headers(
            {
                b"Content-Type": [b"application/json"],
                b"Authorization": [f"Bearer {self._api_key}".encode()],
            }
        )

    def _format_body(self, request: LLMRequest) -> dict[str, Any]:
        # OpenAI chat-completions doesn't support per-block cache
        # breakpoints (its automatic prompt caching kicks in at >=1024
        # tokens regardless), so concatenate any system_blocks back
        # into a single system message.
        system_text = request.system
        if request.system_blocks:
            system_text = "\n\n".join(t for t, _ in request.system_blocks if t)
        messages: list[dict[str, str]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)
        return {
            "model": self._model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

    def _parse_response(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", ""))

    @classmethod
    def validate_config(cls, config) -> list[str]:
        key = config.get("llm", "openai_api_key", fallback="") or config.get(
            "llm", "api_key", fallback=""
        )
        if not key:
            return [
                "codex_apikey: missing [llm] openai_api_key "
                "(get one at https://platform.openai.com/api-keys)"
            ]
        return []
