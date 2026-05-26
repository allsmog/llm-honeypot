# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Codex CLI OAuth provider — consumes a bearer token previously
# ABOUTME: obtained via the Codex CLI OAuth flow (ChatGPT Plus/Pro session).
# ABOUTME: The OAuth dance itself is out of scope here; we just read the
# ABOUTME: token from disk and use it.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from twisted.python import log
from twisted.web.http_headers import Headers

from cowrie.llm.providers.base import LLMProvider, LLMRequest
from cowrie.llm.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from configparser import ConfigParser


@ProviderRegistry.register("codex_oauth")
class CodexOAuthProvider(LLMProvider):
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_HOST = "https://chatgpt.com"
    DEFAULT_PATH = "/backend-api/codex/responses"

    def __init__(self, config: ConfigParser) -> None:
        super().__init__(config)
        self._token_file = os.path.expanduser(
            config.get(
                "llm",
                "codex_oauth_token_file",
                fallback="~/.codex/auth.json",
            )
        )
        self._token = self._load_token()
        self._model = config.get("llm", "model", fallback=self.DEFAULT_MODEL)
        self._host = config.get("llm", "host", fallback=self.DEFAULT_HOST)
        self._path = config.get("llm", "path", fallback=self.DEFAULT_PATH)

    def _load_token(self) -> str:
        path = Path(self._token_file)
        if not path.is_file():
            log.msg(
                f"WARNING: codex_oauth token file not found at {self._token_file}. "
                "Run the Codex CLI auth flow first, then point "
                "[llm] codex_oauth_token_file at the credentials JSON."
            )
            return ""
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            log.err(f"codex_oauth token file {self._token_file} is not JSON: {e}")
            return ""
        # Codex CLI nests the access token under "tokens" in current versions;
        # older versions stored it flat. Accept either shape.
        tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else data
        token = tokens.get("access_token") or tokens.get("token") or ""
        if not token:
            log.err(
                f"codex_oauth token file {self._token_file} has no access_token field"
            )
        return token

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
                b"Authorization": [f"Bearer {self._token}".encode()],
            }
        )

    def _format_body(self, request: LLMRequest) -> dict[str, Any]:
        # Codex CLI uses the OpenAI chat-completions message shape, just at
        # a different endpoint. Keep this in lockstep with CodexAPIKey so a
        # user can switch auth modes without re-prompting.
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)
        return {
            "model": self._model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

    def _parse_response(self, payload: dict[str, Any]) -> str:
        # Try OpenAI chat-completions shape first.
        choices = payload.get("choices") or []
        if choices:
            return choices[0].get("message", {}).get("content", "")
        # Codex responses API shape: {output: [{content: [{text: "..."}]}]}
        output = payload.get("output") or []
        for item in output:
            for block in item.get("content", []):
                if block.get("type") in ("output_text", "text"):
                    return block.get("text", "")
        return ""
