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
    # Codex CLI with ChatGPT OAuth only accepts Codex-specific models; the
    # standard OpenAI gpt-4o-* lineup is rejected at the endpoint with a
    # clear error. Current list comes from ~/.codex/models_cache.json and
    # mirrors what `codex --help -c model=...` accepts.
    DEFAULT_MODEL = "gpt-5.4-mini"
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
        # The Codex OAuth endpoint speaks the OpenAI Responses API, not
        # chat-completions: `instructions` for system, `input` for messages,
        # `max_output_tokens` not `max_tokens`. Verified empirically against
        # https://chatgpt.com/backend-api/codex/responses on 2026-05-26.
        # The Codex backend's /responses endpoint is the OpenAI Responses
        # API shape, but with a constrained parameter set: max_output_tokens
        # is rejected ("Unsupported parameter"), and the standard sampling
        # knobs aren't honored either. We pass only what's accepted; honeypot
        # responses are short by nature so token-cap loss is acceptable.
        body: dict[str, Any] = {
            "model": self._model,
            "instructions": request.system or "",
            "input": [
                {"role": m.role, "content": m.content} for m in request.messages
            ] or [{"role": "user", "content": ""}],
            "store": False,
            "stream": True,
        }
        return body

    def _parse_body(self, body: bytes) -> str:
        # Buffered SSE stream. Each event is a pair of lines:
        #   event: response.output_text.delta
        #   data: {"type":"response.output_text.delta","delta":"hello"}
        # followed by a blank line. We accumulate delta strings and also
        # accept the terminal "response.completed" event's `output_text`
        # convenience field as a fallback.
        text_chunks: list[str] = []
        completed_text: str | None = None
        for raw_line in body.splitlines():
            if not raw_line.startswith(b"data:"):
                continue
            payload_str = raw_line[5:].strip()
            if not payload_str or payload_str == b"[DONE]":
                continue
            try:
                event = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            etype = event.get("type", "")
            if etype == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    text_chunks.append(delta)
            elif etype == "response.completed":
                resp = event.get("response") or {}
                if isinstance(resp.get("output_text"), str):
                    completed_text = resp["output_text"]
                else:
                    # response.completed includes the final assembled output
                    # for redundancy; only used as a fallback if no deltas
                    # arrived (rare — usually means short circuit).
                    output = resp.get("output") or []
                    for item in output:
                        if item.get("type") != "message":
                            continue
                        for block in item.get("content") or []:
                            if block.get("type") in ("output_text", "text"):
                                completed_text = block.get("text", "")
                                break
        if text_chunks:
            return "".join(text_chunks)
        return completed_text or ""

    def _parse_response(self, payload: dict[str, Any]) -> str:
        # Responses API shape: {output: [{type: "message", content: [{type: "output_text", text: ...}]}]}
        output = payload.get("output") or []
        for item in output:
            if item.get("type") != "message":
                continue
            for block in item.get("content", []):
                if block.get("type") in ("output_text", "text"):
                    return block.get("text", "")
        # Some Responses API responses include a convenience top-level field.
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        return ""
