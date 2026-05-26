# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Anthropic Messages API provider authenticated with a Claude Code
# ABOUTME: OAuth bearer token. On macOS the token lives in Keychain under
# ABOUTME: service "Claude Code-credentials"; on Linux it's in a JSON file.
# ABOUTME: Refresh is the caller's responsibility — re-auth in Claude Code
# ABOUTME: when the token expires.

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from twisted.python import log
from twisted.web.http_headers import Headers

from cowrie.llm.providers.anthropic_apikey import _build_anthropic_system
from cowrie.llm.providers.base import LLMProvider, LLMRequest
from cowrie.llm.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from configparser import ConfigParser


MACOS_KEYCHAIN_SERVICE = "Claude Code-credentials"
LINUX_DEFAULT_PATH = "~/.config/claude-code/credentials.json"


@ProviderRegistry.register("anthropic_oauth")
class AnthropicOAuthProvider(LLMProvider):
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"
    DEFAULT_HOST = "https://api.anthropic.com"
    API_PATH = "/v1/messages"
    # Header required by Anthropic when authenticating with an OAuth bearer
    # token instead of an API key. Update this if Anthropic bumps the beta.
    OAUTH_BETA = b"oauth-2025-04-20"

    def __init__(self, config: ConfigParser) -> None:
        super().__init__(config)
        # Config overrides: explicit file path wins over the platform default
        # (useful on macOS too if the user exported the keychain entry to a
        # file, or has multiple Claude Code accounts).
        self._token_file = config.get(
            "llm", "anthropic_oauth_token_file", fallback=""
        )
        self._keychain_service = config.get(
            "llm", "anthropic_oauth_keychain_service", fallback=MACOS_KEYCHAIN_SERVICE
        )
        self._token, self._expires_at = self._load_token()

        self._model = config.get("llm", "model", fallback=self.DEFAULT_MODEL)
        self._host = config.get("llm", "host", fallback=self.DEFAULT_HOST)
        self._cache_system = config.getboolean(
            "llm", "anthropic_cache_system", fallback=True
        )

        if self._expires_at and self._expires_at < int(time.time() * 1000):
            log.msg(
                "WARNING: Claude Code OAuth token appears expired. "
                "Re-authenticate in Claude Code (claude auth login) and restart."
            )

    # ------------------------------------------------------------------
    # Token loading — macOS Keychain, then file, with explicit file
    # overriding the platform default.

    def _load_token(self) -> tuple[str, int]:
        if self._token_file:
            return self._load_from_file(os.path.expanduser(self._token_file))
        if sys.platform == "darwin":
            tok, exp = self._load_from_macos_keychain()
            if tok:
                return tok, exp
            log.msg(
                "anthropic_oauth: Keychain entry not found, falling back to file path"
            )
        return self._load_from_file(os.path.expanduser(LINUX_DEFAULT_PATH))

    def _load_from_macos_keychain(self) -> tuple[str, int]:
        try:
            result = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s",
                    self._keychain_service,
                    "-w",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            log.err(f"anthropic_oauth: Keychain read failed: {e}")
            return "", 0
        if result.returncode != 0:
            return "", 0
        return self._parse_payload(result.stdout, source="keychain")

    def _load_from_file(self, path: str) -> tuple[str, int]:
        p = Path(path)
        if not p.is_file():
            log.msg(
                f"WARNING: anthropic_oauth credentials not found at {path}. "
                "On macOS the token comes from Keychain (service "
                f"{MACOS_KEYCHAIN_SERVICE!r}); on Linux dump the JSON to that path."
            )
            return "", 0
        try:
            return self._parse_payload(p.read_text(), source=path)
        except OSError as e:
            log.err(f"anthropic_oauth: cannot read {path}: {e}")
            return "", 0

    def _parse_payload(self, payload: str, *, source: str) -> tuple[str, int]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            log.err(f"anthropic_oauth: JSON parse failed ({source}): {e}")
            return "", 0
        # Claude Code keychain schema: {"claudeAiOauth": {"accessToken": ..., "expiresAt": ...}}
        oauth = data.get("claudeAiOauth")
        if isinstance(oauth, dict):
            token = oauth.get("accessToken") or ""
            expires = int(oauth.get("expiresAt") or 0)
            if token:
                return token, expires
        # Legacy / manual schema: {"access_token": "..."}
        token = data.get("access_token") or data.get("token") or ""
        if not token:
            log.err(
                f"anthropic_oauth: no access token in {source}. "
                "Expected claudeAiOauth.accessToken or access_token."
            )
        return token, 0

    # ------------------------------------------------------------------
    # Provider interface

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
                b"Authorization": [f"Bearer {self._token}".encode()],
                b"anthropic-version": [b"2023-06-01"],
                b"anthropic-beta": [self.OAUTH_BETA],
            }
        )

    def _format_body(self, request: LLMRequest) -> dict[str, Any]:
        system = _build_anthropic_system(request, cache_default=self._cache_system)
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

    @classmethod
    def validate_config(cls, config) -> list[str]:
        # Either an explicit token file exists, or on macOS the keychain
        # entry exists. We can't check keychain at module-import time on
        # non-darwin platforms, so accept "no file + non-darwin" as a
        # config error.
        token_file = os.path.expanduser(
            config.get("llm", "anthropic_oauth_token_file", fallback="")
        )
        if token_file and Path(token_file).is_file():
            return []
        if sys.platform == "darwin":
            service = config.get(
                "llm", "anthropic_oauth_keychain_service",
                fallback=MACOS_KEYCHAIN_SERVICE,
            )
            try:
                result = subprocess.run(
                    ["security", "find-generic-password", "-s", service],
                    capture_output=True, text=True, timeout=3, check=False,
                )
                if result.returncode == 0:
                    return []
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            return [
                f"anthropic_oauth: no token at [llm] anthropic_oauth_token_file "
                f"and no keychain entry for service {service!r}. "
                "Run `claude auth login` or set anthropic_oauth_token_file."
            ]
        default_path = os.path.expanduser(LINUX_DEFAULT_PATH)
        if Path(default_path).is_file():
            return []
        return [
            f"anthropic_oauth: no token at {default_path} and no override "
            "in [llm] anthropic_oauth_token_file. Run `claude auth login`."
        ]

    def _on_auth_failure(self) -> bool:
        # Re-read the source-of-truth (keychain on macOS, file otherwise).
        # If Claude Code has refreshed the token since we loaded it, the
        # new value will differ — retry once with the fresh token. If it
        # hasn't changed, the 401 reflects a real auth problem (revoked,
        # expired without refresh, user logged out) and retrying just
        # burns a round-trip.
        old = self._token
        new, expires = self._load_token()
        if new and new != old:
            self._token = new
            self._expires_at = expires
            return True
        return False
