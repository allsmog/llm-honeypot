# SPDX-FileCopyrightText: 2025-2026 Michel Oosterhof <michel@oosterhof.net>
#
# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Thin adapter that takes Cowrie's existing prompt format and
# ABOUTME: dispatches to a pluggable LLMProvider selected via config.
# ABOUTME: Keeps the historical LLMClient.get_response(list[str]) shape so
# ABOUTME: cowrie.llm.protocol does not need to change.

from __future__ import annotations

from typing import TYPE_CHECKING

from twisted.internet.defer import Deferred
from twisted.python import log

from cowrie.core.config import CowrieConfig
from cowrie.llm.providers import ProviderRegistry  # noqa: F401  — triggers provider registration
from cowrie.llm.providers.base import LLMMessage, LLMProvider, LLMRequest

if TYPE_CHECKING:
    from collections.abc import Iterable


DEFAULT_PROVIDER = "anthropic_apikey"


def _messages_from_prompt(prompt: Iterable[str]) -> tuple[str, list[LLMMessage]]:
    """Parse Cowrie's flat prompt list into (system, conversation).

    Cowrie's protocol passes ``[system_prompt, "User: cmd1", "System: out1",
    "User: cmd2", ...]``. The "System:" lines are model outputs (assistant
    role), not actual system prompts — naming is from Cowrie's existing
    implementation and we preserve it for compatibility.
    """
    items = list(prompt)
    if not items:
        return "", []
    system = items[0]
    messages: list[LLMMessage] = []
    for raw in items[1:]:
        if raw.startswith("User:"):
            messages.append(LLMMessage(role="user", content=raw[len("User:") :].strip()))
        elif raw.startswith("System:"):
            messages.append(
                LLMMessage(role="assistant", content=raw[len("System:") :].strip())
            )
        else:
            messages.append(LLMMessage(role="user", content=raw))
    return system, messages


class LLMClient:
    """Adapter that the rest of Cowrie talks to.

    Reads ``[llm] provider`` from cowrie.cfg, instantiates the matching
    provider once, and dispatches every command turn through it.
    """

    def __init__(self) -> None:
        provider_name = CowrieConfig.get("llm", "provider", fallback=DEFAULT_PROVIDER)
        try:
            errors = ProviderRegistry.validate(provider_name, CowrieConfig)
        except ValueError as e:
            # Unknown provider name — that's a config typo, not a credential gap.
            log.err(f"LLM provider init failed: {e}")
            raise
        if errors:
            joined = "\n  - " + "\n  - ".join(errors)
            log.err(f"LLM provider {provider_name!r} config validation failed:{joined}")
            raise RuntimeError(
                f"LLM provider {provider_name!r} is misconfigured. "
                f"Errors:{joined}"
            )
        self.provider: LLMProvider = ProviderRegistry.create(
            provider_name, CowrieConfig
        )
        self.max_tokens = CowrieConfig.getint("llm", "max_tokens", fallback=500)
        self.temperature = CowrieConfig.getfloat("llm", "temperature", fallback=0.7)
        log.msg(
            f"LLMClient initialized: provider={provider_name} model={self.provider.model}"
        )

    def generate(self, request: LLMRequest) -> Deferred:
        """Pass an LLMRequest straight through to the provider.

        Used by the interactive protocol when it needs the two-segment
        system_blocks shape for prompt caching. Returns Deferred[str].
        """
        return self.provider.generate(request)

    def generate_streaming(self, request: LLMRequest, on_chunk) -> Deferred:
        """Stream the response, calling on_chunk(text) per delta.

        Returns Deferred[str] of the full accumulated text. The protocol
        layer uses this when [llm] stream = true to make responses drip
        rather than appear in one block — closer to real-shell behavior
        for commands like `tail -f`.

        Falls back to provider.generate() if the provider doesn't
        support streaming.
        """
        if self.provider._supports_streaming():
            return self.provider.generate_streaming(request, on_chunk)
        return self.provider.generate(request)

    def supports_streaming(self) -> bool:
        return self.provider._supports_streaming()

    def get_response(self, prompt: list[str]) -> Deferred:
        """Legacy: build an LLMRequest from Cowrie's prompt list and delegate.

        Kept for the exec-mode path (cowrie/llm/protocol.py's
        HoneyPotExecProtocol), which is one-shot and doesn't benefit
        from the two-segment cache split.

        Returns a Deferred[str]. On any provider-side error the Deferred
        fires with ``""``; the provider has already logged the failure.
        """
        system, messages = _messages_from_prompt(prompt)
        request = LLMRequest(
            system=system,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return self.provider.generate(request)
