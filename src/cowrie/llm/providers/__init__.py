# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: LLM provider package. Importing this module registers all
# ABOUTME: built-in providers with the ProviderRegistry.

from __future__ import annotations

from cowrie.llm.providers import (
    anthropic_apikey,
    anthropic_oauth,
    codex_apikey,
    codex_oauth,
)
from cowrie.llm.providers.base import LLMMessage, LLMProvider, LLMRequest
from cowrie.llm.providers.registry import ProviderRegistry

__all__ = [
    "LLMMessage",
    "LLMProvider",
    "LLMRequest",
    "ProviderRegistry",
    "anthropic_apikey",
    "anthropic_oauth",
    "codex_apikey",
    "codex_oauth",
]
