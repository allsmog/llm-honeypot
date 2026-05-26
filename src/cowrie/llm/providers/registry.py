# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Provider registry — lookup by name from config, instantiation,
# ABOUTME: and listing of available providers. New providers register via
# ABOUTME: the @ProviderRegistry.register decorator.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configparser import ConfigParser

    from cowrie.llm.providers.base import LLMProvider


class ProviderRegistry:
    _registry: dict[str, type[LLMProvider]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(provider_cls: type[LLMProvider]):
            provider_cls.name = name
            cls._registry[name] = provider_cls
            return provider_cls

        return decorator

    @classmethod
    def create(cls, name: str, config: ConfigParser) -> LLMProvider:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<none registered>"
            raise ValueError(
                f"Unknown LLM provider {name!r}. Available providers: {available}"
            )
        return cls._registry[name](config)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)
