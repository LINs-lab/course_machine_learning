# Ported (semantics preserved) from project_collab_auto_research/graph/llm.py:
# the LLMProvider Protocol is verbatim; load_provider() keeps the lazy,
# config-driven selection shape with the keyless provider as the default.
"""Provider abstraction for the generation backend (keyless by default).

``config.yaml`` selects the provider:

    llm.provider ∈ {echo (default), openai_compat}

- ``echo``: deterministic, keyless provider — a pure function of the prompt
  (see rsi.llm.echo). The demo runs entirely on it.
- ``openai_compat``: any OpenAI-compatible endpoint. Optional import; the key
  is read from an env var by the provider itself, never from config.

TODO(homework): wire a real provider of your choice behind this Protocol and
register it in PROVIDERS (see docs/homework.md).
"""
from __future__ import annotations

from typing import Protocol

from rsi import config


class LLMConfigError(RuntimeError):
    pass


class LLMProvider(Protocol):
    def compose(self, prompt: str, max_tokens: int) -> str: ...
    def analyze(self, prompt: str, max_tokens: int) -> str: ...
    def rerank(self, query: str, candidates: list, max_tokens: int = 4096) -> list: ...


class NullProvider:
    """Instant no-op provider: empty compose/analyze, passthrough rerank."""

    def compose(self, prompt: str, max_tokens: int) -> str:
        return ""

    def analyze(self, prompt: str, max_tokens: int) -> str:
        return ""

    def rerank(self, query: str, candidates: list, max_tokens: int = 4096) -> list:
        return candidates


def load_provider(cfg: dict | None = None) -> LLMProvider:
    """Build the provider named by cfg['llm.provider'] / config['llm.provider']."""
    settings = cfg if cfg is not None else {}
    name = settings.get("llm.provider") or config.get("llm.provider")
    if name == "echo":
        from rsi.llm.echo import EchoProvider
        return EchoProvider()
    if name == "null":
        return NullProvider()
    if name == "openai_compat":
        from rsi.llm.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(settings=settings)
    raise LLMConfigError(
        f"unknown llm.provider {name!r}; use 'echo', 'null', or "
        f"'openai_compat'")
