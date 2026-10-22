"""OpenAI-compatible provider (optional; never on the default wiring path).

Skeleton with the security posture of the source system: the API key is read
from an env var (RSI_OPENAI_API_KEY) at call time, never stored in config, and
the package never requires this module unless the user selects it.

TODO(homework): complete request/retry/timeout handling and register the
provider under a profile of your choosing (docs/homework.md, task L-1).
"""
from __future__ import annotations

import json
import os
import urllib.request

from rsi.llm.base import LLMConfigError

_TIMEOUT_S = 60


class OpenAICompatProvider:
    def __init__(self, settings: dict | None = None):
        s = settings or {}
        self.base_url = (s.get("llm.openai_compat.base_url")
                         or os.environ.get("RSI_OPENAI_BASE_URL", ""))
        self.model = (s.get("llm.openai_compat.model")
                      or os.environ.get("RSI_OPENAI_MODEL", ""))
        # optional extra request-body fields (e.g. vLLM/Qwen3's
        # {"chat_template_kwargs": {"enable_thinking": false}})
        self.extra_body = dict(s.get("llm.openai_compat.extra_body") or {})
        if not self.base_url or not self.model:
            raise LLMConfigError(
                "openai_compat needs llm.openai_compat.base_url and "
                ".model (or RSI_OPENAI_BASE_URL / RSI_OPENAI_MODEL)")

    def _key(self) -> str:
        key = os.environ.get("RSI_OPENAI_API_KEY", "")
        if not key:
            raise LLMConfigError(
                "RSI_OPENAI_API_KEY is not set; refusing to ship an empty key")
        return key

    def _chat(self, messages: list[dict], max_tokens: int) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            **self.extra_body,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._key()}"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode())
        return payload["choices"][0]["message"]["content"]

    def compose(self, prompt: str, max_tokens: int) -> str:
        return self._chat([{"role": "user", "content": prompt}], max_tokens)

    def analyze(self, prompt: str, max_tokens: int) -> str:
        return self._chat([{"role": "user", "content": prompt}], max_tokens)

    def rerank(self, query: str, candidates: list, max_tokens: int = 4096) -> list:
        raise NotImplementedError("TODO(homework): rerank via the endpoint")
