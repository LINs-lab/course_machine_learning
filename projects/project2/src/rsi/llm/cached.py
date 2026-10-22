"""Content-addressed cache wrapper for any LLMProvider (resume + audit).

Every compose/analyze call is keyed by the sha256 of the full request
(provider identity + model + method + arguments + extra request-body
fields). On a hit the stored response is replayed WITHOUT a network call,
so an interrupted session resumes through the LLM-driven parts of its
trajectory deterministically; on a miss the inner provider is called and
the exchange is stored before it is returned (a crash can never lose a
response the engine has already seen).

The cache DB doubles as the complete agent-context log: every row holds
the full prompt and the full raw response, with wall-clock metadata. Keep
it under the session dir so the session archive is self-contained.

Replay alone does not make a FRESH call reproducible — pair this wrapper
with a deterministic request (temperature 0 + an explicit seed, both set
via the provider's extra_body) and the same endpoint/model.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    request TEXT NOT NULL,
    response TEXT NOT NULL,
    latency_s REAL NOT NULL,
    created TEXT NOT NULL
);
"""


class CachedProvider:
    """Wrap ``inner`` with a sqlite call cache at ``cache_path``."""

    def __init__(self, inner, cache_path: str | Path, identity: dict | None = None):
        self.inner = inner
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # anything that changes the response distribution must change the
        # key: endpoint, model, and the extra request body (temperature,
        # seed, chat-template kwargs) all qualify
        self.identity = identity or {
            "base_url": getattr(inner, "base_url", ""),
            "model": getattr(inner, "model", ""),
            "extra_body": getattr(inner, "extra_body", {}),
        }
        self.hits = 0
        self.misses = 0
        db = sqlite3.connect(self.cache_path)
        try:
            db.executescript(_SCHEMA)
        finally:
            db.close()

    def _key(self, method: str, payload: dict) -> str:
        blob = json.dumps(
            {"identity": self.identity, "method": method, "payload": payload},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _call(self, method: str, payload: dict, fn) -> tuple[str, bool]:
        key = self._key(method, payload)
        db = sqlite3.connect(self.cache_path)
        try:
            row = db.execute(
                "SELECT response FROM calls WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                self.hits += 1
                return row[0], True
            t0 = time.monotonic()
            response = fn()
            latency = time.monotonic() - t0
            db.execute(
                "INSERT INTO calls (key, method, request, response,"
                " latency_s, created) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    method,
                    json.dumps(payload, ensure_ascii=False),
                    response,
                    latency,
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
            db.commit()
            self.misses += 1
            return response, False
        finally:
            db.close()

    def compose(self, prompt: str, max_tokens: int) -> str:
        text, _ = self._call(
            "compose",
            {"prompt": prompt, "max_tokens": max_tokens},
            lambda: self.inner.compose(prompt, max_tokens),
        )
        return text

    def analyze(self, prompt: str, max_tokens: int) -> str:
        text, _ = self._call(
            "analyze",
            {"prompt": prompt, "max_tokens": max_tokens},
            lambda: self.inner.analyze(prompt, max_tokens),
        )
        return text

    def compose_cached(self, prompt: str, max_tokens: int) -> tuple[str, bool]:
        """compose + a cache-hit flag (for call-level logging)."""
        return self._call(
            "compose",
            {"prompt": prompt, "max_tokens": max_tokens},
            lambda: self.inner.compose(prompt, max_tokens),
        )

    def rerank(self, query: str, candidates: list, max_tokens: int = 4096) -> list:
        # rerank returns structured data; delegate uncached (the reference
        # providers do not implement it anyway)
        return self.inner.rerank(query, candidates, max_tokens)
