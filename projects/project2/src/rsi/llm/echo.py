"""Deterministic, keyless LLM provider — a pure function of the prompt.

The demo and the whole test suite run on this provider: no network, no clock,
no randomness, so identical prompts always produce identical completions.

Selection rule: if the prompt embeds a numbered ``<options>`` block, the
provider answers with the option whose index is derived from a hash of the
prompt (stable across runs and machines). Otherwise it echoes a compact
digest of the prompt. This is intentionally NOT a good proposer — it exists
so the harness is observable end-to-end without credentials.
"""
from __future__ import annotations

import hashlib
import re

_OPTION_RE = re.compile(r"<options>(.*?)</options>", re.DOTALL)
_ITEM_RE = re.compile(r"^\s*(\d+)[.)]\s*(.+)$", re.MULTILINE)


class EchoProvider:
    def compose(self, prompt: str, max_tokens: int) -> str:
        return self._answer(prompt)

    def analyze(self, prompt: str, max_tokens: int) -> str:
        return self._answer(prompt)

    def rerank(self, query: str, candidates: list, max_tokens: int = 4096) -> list:
        # Deterministic stable reordering keyed on the query (never random).
        if not candidates:
            return candidates
        rank = lambda item: hashlib.blake2b(  # noqa: E731
            f"{query}|{item}".encode(), digest_size=8).digest()
        return sorted(candidates, key=rank)

    def _answer(self, prompt: str) -> str:
        m = _OPTION_RE.search(prompt)
        if m:
            items = _ITEM_RE.findall(m.group(1))
            if items:
                digest = hashlib.blake2b(prompt.encode(), digest_size=8).digest()
                pick = int.from_bytes(digest, "big") % len(items)
                return items[pick][1].strip()
        digest = hashlib.blake2b(prompt.encode(), digest_size=8).hexdigest()[:16]
        return f"[echo:{digest}]"
