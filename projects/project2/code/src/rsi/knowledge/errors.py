# Ported (semantics preserved) from
# project_collab_auto_research_shared_kb/protocol/__init__.py + server/common/errors.py:
# the frozen error-code vocabulary travels with the contract; the HTTP/MCP
# envelope is replaced by a typed exception (in-process calls, no transport).
"""Frozen error codes for the knowledge platform (v0.2 vocabulary)."""

from __future__ import annotations

ERROR_CODES = (
    "RSI_INVALID_EDGE_TYPE",
    "RSI_INVALID_TRANSITION",
    "RSI_MISSING_TESTABILITY",
    "RSI_MISSING_EPOCH",
    "RSI_MISSING_BUDGET",
    "RSI_DANGLING_REF",
    "RSI_VERSION_CONFLICT",
    "RSI_EPOCH_FROZEN",
    "RSI_IDEMPOTENT_REPLAY",
    "RSI_FORBIDDEN_ACTOR",
)


class PlatformError(RuntimeError):
    """A rejected write, carrying the frozen code and retryability.

    ``RSI_IDEMPOTENT_REPLAY`` is special: it is a SUCCESS-shaped rejection —
    the write already happened and ``result`` holds the first outcome.
    """

    def __init__(self, code: str, message: str, *, retryable: bool = False,
                 result: dict | None = None):
        if code not in ERROR_CODES:
            raise ValueError(f"unknown platform error code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable
        self.result = result
