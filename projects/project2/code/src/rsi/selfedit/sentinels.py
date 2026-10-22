# Ported (semantics preserved, near-verbatim) from
# project_collab_auto_research/graph/self_improve/sentinels.py.
"""Tight parser for the single proposer-editable region in a guarded source file.

The boundary tokens are distinctive COMMENT strings; the contract is strict
because this is the trust boundary of the self-improvement loop: code outside
the sentinels may never change, and a body may not smuggle a new boundary.
"""
from __future__ import annotations

MUTABLE_OPEN = "# <<<SELF-IMPROVE-MUTABLE"
MUTABLE_CLOSE = "# SELF-IMPROVE-MUTABLE>>>"


class SentinelError(RuntimeError):
    """The source does not contain exactly one well-formed mutable region."""


def split_regions(source: str, open_token: str = None, close_token: str = None):
    """(before, mutable, after) or raise SentinelError. Exactly one OPEN and one
    CLOSE, open strictly before close, and NO other occurrence of either token
    anywhere (so a body cannot smuggle a boundary). The mutable region EXCLUDES
    the sentinel lines themselves."""
    ot = open_token or MUTABLE_OPEN
    ct = close_token or MUTABLE_CLOSE
    if source.count(ot) != 1 or source.count(ct) != 1:
        raise SentinelError("expected exactly one open and one close sentinel")
    o = source.index(ot)
    c = source.index(ct)
    if o >= c:
        raise SentinelError("open sentinel must precede close sentinel")
    before = source[:o]
    # mutable = between the open line and the close line (sentinel lines excluded).
    # The open sentinel MUST be terminated by a newline (else malformed →
    # SentinelError, never a raw ValueError).
    nl = source.find("\n", o)
    if nl == -1:
        raise SentinelError("open sentinel not terminated by a newline")
    open_line_end = nl + 1
    mutable = source[open_line_end:c]
    after = source[c:]
    return before, mutable, after


def replace_mutable(source: str, new_mutable: str) -> str:
    """Rebuild the source with the mutable region swapped (sentinels preserved)."""
    before, _, after = split_regions(source)
    return f"{before}{MUTABLE_OPEN}\n{new_mutable}{after}"
