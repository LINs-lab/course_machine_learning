# Ported (verbatim) from
# project_collab_auto_research_shared_kb/server/common/statemachine.py
# (frozen v0.2 §3.2 / §4.1).
"""Explicit state-machine transition tables (frozen v0.2 §3.2 / §4.1)."""

from __future__ import annotations

from typing import Dict, FrozenSet

# Question (§3.2)
QUESTION_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    "open": frozenset(
        {"partially_answered", "answered", "refined", "merged", "retired"}
    ),
    "partially_answered": frozenset({"answered", "refined"}),
    "refined": frozenset({"open", "answered"}),
    "answered": frozenset({"open"}),
    "merged": frozenset(),
    "retired": frozenset(),
}
# transitions that must carry a conversion_event with evidence_refs
QUESTION_EVIDENCE_REQUIRED = frozenset({"partially_answered", "answered"})

# Hypothesis (§4.1)
HYPOTHESIS_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    "draft": frozenset({"active"}),
    "active": frozenset({"supported", "refuted", "merged"}),
    "supported": frozenset(),
    "refuted": frozenset(),
    "merged": frozenset(),
}
# supported/refuted require evidence counts at epoch thresholds
HYPOTHESIS_EVIDENCE_REQUIRED = frozenset({"supported", "refuted"})


def legal(table: Dict[str, FrozenSet[str]], frm: str, to: str) -> bool:
    return to in table.get(frm, frozenset())
