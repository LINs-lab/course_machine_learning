# Ported (semantics preserved) from
# project_collab_auto_research/graph/self_improve/edit_ledger.py: append-only
# JSONL, fold-by-id (last row per edit_id is the current state). Divergence:
# the source reverts via a local git branch; the base system stores the prior
# mutable-region bytes base64 in the applying row (``prev_mutable_b64``) and
# restores them directly — no git required, atomic because guard confines
# every edit to one sentinel region of one file by construction.
"""Append-only improvement ledger.

Lifecycle (source vocabulary): ``applying → observing → kept | reverted``,
plus ``failed`` (apply/observe error) and ``rejected`` (guard veto). **kept
== promoted**: the edit stays live and shapes subsequent cycles — that
re-entry is what makes the loop recursive self-improvement rather than a
pile of experiments.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

IMPROVEMENT_TRANSITIONS: dict[str, frozenset[str]] = {
    "applying": frozenset({"observing", "failed", "rejected"}),
    "observing": frozenset({"kept", "reverted", "failed"}),
}
IMPROVEMENT_TERMINAL = frozenset({"kept", "reverted", "failed", "rejected"})


class LedgerError(RuntimeError):
    pass


@dataclass
class EditRow:
    edit_id: str
    surface: str
    target: str
    rationale: str
    status: str
    waste_refs: list[str] = field(default_factory=list)
    prev_mutable_b64: str = ""
    candidate_mutable: str = ""
    applied_at_cycle: int = 0
    observed_cycles: list[str] = field(default_factory=list)


class EditLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            text = self.path.read_text()
        else:
            text = ""
        self._rows: list[dict] = [json.loads(line) for line in
                                  text.splitlines() if line.strip()]

    def append(self, row: EditRow) -> None:
        # A row may be born "applying" (live path) or already terminal
        # (guard veto / apply failure recorded as rejected / failed).
        if row.status != "applying" and row.status not in IMPROVEMENT_TERMINAL:
            raise LedgerError(f"bad initial status {row.status!r}")
        self._write(asdict(row))

    def update_status(self, edit_id: str, status: str,
                      observed_cycle: str | None = None) -> dict:
        cur = self.fold().get(edit_id)
        if cur is None:
            raise LedgerError(f"unknown edit {edit_id!r}")
        if status not in IMPROVEMENT_TRANSITIONS.get(cur["status"], frozenset()):
            raise LedgerError(
                f"illegal transition {cur['status']} -> {status}")
        new = dict(cur)
        new["status"] = status
        if observed_cycle:
            new["observed_cycles"] = list(cur.get("observed_cycles") or [])
            new["observed_cycles"].append(observed_cycle)
        self._write(new)
        return new

    def observe_cycle(self, edit_id: str, run_id: str) -> None:
        """Note that a run happened under observation (status unchanged:
        observing → observing is a bookkeeping event, not a transition)."""
        cur = self.fold().get(edit_id)
        if cur and cur["status"] == "observing":
            new = dict(cur)
            new["observed_cycles"] = list(cur.get("observed_cycles") or [])
            new["observed_cycles"].append(run_id)
            self._write(new)

    def mark_applied_at(self, edit_id: str, runs_index: int) -> None:
        """Record where observation started (index into the engine's run
        list) as a fresh appended row — the ledger is never rewritten."""
        cur = self.fold().get(edit_id)
        if cur is None:
            raise LedgerError(f"unknown edit {edit_id!r}")
        new = dict(cur)
        new["applied_at_cycle"] = runs_index
        self._write(new)

    def mark_terminated_at(self, edit_id: str, runs_index: int) -> None:
        """Record the run index at which the edit reached a terminal state
        (kept/reverted) — lets reports draw the exact observation window."""
        cur = self.fold().get(edit_id)
        if cur is None:
            raise LedgerError(f"unknown edit {edit_id!r}")
        if cur["status"] not in ("kept", "reverted"):
            return  # not terminal yet; nothing to record
        new = dict(cur)
        new["terminated_at"] = runs_index
        self._write(new)

    def fold(self) -> dict[str, dict]:
        """Current state per edit_id (last row wins)."""
        out: dict[str, dict] = {}
        for row in self._rows:
            out[row["edit_id"]] = row
        return out

    def active_observation(self) -> dict | None:
        """The edit whose CURRENT (folded) state is ``observing``.

        Raw-row scanning is wrong here: after a terminal row (kept/reverted),
        older observing rows still exist in the append-only log. Only the
        folded last-row-per-edit state counts (at most one, engine-enforced)."""
        folds = self.fold()
        for row in reversed(self._rows):
            cur = folds[row["edit_id"]]
            if cur["status"] == "observing":
                return cur
        return None

    def rows(self) -> list[dict]:
        return list(self._rows)

    def restore_bytes(self, edit_id: str) -> bytes:
        """The mutable-region bytes to restore on revert."""
        row = self.fold()[edit_id]
        return base64.b64decode(row["prev_mutable_b64"])

    def _write(self, row: dict) -> None:
        """Append exactly one JSON line (append-only truth, never rewritten)."""
        self._rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
