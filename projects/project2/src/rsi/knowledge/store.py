# Ported (semantics preserved, unified) from
# project_collab_auto_research_shared_kb/server/{question,hypothesis,insight,
# evidence,knowledge}/store.py: the source ships five near-duplicate sqlite
# stores (JSON blob + a few indexed columns); the base system folds the four
# versioned-entity stores into ONE generic SQLiteStore driven by a per-kind
# spec, and keeps evidence as a separate append-only EvidenceLog (a
# semantically different contract: no update/delete paths exist by design).
"""Sqlite-backed storage for the knowledge platform.

Entity records are JSON blobs (schema-validated at the platform boundary)
plus a few indexed columns; ``put(bump=True)`` performs the optimistic
version bump. FTS5 full-text search is enabled when the sqlite build has it,
degrading to LIKE (the source system's precedent for degraded-mode search).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

# kind -> (table, id_column, indexed columns with sqlite types, fts columns)
_COL_TYPES = {"status": "TEXT", "version": "INTEGER", "saturation": "REAL",
              "domain": "TEXT", "level": "INTEGER"}
_KINDS = {
    "question": ("questions", "id", ("status", "version", "saturation"),
                 ("statement",)),
    "hypothesis": ("hypotheses", "id", ("status", "version"), ()),
    "insight": ("insights", "insight_id", ("status",), ("statement",)),
    "node": ("nodes", "node_id", ("status", "version", "domain", "level"), ()),
}


class SQLiteStore:
    """One store per entity kind; one sqlite file per store."""

    def __init__(self, path: str | Path, kind: str):
        if kind not in _KINDS:
            raise ValueError(f"unknown entity kind {kind!r}")
        self.kind = kind
        self.table, self.id_col, self.cols, self.fts_cols = _KINDS[kind]
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        cols_sql = ", ".join(f"{c} {_COL_TYPES.get(c, 'TEXT')}"
                             for c in self.cols)
        self.db.executescript(
            f"CREATE TABLE IF NOT EXISTS {self.table} "
            f"({self.id_col} TEXT PRIMARY KEY, {cols_sql}, json TEXT NOT NULL)")
        self.fts_table = None
        if self.fts_cols:
            try:
                self.fts_table = f"{self.table}_fts"
                self.db.executescript(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.fts_table} "
                    f"USING fts5({self.id_col} UNINDEXED, "
                    f"{', '.join(self.fts_cols)})")
            except sqlite3.OperationalError:
                self.fts_table = None  # degraded: LIKE search

    # -- entity path (versioned) ------------------------------------------

    def put(self, record: dict, bump: bool = False,
            commit: bool = True) -> dict:
        """Upsert with optimistic version bump (source QuestionStore.put).

        ``commit=False`` defers the commit for multi-entity ops (merge /
        spawn / refine) so the whole op lands in ONE transaction — call
        ``self.commit()`` at the end (a crash mid-op then leaves nothing
        half-applied). Any sqlite failure rolls the transaction back: a
        write the caller saw fail must never materialize inside a LATER
        commit."""
        existing = self.get(record[self.id_col])
        version = ((existing["version"] + 1) if (existing and bump)
                   else record.get("version", 1))
        record = {**record, "version": version}
        col_vals = [record.get(c) for c in self.cols]
        assign = ", ".join(f"{c}=excluded.{c}" for c in self.cols)
        try:
            self.db.execute(
                f"INSERT INTO {self.table} ({self.id_col}, {', '.join(self.cols)}"
                f", json) VALUES ({', '.join('?' * (len(self.cols) + 2))})"
                f" ON CONFLICT({self.id_col}) DO UPDATE SET {assign},"
                f" json=excluded.json",
                (record[self.id_col], *col_vals, json.dumps(record)))
            if self.fts_table is not None:
                row = self.db.execute(
                    f"SELECT rowid FROM {self.fts_table} WHERE {self.id_col} = ?",
                    (record[self.id_col],)).fetchone()
                if row:
                    self.db.execute(
                        f"DELETE FROM {self.fts_table} WHERE rowid = ?",
                        (row[0],))
                self.db.execute(
                    f"INSERT INTO {self.fts_table} ({self.id_col}, "
                    f"{', '.join(self.fts_cols)}) VALUES ({', '.join('?' * (len(self.fts_cols) + 1))})",
                    (record[self.id_col],
                     *(str(record.get(c, "")) for c in self.fts_cols)))
            if commit:
                self.db.commit()
        except sqlite3.Error:
            self.db.rollback()
            raise
        return record

    def commit(self) -> None:
        self.db.commit()

    def get(self, obj_id: str) -> dict | None:
        row = self.db.execute(
            f"SELECT json FROM {self.table} WHERE {self.id_col} = ?",
            (obj_id,)).fetchone()
        return json.loads(row["json"]) if row else None

    def all(self) -> list[dict]:
        rows = self.db.execute(
            f"SELECT json FROM {self.table} ORDER BY rowid").fetchall()
        return [json.loads(r["json"]) for r in rows]

    def search(self, text: str, limit: int = 20) -> list[dict]:
        """FTS5 when available, LIKE fallback otherwise."""
        if not text:
            return []
        if self.fts_table is not None:
            q = " ".join(f'"{t}"' for t in text.split() if t)
            try:
                rows = self.db.execute(
                    f"SELECT {self.id_col} FROM {self.fts_table} "
                    f"WHERE {self.fts_cols[0]} MATCH ? LIMIT ?",
                    (q, limit)).fetchall()
                return [self.get(r[0]) for r in rows]
            except sqlite3.OperationalError:
                pass
        pat = text.lower()
        rows = self.db.execute(
            f"SELECT json FROM {self.table}").fetchall()
        # match the same columns FTS indexes (NOT the whole JSON blob — a
        # substring like "open" would hit every record's "status": "open")
        out = []
        for r in rows:
            rec = json.loads(r["json"])
            hay = " ".join(str(rec.get(c, "")) for c in self.fts_cols).lower()
            if pat in hay:
                out.append(rec)
            if len(out) >= limit:
                break
        return out


class EvidenceLog:
    """Append-only evidence storage (source EvidenceStore / ADR-002).

    No update/delete paths exist by design; corrections append a new record
    with ``replay.replay_of`` pointing at the prior one. ``seq`` is the
    notification cursor (poll)."""

    DDL = """
CREATE TABLE IF NOT EXISTS evidence (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    evidence_kind TEXT NOT NULL,
    epoch TEXT NOT NULL,
    run_id TEXT,
    replayable INTEGER NOT NULL DEFAULT 1,
    json TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence(run_id);
"""

    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(self.DDL)

    def append(self, record: dict) -> None:
        rr = record.get("run_result") or {}
        self.db.execute(
            "INSERT INTO evidence (id, evidence_kind, epoch, run_id,"
            " replayable, json, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record["id"], record["evidence_kind"], record["epoch"],
             rr.get("run_id"), int(record["replay"]["replayable"]),
             json.dumps(record), time.time()))
        self.db.commit()

    def get(self, evidence_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT json FROM evidence WHERE id = ?", (evidence_id,)).fetchone()
        return json.loads(row["json"]) if row else None

    def by_run(self, run_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT json FROM evidence WHERE run_id = ? ORDER BY seq",
            (run_id,)).fetchall()
        return [json.loads(r["json"]) for r in rows]

    def by_target(self, target_id: str, direction: str | None = None,
                  limit: int = 50) -> list[dict]:
        out = []
        for r in self.db.execute(
                "SELECT json FROM evidence ORDER BY seq").fetchall():
            rec = json.loads(r["json"])
            for link in rec.get("links", []):
                if link["to"] == target_id and (direction is None
                                                or link["type"] == direction):
                    out.append(rec)
                    break
        return out[:limit]

    def all(self, limit: int = 200) -> list[dict]:
        rows = self.db.execute(
            "SELECT json FROM evidence ORDER BY seq LIMIT ?", (limit,)).fetchall()
        return [json.loads(r["json"]) for r in rows]

    def poll(self, cursor: int = 0, limit: int = 100) -> tuple:
        rows = self.db.execute(
            "SELECT seq, id FROM evidence WHERE seq > ? ORDER BY seq LIMIT ?",
            (cursor, limit)).fetchall()
        next_cursor = rows[-1]["seq"] if rows else cursor
        return ([{"seq": r["seq"], "evidence_id": r["id"]} for r in rows],
                next_cursor)
