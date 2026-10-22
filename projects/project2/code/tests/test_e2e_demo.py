"""End-to-end acceptance: the whole system runs keyless on the demo task.

This is the gate for "runs a minimal demo" — everything else in the suite is
unit-level. Runs the engine twice with the same seed to assert determinism.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rsi.core.engine import EngineConfig, build_default_engine

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    base = tmp_path_factory.mktemp("session")
    task = base / "task" / "quad-optim"
    shutil.copytree(REPO / "tasks" / "quad-optim", task)
    engine = build_default_engine(EngineConfig(
        task_dir=str(task), out_dir=str(base / "out"), cycles=6,
        middle_every=3, seed=7))
    summary = engine.run("Minimize final_loss on quad-optim")
    return base / "out", summary, engine


def test_all_cycles_recorded(session):
    out, summary, engine = session
    lines = [json.loads(l) for l in
             (out / "state" / "runs.jsonl").read_text().splitlines()]
    assert len(lines) == 6 == summary["cycles"]
    assert summary["keep"] >= 1
    assert {r["status"] for r in lines} <= {"keep", "discard", "fail"}
    # all runs archived under their submission epoch
    assert all(r["epoch"] == "e001" for r in lines)


def test_evidence_and_links_in_kb(session):
    out, _s, _e = session
    rows = _kb(out, "evidence")
    assert len(rows) == 6  # one per run, idempotent
    supports = [r for r in rows
                if any(l["type"] == "supports" for l in r["links"])]
    assert len(supports) >= 1


def test_hypothesis_confidence_moved(session):
    out, summary, _e = session
    h = _kb_hypothesis(out)
    assert h["evidence_counts"]["support"] >= 1
    assert h["confidence"] != 0.5 or h["status"] in ("supported", "refuted")


def test_question_graph_updated(session):
    out, summary, _e = session
    questions = _kb(out, "questions")
    ids = {q["id"] for q in questions}
    assert "quad-optim-q0" in ids
    q0 = next(q for q in questions if q["id"] == "quad-optim-q0")
    assert q0["best_result"] is not None
    # saturation is bound at creation and kept in [0,1]; a still-improving
    # question legitimately stays open at low saturation
    assert "saturation" in q0 and 0.0 <= q0["saturation"] <= 1.0


def test_self_improvement_reached_terminal_status(session):
    out, summary, _e = session
    ledger_rows = [json.loads(l) for l in
                   (out / "state" / "ledger.jsonl").read_text().splitlines()]
    assert ledger_rows, "expected at least one self-improvement event"
    terminal = summary["improvements"]
    assert any(v in ("kept", "reverted", "rejected", "failed")
               for v in terminal.values())
    # kept ⇒ the strategy artifact still parses and differs from default
    if "kept" in terminal.values():
        from rsi.core.proposer import load_strategy
        strat = load_strategy(out / "state" / "strategy.py")
        assert strat["PRIORITY"] != ["MOMENTUM", "LR", "CLIP"]


def test_determinism(tmp_path):
    task = tmp_path / "task" / "quad-optim"
    shutil.copytree(REPO / "tasks" / "quad-optim", task)
    outs = []
    for i in (1, 2):
        out = tmp_path / f"run{i}"
        eng = build_default_engine(EngineConfig(
            task_dir=str(task), out_dir=str(out), cycles=6, middle_every=3,
            seed=7))
        s = eng.run("Minimize final_loss on quad-optim")
        outs.append((s, [json.loads(l)["metric"]["value"] for l in
                         (out / "state" / "runs.jsonl").read_text()
                         .splitlines()]))
    assert outs[0] == outs[1]


def test_cli_entrypoints(session):
    out, _s, _e = session
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]}
    r1 = subprocess.run([sys.executable, "-m", "rsi.knowledge", "--kb",
                         str(out / "kb"), "frontier"],
                        capture_output=True, text=True, env=env)
    assert r1.returncode == 0 and "quad-optim" in r1.stdout
    r2 = subprocess.run([sys.executable, "-m", "rsi.evaluation", "--runs",
                         str(out / "state" / "runs.jsonl"), "report"],
                        capture_output=True, text=True, env=env)
    assert r2.returncode == 0 and "e001" in r2.stdout


def _kb(out, what):
    import sqlite3
    table = {"questions": "questions", "evidence": "evidence"}[what]
    db = sqlite3.connect(out / "kb" / f"{table}.sqlite")
    db.row_factory = sqlite3.Row
    return [json.loads(r["json"]) for r in db.execute(
        f"SELECT json FROM {table} ORDER BY rowid")]


def _kb_hypothesis(out):
    import sqlite3
    db = sqlite3.connect(out / "kb" / "hypotheses.sqlite")
    db.row_factory = sqlite3.Row
    row = db.execute(
        "SELECT json FROM hypotheses ORDER BY rowid LIMIT 1").fetchone()
    return json.loads(row["json"])
