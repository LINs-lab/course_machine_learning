"""E2E smoke for the iris-ood task: the OOD-generalization research task.

Asserts the discrimination contract holds under the engine: the baseline
lands near chance, and at least one knob keeps materially above it within a
small budget (the full gradient — knob ceiling ≈0.87, rewrite ceiling
≈0.93+ — is documented in tasks/iris-ood/program.md).
"""
import json
import shutil
from pathlib import Path

import pytest

from rsi.core.engine import EngineConfig, build_default_engine

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def iris_session(tmp_path_factory):
    base = tmp_path_factory.mktemp("iris")
    task = base / "task" / "iris-ood"
    shutil.copytree(REPO / "tasks" / "iris-ood", task)
    engine = build_default_engine(EngineConfig(
        task_dir=str(task), out_dir=str(base / "out"), cycles=8,
        middle_every=3, seed=7))
    summary = engine.run("Maximize ood_accuracy on iris-ood")
    return base / "out", summary


def test_task_has_discrimination():
    # direct scorer probe (no engine): baseline near chance, CLIP=0.5 well above
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(REPO / "tasks" / "iris-ood", Path(td) / "t")
        work = Path(td) / "t"
        env = {**os.environ, "RSI_BENCHMARK_SEED": "7"}
        subprocess.run(["python3", "run.py"], cwd=work, env=env, check=True)
        base = json.loads((work / "result.json").read_text())["ood_accuracy"]
        assert base <= 0.45  # near chance under the warp
        src = (work / "model.py").read_text()
        (work / "model.py").write_text(src.replace("CLIP = 10.0",
                                                   "CLIP = 0.5"))
        subprocess.run(["python3", "run.py"], cwd=work, env=env, check=True)
        good = json.loads((work / "result.json").read_text())["ood_accuracy"]
        assert good >= 0.8  # the single-knob path is reachable


def test_engine_improves_over_baseline(iris_session):
    out, summary = iris_session
    runs = [json.loads(l) for l in
            (out / "state" / "runs.jsonl").read_text().splitlines()]
    baseline = next(r["metric"]["value"] for r in runs
                    if r["variable_changed"] == "baseline")
    best_kept = max((r["metric"]["value"] for r in runs
                     if r["status"] == "keep"), default=None)
    assert best_kept is not None and best_kept > baseline + 0.1


def test_engine_kb_populated(iris_session):
    out, summary = iris_session
    questions = [json.loads(r[0]) for r in _rows(out, "questions")]
    assert any(q["id"] == "iris-ood-q0" for q in questions)
    assert summary["cycles"] >= 6


def _rows(out, table):
    import sqlite3
    db = sqlite3.connect(out / "kb" / f"{table}.sqlite")
    return db.execute(f"SELECT json FROM {table} ORDER BY rowid").fetchall()
