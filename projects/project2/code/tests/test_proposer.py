from rsi.core.proposer import (HeuristicProposer, ensure_strategy_file,
                               load_strategy)
from rsi.tasks.spec import load_task


def test_strategy_file_roundtrip(tmp_path):
    p = ensure_strategy_file(tmp_path / "state" / "strategy.py")
    assert ensure_strategy_file(p) == p  # idempotent
    s = load_strategy(p)
    assert s["PRIORITY"] == ["MOMENTUM", "LR", "CLIP"]
    assert s["EXPLORE_WIDTH"] == 6


def test_strategy_reads_only_mutable_region(tmp_path):
    p = ensure_strategy_file(tmp_path / "state" / "strategy.py")
    text = p.read_text().replace('"""Proposer strategy',
                                 'FROZEN_MARKER = 123\n"""Proposer strategy')
    p.write_text(text)
    assert "FROZEN_MARKER" not in load_strategy(p)


def _question():
    return {"id": "quad-optim-q0", "benchmark_ref": {"name": "quad-optim",
                                                     "epoch": "e001"}}


def test_baseline_proposal_first(demo_task):
    spec = load_task(demo_task)
    prop = HeuristicProposer().propose(_question(), spec, None, None, [], {})
    assert prop.variable_changed == "baseline" and prop.params == {}


def test_menu_walk_deterministic_and_single_variable(demo_task):
    spec = load_task(demo_task)
    hist = []

    def _hist(p, status="discard"):
        hist.append(type("R", (), {
            "variable_changed": p.variable_changed, "params": dict(p.params),
            "status": status, "hypothesis_id": "h",
            "question_id": "quad-optim-q0"})())

    p1 = HeuristicProposer().propose(_question(), spec, None, 1.0, hist, {})
    _hist(p1)
    p2 = HeuristicProposer().propose(_question(), spec, None, 1.0, hist, {})
    assert p1.params != p2.params          # walks the grid, never repeats
    assert len(p2.params) == 1             # single-variable control
    # knobs come from the task's editable file, not a hardcoded menu
    assert p1.variable_changed.startswith("solver.")


def test_knobs_derived_from_editable_file(tmp_path):
    # a task whose editable file uses OTHER knob names must not crash the
    # proposer (regression: the hardcoded LR/MOMENTUM/CLIP menu did)
    task = tmp_path / "task" / "weird"
    task.mkdir(parents=True)
    (task / "algo.py").write_text("STEP_SIZE = 0.1\nTEMP = 1.0\n")
    (task / "program.md").write_text("""---
schema_version: 1
name: weird
entrypoint: ["python3", "algo.py"]
metric:
  name: score
  direction: min
  result_file: result.json
  result_key: score
editable: ["algo.py"]
---
""")
    spec = load_task(task)
    p = HeuristicProposer().propose(_question(), spec, None, 1.0, [], {})
    assert p is not None and p.variable_changed == "algo.STEP_SIZE"


def test_strategy_changes_proposal_order(demo_task):
    spec = load_task(demo_task)
    strat = {"PRIORITY": ["LR", "MOMENTUM", "CLIP"], "EXPLORE_WIDTH": 3}
    p = HeuristicProposer().propose(_question(), spec, None, 1.0, [], strat)
    assert p.variable_changed == "solver.LR"


def test_exhaustion_returns_none(demo_task):
    spec = load_task(demo_task)
    strat = {"EXPLORE_WIDTH": 1}  # one candidate per knob
    # first grid candidate per knob: LR 1e-3*2, MOMENTUM zero-case 0.9,
    # CLIP 1e9*2 — all tried ⇒ exhausted
    hist = [
        type("R", (), {"variable_changed": "solver.LR",
                       "params": {"LR": 0.002}, "status": "discard",
                       "hypothesis_id": "h",
                       "question_id": "quad-optim-q0"})(),
        type("R", (), {"variable_changed": "solver.MOMENTUM",
                       "params": {"MOMENTUM": 0.9}, "status": "discard",
                       "hypothesis_id": "h",
                       "question_id": "quad-optim-q0"})(),
        type("R", (), {"variable_changed": "solver.CLIP",
                       "params": {"CLIP": 2e9}, "status": "discard",
                       "hypothesis_id": "h",
                       "question_id": "quad-optim-q0"})(),
    ]
    assert HeuristicProposer().propose(
        _question(), spec, None, 1.0, hist, strat) is None
