import pytest

from rsi.tasks.runner import (LocalRunner, TaskRunError, apply_params,
                              parse_params, stage_workdir)
from rsi.tasks.spec import load_task


def test_apply_params_rewrites_assignment(demo_task):
    spec = load_task(demo_task)
    src = (spec.editable_file).read_text()
    out = apply_params(src, {"LR": 0.5, "MOMENTUM": 0.9})
    assert "LR = 0.5" in out and "MOMENTUM = 0.9" in out
    assert "CLIP = 1e9" in out  # untouched knob stays


def test_apply_params_unknown_knob_rejected():
    with pytest.raises(TaskRunError):
        apply_params("LR = 1\n", {"NOT_THERE": 1})


def test_parse_params_roundtrip():
    src = "LR = 1e-3\nMOMENTUM = 0.0\n"
    assert parse_params(src, ["LR", "MOMENTUM"]) == {"LR": 1e-3,
                                                     "MOMENTUM": 0.0}


def test_stage_workdir_only_editable_differs(demo_task, tmp_path):
    spec = load_task(demo_task)
    work = stage_workdir(spec, tmp_path / "w", None, {"LR": 0.25})
    staged = (work / "solver.py").read_text()
    assert "LR = 0.25" in staged
    # everything else identical to the pristine bundle
    for f in ("run.py", "program.md"):
        assert (work / f).read_bytes() == (spec.env_dir / f).read_bytes()


def test_stage_workdir_accumulates_incumbent(demo_task, tmp_path):
    spec = load_task(demo_task)
    inc = stage_workdir(spec, tmp_path / "w1", None, {"LR": 0.25})
    inc_bytes = (inc / "solver.py").read_bytes()
    second = stage_workdir(spec, tmp_path / "w2", inc_bytes, {"MOMENTUM": 0.9})
    text = (second / "solver.py").read_text()
    assert "LR = 0.25" in text and "MOMENTUM = 0.9" in text  # keeps accumulate


def test_local_runner_dispatch_and_timeout(demo_task, tmp_path):
    spec = load_task(demo_task)
    work = stage_workdir(spec, tmp_path / "w", None, {})
    runner = LocalRunner()
    res = runner.dispatch(spec, work,
                          env_extra={"RSI_BENCHMARK_SEED": "7",
                                     "RSI_EPOCH": "e001"})
    assert res.exit_code == 0 and not res.timed_out
    import json
    data = json.loads((work / "result.json").read_text())
    assert data["observations"]["epoch"] == "e001"
    assert data["observations"]["seed"] == 7
    assert res.log_path.exists()


def test_scope_guard_rejects_tampered_run_py(demo_task, tmp_path):
    import shutil
    from rsi.tasks.runner import _scope_guard
    spec = load_task(demo_task)
    work = tmp_path / "staged"
    shutil.copytree(demo_task, work)
    (work / "run.py").write_text("tampered = True\n")  # non-editable changed
    with pytest.raises(TaskRunError, match="scope guard"):
        _scope_guard(spec, work)
