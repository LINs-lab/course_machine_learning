"""Regression tests for the SECOND full code review round.

Each test names the finding it pins down. Round-2 themes: guard smuggles
beyond statement shape, docstring-immune knob rewriting, frozen-benchmark
env leaks, median-anchored diagnostics, question transition evidence-chain
persistence, state-machine obedience for merge/refine, correction-aware
confidence aggregation, persistent envelope idempotency, and grid-walk
liveness (a failing value must never head-of-line block exploration).
"""
import json
import subprocess
import types
from pathlib import Path

import pytest

from rsi.core.proposer import HeuristicProposer, STRATEGY_TEMPLATE
from rsi.evaluation.diagnostics import (SATURATE_QUESTION, UPGRADE_BENCHMARK,
                                        NaiveDiagnostician)
from rsi.knowledge.concepts import EvidenceLink, MachineEvidence
from rsi.knowledge.errors import PlatformError
from rsi.knowledge.platform import KnowledgePlatform
from rsi.selfedit import sentinels
from rsi.selfedit.guard import check_candidate
from rsi.tasks import runner
from rsi.tasks.runner import LocalRunner, apply_params, parse_params
from rsi.tasks.spec import TaskError, load_task
from rsi.core.types import RunRecord
from rsi.evaluation.gates import MetricReading


# -- guard: shape lock covers targets AND value types -------------------------

def _candidate(mutable: str) -> str:
    return sentinels.replace_mutable(STRATEGY_TEMPLATE, mutable)


def test_guard_rejects_multi_target_smuggle():
    # 'PRIORITY = __import__("os").system = [...]' has the same Name-only
    # signature as a plain assign — the lock must see ALL targets
    cand = _candidate('PRIORITY = __import__("os").system = ["LR"]\n'
                      'EXPLORE_WIDTH = 6\n')
    v = check_candidate(STRATEGY_TEMPLATE, cand)
    assert not v.ok and v.layer == "mutable-shape"


def test_guard_rejects_value_type_change():
    cand = _candidate('PRIORITY = ["MOMENTUM", "LR", "CLIP"]\n'
                      'EXPLORE_WIDTH = "lots"\n')
    v = check_candidate(STRATEGY_TEMPLATE, cand)
    assert not v.ok and v.layer == "mutable-shape"


def test_guard_rejects_list_to_string_retype():
    cand = _candidate('PRIORITY = "LR"\nEXPLORE_WIDTH = 6\n')
    v = check_candidate(STRATEGY_TEMPLATE, cand)
    assert not v.ok and v.layer == "mutable-shape"


def test_guard_accepts_value_rotation():
    cand = _candidate('PRIORITY = ["LR", "CLIP", "MOMENTUM"]\n'
                      'EXPLORE_WIDTH = 6\n')
    assert check_candidate(STRATEGY_TEMPLATE, cand).ok


# -- runner: AST-located knobs (docstring immunity) ---------------------------

def test_apply_params_ignores_docstring_lookalike():
    src = '"""Docstring.\nLR = 0.9\n"""\nLR = 1e-3\n'
    out = apply_params(src, {"LR": 0.5})
    assert "LR = 0.9" in out                       # docstring untouched
    assert out.rstrip().endswith("LR = 0.5")       # the real knob rewritten
    compile(out, "m.py", "exec")


def test_parse_params_ignores_docstring_lookalike():
    src = '"""Docstring.\nLR = 0.9\n"""\nLR = 1e-3\n'
    assert parse_params(src) == {"LR": 1e-3}


def test_parse_and_apply_annotated_assignment():
    src = "X: float = 1.0\nY = 2\n"
    assert parse_params(src) == {"X": 1.0, "Y": 2}
    out = apply_params(src, {"X": 3.5})
    assert "X = 3.5" in out
    compile(out, "m.py", "exec")


def test_apply_params_multiline_value_span():
    src = "GRID = [\n    1,\n    2,\n]\nLR = 0.1\n"
    out = apply_params(src, {"GRID": [3, 4]})
    assert "GRID = [3, 4]" in out
    compile(out, "m.py", "exec")


def test_dispatch_filters_ambient_rsi_env(monkeypatch, tmp_path):
    # a shell-exported RSI_STEPS must NOT reach the task subprocess — the
    # harness owns the RSI_* namespace (frozen-benchmark integrity)
    captured = {}

    class _Proc:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(argv, cwd, env, capture_output, text, timeout):
        captured.update(env)
        return _Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("RSI_STEPS", "99999")
    spec = types.SimpleNamespace(entrypoint=["python3", "run.py"],
                                 time_budget=5)
    LocalRunner().dispatch(spec, tmp_path, {"RSI_EPOCH": "e001"})
    assert "RSI_STEPS" not in captured
    assert captured["RSI_EPOCH"] == "e001"


# -- diagnostics: median-anchored exclusion + recommendation routing ----------

def _run(i, value, baseline=None, status="keep", epoch="e001",
         direction="min"):
    return RunRecord(
        run_id=f"cycle-{i:03d}", proposal_id=f"p{i}", hypothesis_id="h",
        question_id="q", task="t", epoch=epoch, status=status,
        metric=MetricReading(name="m", direction=direction, value=value,
                             baseline=baseline, failure_mode=None),
        variable_changed=f"v{i}", created="", duration_s=0.1,
        verdict_reasons=[])


def test_zero_best_does_not_blind_diagnostics():
    # quad-optim clamps at exactly 0.0 — a best-anchored outlier scale would
    # exclude every informative run once the task is solved
    diag = NaiveDiagnostician()
    runs = [_run(1, 0.5), _run(2, 0.0, baseline=0.5),
            _run(3, 0.4, baseline=0.0, status="discard")]
    r = diag.diagnose(runs, "e001", "q")
    assert r.mature and r.n_runs == 3


def test_epoch_less_run_belongs_to_no_epoch():
    diag = NaiveDiagnostician()
    ghost = types.SimpleNamespace(
        status="keep",
        metric=MetricReading(name="m", direction="min", value=1e9,
                             baseline=None, failure_mode=None))
    runs = [ghost, _run(1, 1.0)]
    r = diag.diagnose(runs, "e001", "q")
    assert r.n_runs == 1            # the epoch-less record is not counted


def test_saturated_high_discrimination_saturates_question():
    diag = NaiveDiagnostician()
    runs = [_run(1, 10.0), _run(2, 5.0, baseline=10.0),
            _run(3, 2.0, baseline=5.0)]
    runs += [_run(i, 2.5, baseline=2.0, status="discard")
             for i in range(4, 8)]
    r = diag.diagnose(runs, "e001", "q")
    assert r.saturation >= 0.8 and r.discrimination >= 1.0
    assert r.recommendation == SATURATE_QUESTION


def test_saturated_low_discrimination_upgrades_benchmark():
    diag = NaiveDiagnostician()
    runs = [_run(1, 10.0), _run(2, 5.0, baseline=10.0)]
    runs += [_run(i, 4.9, baseline=4.9, status="discard")
             for i in range(3, 9)]
    r = diag.diagnose(runs, "e001", "q")
    assert r.saturation >= 0.8 and r.discrimination < 1.0
    assert r.recommendation == UPGRADE_BENCHMARK


# -- spec: fail-fast manifest validation ---------------------------------------

def _write_task(tmp_path, extra_fm="", editable="solver.py"):
    d = tmp_path / "mytask"
    d.mkdir()
    (d / "solver.py").write_text("X = 1\n")
    (d / "run.py").write_text("print('ok')\n")
    (d / "program.md").write_text(
        f"---\nschema_version: 1\nname: mytask\n"
        f"entrypoint: [python3, run.py]\n"
        f"metric: {{name: m, direction: min, result_file: result.json,"
        f" result_key: k}}\n"
        f"editable: [{editable}]\n{extra_fm}---\nbody\n")
    return d


def test_spec_rejects_nonstring_result_file(tmp_path):
    d = _write_task(tmp_path)
    text = (d / "program.md").read_text().replace("result_file: result.json",
                                                  "result_file: 123")
    (d / "program.md").write_text(text)
    with pytest.raises(TaskError, match="metric.result_file"):
        load_task(d)


def test_spec_normalizes_dot_slash_editable(tmp_path):
    spec = load_task(_write_task(tmp_path, editable="./solver.py"))
    assert spec.editable == ["solver.py"]


def test_spec_rejects_entrypoint_traversal(tmp_path):
    d = _write_task(tmp_path)
    text = (d / "program.md").read_text().replace(
        "entrypoint: [python3, run.py]",
        "entrypoint: [python3, ../../evil.py]")
    (d / "program.md").write_text(text)
    with pytest.raises(TaskError, match="entrypoint"):
        load_task(d)


# -- platform: evidence-chain persistence + state-machine obedience -----------

def _q(i=0):
    return {"id": f"q{i}", "statement": f"st {i}", "scope": "sc",
            "benchmark_ref": {"name": "b", "epoch": "e001"},
            "status": "open", "version": 1, "saturation": 0.0,
            "uncertainty": [], "edges": []}


def _h(i=0):
    return {"id": f"q{i}-h0", "statement": "hyp",
            "derived_from": [{"interface": "I-Q", "object_id": f"q{i}"}],
            "testability": {"procedure": "p", "expected_evidence": "e",
                            "falsifiable_by": "f"},
            "status": "draft", "version": 1}


def _ev(i, run, to="q0-h0", ltype="supports", replay_of=None):
    e = MachineEvidence(
        id=f"ev{i}", epoch="e001",
        run_result={
            "run_id": run, "env": "t", "status": "keep", "failure_mode": None,
            "created": "2026-01-01T00:00:00+00:00", "duration_s": 1,
            "hypothesis": to, "variable_changed": "solver.LR",
            "metric": {"name": "m", "direction": "min", "value": 1.0,
                       "baseline": 2.0, "delta": 1.0},
            "benchmark_ref": {"name": "b", "epoch": "e001",
                              "task_version": "v1"}},
        links=[EvidenceLink(ltype, to, 0.5, 1.0)],
        run={"revision": run, "source_hash": "ab"}).to_dict()
    if replay_of:
        e["replay"]["replay_of"] = replay_of
    return e


def test_transition_persists_bare_evidence_refs(tmp_path):
    kb = KnowledgePlatform(tmp_path / "kb")
    kb.create_question(_q())
    kb.create_hypothesis(_h())
    kb.push_run(_ev(1, "r1"))
    kb.transition_question("q0", "partially_answered", evidence_refs=["ev1"])
    q = kb.get_question("q0")
    assert "ev1" in q["evidence_refs"]
    assert q["conversion_events"][-1]["evidence_refs"] == ["ev1"]


def test_transition_rejects_dangling_evidence_ref(tmp_path):
    kb = KnowledgePlatform(tmp_path / "kb")
    kb.create_question(_q())
    with pytest.raises(PlatformError) as ei:
        kb.transition_question("q0", "answered", evidence_refs=["ghost"])
    assert ei.value.code == "RSI_DANGLING_REF"


def test_refine_and_merge_obey_state_machine(tmp_path):
    kb = KnowledgePlatform(tmp_path / "kb")
    kb.create_question(_q(0))
    kb.create_question(_q(1))
    kb.transition_question("q0", "retired")
    with pytest.raises(PlatformError) as ei:
        kb.refine_question("q0", [dict(_q(2))])
    assert ei.value.code == "RSI_INVALID_TRANSITION"
    kb.create_question(_q(2))
    kb.transition_question("q2", "retired")
    with pytest.raises(PlatformError):
        kb.merge_questions("q2", "q1")


def test_merge_hypothesis_from_terminal_state_rejected(tmp_path):
    kb = KnowledgePlatform(tmp_path / "kb")
    kb.create_question(_q())
    kb.create_hypothesis(_h())
    kb.create_hypothesis({**_h(), "id": "q0-h1"})
    # draft -> merged is not a legal transition (only active -> merged is)
    with pytest.raises(PlatformError) as ei:
        kb.merge_hypotheses("q0-h0", "q0-h1")
    assert ei.value.code == "RSI_INVALID_TRANSITION"


def test_confidence_ignores_superseded_evidence(tmp_path):
    kb = KnowledgePlatform(tmp_path / "kb")
    kb.create_question(_q())
    kb.create_hypothesis(_h())
    kb.push_run(_ev(1, "r1", ltype="refutes"))
    # correction: E2 retracts E1 and relabels the link as supports
    kb.push_run(_ev(2, "r1", ltype="supports", replay_of="ev1"))
    out = kb.update_confidence("q0-h0")
    assert out["evidence_counts"] == {"support": 1, "refute": 0,
                                      "inconclusive": 0}


def test_change_proposal_idempotency_survives_restart(tmp_path):
    env = {"proposal_id": "cp-1", "actor": "engine",
           "targets": [{"interface": "I-Q", "object_id": "q9"}],
           "op": "update", "params": {"fields": {"saturation": 0.5}},
           "idempotency_key": "k-1"}
    kb1 = KnowledgePlatform(tmp_path / "kb")
    kb1.create_question({"id": "q9", "statement": "s", "scope": "sc",
                         "benchmark_ref": {"name": "b", "epoch": "e001"},
                         "status": "open", "version": 1, "saturation": 0.0,
                         "uncertainty": [], "edges": []})
    kb1.submit(env)
    kb2 = KnowledgePlatform(tmp_path / "kb")   # "restarted" process
    with pytest.raises(PlatformError) as ei:
        kb2.submit(env)
    assert ei.value.code == "RSI_IDEMPOTENT_REPLAY"


def test_update_question_rejects_empty_fields(tmp_path):
    kb = KnowledgePlatform(tmp_path / "kb")
    kb.create_question(_q())
    with pytest.raises(PlatformError, match="at least one field"):
        kb.update_question("q0", {})


def test_attach_evidence_is_idempotent(tmp_path):
    kb = KnowledgePlatform(tmp_path / "kb")
    kb.create_question(_q())
    kb.create_hypothesis(_h())
    kb.push_run(_ev(1, "r1"))
    kb.attach_evidence("q0", ["ev1"])
    kb.attach_evidence("q0", ["ev1"])
    assert kb.get_question("q0")["evidence_refs"] == ["ev1"]


# -- proposer liveness: a failing value must not block the grid ----------------

def _hist_run(qid, var, params, status):
    return RunRecord(
        run_id="cycle-001", proposal_id="p", hypothesis_id="h",
        question_id=qid, task="t", epoch="e001", status=status,
        metric=MetricReading(name="m", direction="min", value=None,
                             baseline=None, failure_mode="crash"),
        variable_changed=var, params=params, created="", duration_s=0.1,
        verdict_reasons=[])


class _Spec:
    class editable_file:  # noqa: D401 — only .read_text() is used
        @staticmethod
        def read_text():
            return "LR = 0.1\n"
    editable = ["solver.py"]


def _question():
    return {"id": "q0", "benchmark_ref": {"name": "t", "epoch": "e001"}}


def test_failed_value_does_not_block_grid_walk():
    p = HeuristicProposer()
    q = _question()
    hist = [_hist_run("q0", "baseline", {}, "keep"),
            _hist_run("q0", "solver.LR", {"LR": 0.2}, "fail")]
    prop = p.propose(q, _Spec(), {"LR": 0.1}, 1.0, hist,
                     {"PRIORITY": ["LR"], "EXPLORE_WIDTH": 6})
    assert prop is not None
    assert prop.params != {"LR": 0.2}      # the next candidate, not a retry


def test_failed_value_retried_once_then_exhausted():
    p = HeuristicProposer()
    q = _question()
    hist = [_hist_run("q0", "baseline", {}, "keep")]
    # value_grid(0.1) == [0.2, 0.05, 2.0, 0.005, 0.5, 0.02]: fail them all
    for v in (0.2, 0.05, 2.0, 0.005, 0.5, 0.02):
        hist.append(_hist_run("q0", "solver.LR", {"LR": v}, "fail"))
    # every grid value failed exactly once → each earns ONE deferred retry
    for _ in range(6):
        prop = p.propose(q, _Spec(), {"LR": 0.1}, 1.0, hist,
                         {"PRIORITY": ["LR"], "EXPLORE_WIDTH": 6})
        assert prop is not None and "retry" in prop.rationale
        hist.append(_hist_run("q0", prop.variable_changed, prop.params,
                              "fail"))
    # all pairs twice-failed → dead; grid exhausted → None (slow-loop trigger)
    assert p.propose(q, _Spec(), {"LR": 0.1}, 1.0, hist,
                     {"PRIORITY": ["LR"], "EXPLORE_WIDTH": 6}) is None


def test_failed_baseline_is_not_reproposed_forever():
    p = HeuristicProposer()
    q = _question()
    hist = [_hist_run("q0", "baseline", {}, "fail")]
    prop = p.propose(q, _Spec(), {"LR": 0.1}, None, hist,
                     {"PRIORITY": ["LR"], "EXPLORE_WIDTH": 6})
    assert prop is not None and prop.variable_changed != "baseline"


# -- engine: epoch upgrade CONTINUES the question (threshold-collision fix) ---

def test_upgrade_benchmark_continues_same_question(tmp_path):
    """The diagnostician's upgrade_benchmark and the slow loop's abandon used
    to share ONE threshold (0.8): every upgrade instantly abandoned the very
    question that motivated it, and the re-baseline path was dead code. Now
    an upgrade resets the question's saturation and the question continues
    on the new epoch (with a baseline re-measurement)."""
    from rsi.core.engine import EngineConfig, build_default_engine
    from rsi.evaluation.diagnostics import DiagnosisReport

    class _AlwaysUpgrade:
        def diagnose(self, history, epoch, question_id=""):
            return DiagnosisReport(
                question_id=question_id, epoch=epoch, n_runs=len(history),
                saturation=0.95, noise_floor=0.0, discrimination=0.5,
                recommendation="upgrade_benchmark", mature=True)

    repo = Path(__file__).resolve().parent.parent
    cfg = EngineConfig(task_dir=str(repo / "tasks" / "quad-optim"),
                       out_dir=str(tmp_path / "out"), cycles=6,
                       middle_every=3, seed=7, improver="noop")
    engine = build_default_engine(cfg)
    engine.diagnostician = _AlwaysUpgrade()
    summary = engine.run("Minimize final_loss on quad-optim")
    assert len(summary["epochs"]) >= 2                    # upgrade fired
    assert summary["questions"] == ["quad-optim-q0"]      # NOT abandoned
    runs = [json.loads(l) for l in
            (tmp_path / "out" / "state" / "runs.jsonl").read_text()
            .splitlines()]
    e2 = [r for r in runs if r["epoch"] == "e002"]
    assert e2 and e2[0]["variable_changed"] == "baseline"  # re-baselined
