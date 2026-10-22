"""Regression tests for the issues found in the first full code review.

Each test names the review finding it pins down (see docs/ and the commit
history for the full review report).
"""
import json
import shutil

import pytest

from rsi.core.engine import EngineConfig, build_default_engine
from rsi.evaluation.benchmark import EpochManager, EpochStateError
from rsi.evaluation.diagnostics import Diagnostician, DiagnosisReport
from rsi.knowledge.concepts import MachineEvidence
from rsi.knowledge.errors import PlatformError
from rsi.knowledge.platform import KnowledgePlatform
from rsi.tasks.runner import LocalRunner, apply_params
from rsi.tasks.spec import TaskMetric, TaskSpec

REPO = __import__("pathlib").Path(__file__).resolve().parent.parent


# -- runner: apply_params scope (finding A4) ----------------------------------

def test_apply_params_never_rewrites_indented_local():
    src = ("def helper():\n    LR = 0.5\n    return LR\n\nLR = 1e-3\n")
    out = apply_params(src, {"LR": 1e-2})
    assert "    LR = 0.5" in out          # the local is untouched
    assert out.splitlines()[4] == "LR = 0.01"


def test_apply_params_unknown_indented_only_knob_rejected():
    with pytest.raises(Exception, match="no assignment"):
        apply_params("def f():\n    K = 1\n", {"K": 2})


def test_dispatch_missing_binary_is_a_fail_not_a_crash(tmp_path):
    # finding A8: OSError from subprocess must become a DispatchResult
    work = tmp_path / "w"
    work.mkdir()
    spec = TaskSpec(name="t", entrypoint=["./does-not-exist"],
                    time_budget=5,
                    metric=TaskMetric("m", "min", "r.json", "k"),
                    editable=["e.py"], readonly=[], env_dir=work,
                    protocol_body="")
    res = LocalRunner().dispatch(spec, work, {})
    assert res.exit_code == 127 and not res.timed_out
    assert "dispatch error" in (work / "run.log").read_text()


# -- platform: idempotency + envelope contract (findings B1/B2/B3/B4) ---------

def _kb(tmp_path):
    return KnowledgePlatform(tmp_path / "kb")


def _q():
    return {"id": "q0", "statement": "s", "scope": "sc",
            "benchmark_ref": {"name": "b", "epoch": "e001"},
            "status": "open", "version": 1, "saturation": 0.0,
            "uncertainty": [], "edges": []}


def _h(kb):
    kb.create_question(_q())
    return {"id": "q0-h0", "statement": "st",
            "derived_from": [{"interface": "I-Q", "object_id": "q0"}],
            "testability": {"procedure": "p", "expected_evidence": "e",
                            "falsifiable_by": "f"},
            "status": "draft", "version": 1}


def _ev(eid, run="r1", replay_of=None, links=None):
    from rsi.knowledge.concepts import EvidenceLink
    if links is None:
        links = [{"type": "supports", "to": "q0-h0", "strength": 0.5,
                  "confidence": 0.5}]
    return MachineEvidence(
        id=eid, epoch="e001",
        run_result={"run_id": run, "env": "t", "status": "keep",
                    "failure_mode": None,
                    "created": "2026-01-01T00:00:00+00:00", "duration_s": 1,
                    "hypothesis": "q0-h0", "variable_changed": "solver.LR",
                    "metric": {"name": "m", "direction": "min", "value": 1.0,
                               "baseline": 2.0, "delta": 1.0},
                    "benchmark_ref": {"name": "b", "epoch": "e001",
                                      "task_version": "v1"}},
        links=[EvidenceLink(**l) for l in links],
        run={"revision": run, "source_hash": "ab"},
        replay={"replayable": True, "replay_of": replay_of}).to_dict()


def test_push_run_duplicate_id_is_replay_not_integrity_error(tmp_path):
    kb = _kb(tmp_path)
    kb.create_hypothesis(_h(kb))
    kb.push_run(_ev("ev1", "r1"))
    # a DIFFERENT run_id but same evidence id: replay, not a raw sqlite error
    with pytest.raises(PlatformError) as ei:
        kb.push_run(_ev("ev1", "r2"))
    assert ei.value.code == "RSI_IDEMPOTENT_REPLAY"


def test_push_run_correction_via_replay_of_allowed(tmp_path):
    kb = _kb(tmp_path)
    kb.create_hypothesis(_h(kb))
    kb.push_run(_ev("ev1", "r1"))
    # documented correction flow: new record pointing back at the prior one
    kb.push_run(_ev("ev2", "r1", replay_of="ev1"))
    assert len(kb.evidence_by_run("r1")) == 2


def test_envelope_missing_params_is_platform_error(tmp_path):
    kb = _kb(tmp_path)
    kb.create_question(_q())
    with pytest.raises(PlatformError, match="missing"):
        kb.submit({"proposal_id": "cp1", "idempotency_key": "k1",
                   "actor": "engine",
                   "targets": [{"interface": "I-Q", "object_id": "q0"}],
                   "op": "refine", "params": {}})
    with pytest.raises(PlatformError, match="missing"):
        kb.submit({"proposal_id": "cp2", "idempotency_key": "k2",
                   "actor": "engine",
                   "targets": [{"interface": "I-H", "object_id": "q0-h0"}],
                   "op": "transition", "params": {}})


def test_hypothesis_merge_dispatch(tmp_path):
    kb = _kb(tmp_path)
    h = _h(kb)  # creates q0 once
    kb.create_hypothesis(h)
    kb.create_hypothesis({**h, "id": "q0-h1"})
    kb.transition_hypothesis("q0-h0", "active")
    out = kb.submit({"proposal_id": "cp3", "idempotency_key": "k3",
                     "actor": "engine",
                     "targets": [{"interface": "I-H", "object_id": "q0-h0"}],
                     "op": "merge", "params": {"into": "q0-h1"}})
    assert out["hypothesis"]["status"] == "merged"
    survivor = kb.get_hypothesis("q0-h1")
    mounts = {m["object_id"] for m in survivor["derived_from"]}
    assert mounts == {"q0"}
    with pytest.raises(PlatformError, match="itself"):
        kb.merge_hypotheses("q0-h1", "q0-h1")


def test_attach_evidence_dangling_ref_rejected(tmp_path):
    kb = _kb(tmp_path)
    kb.create_question(_q())
    with pytest.raises(PlatformError, match="not found"):
        kb.attach_evidence("q0", ["ghost-ev"])


def test_spawn_dangling_source_rejected(tmp_path):
    kb = _kb(tmp_path)
    kb.create_question(_q())
    with pytest.raises(PlatformError, match="not found"):
        kb.spawn_question("q0", {**_q(), "id": "q1"},
                          source={"evidence_id": "ghost-ev"})


def test_missing_testability_whole_key_maps_correctly(tmp_path):
    kb = _kb(tmp_path)
    h = _h(kb)
    del h["testability"]
    with pytest.raises(PlatformError) as ei:
        kb.create_hypothesis(h)
    assert ei.value.code == "RSI_MISSING_TESTABILITY"


def test_zero_strength_link_contributes_zero(tmp_path):
    kb = _kb(tmp_path)
    kb.create_hypothesis(_h(kb))
    links = [{"type": "supports", "to": "q0-h0", "strength": 0.0,
              "confidence": 1.0}]
    kb.push_run(_ev("ev1", "r1", links=links))
    out = kb.update_confidence("q0-h0")
    assert out["confidence"] == 0.5  # zero weight moves nothing


def test_confidence_not_truncated_at_50(tmp_path):
    kb = _kb(tmp_path)
    kb.create_hypothesis(_h(kb))
    for i in range(60):
        kb.push_run(_ev(f"ev{i}", f"r{i}"))
    out = kb.update_confidence("q0-h0")
    assert out["evidence_counts"]["support"] == 60


def test_question_self_merge_rejected(tmp_path):
    kb = _kb(tmp_path)
    kb.create_question(_q())
    with pytest.raises(PlatformError, match="itself"):
        kb.merge_questions("q0", "q0")


# -- epoch manager robustness (finding B7) --------------------------------------

def test_epoch_state_corrupt_file_clear_error(tmp_path):
    p = tmp_path / "epochs.json"
    p.write_text("{not json")
    with pytest.raises(EpochStateError):
        EpochManager(p)
    p.write_text(json.dumps({"current": None}))  # missing epochs key
    with pytest.raises(EpochStateError):
        EpochManager(p)


def test_epoch_ids_monotonic_from_hand_edited_state(tmp_path):
    p = tmp_path / "epochs.json"
    p.write_text(json.dumps({
        "current": {"name": "b", "epoch": "e005", "task_version": "v1"},
        "epochs": [{"name": "b", "epoch": "e005", "task_version": "v1",
                    "seed": 1}]}))
    m = EpochManager(p)
    ref = m.upgrade(_bench())
    assert ref.epoch == "e006"  # not len+1 == e002


def _bench(seed=7):
    from rsi.evaluation.benchmark import Benchmark
    return Benchmark(name="b", task="t", metric_name="m",
                     metric_direction="min", seed=seed)


# -- engine: epoch-tag/seed consistency after a forced upgrade (finding A1) ----

class _AlwaysUpgrade(Diagnostician):
    """Test diagnostician: forces a benchmark upgrade on the first middle
    tick, then behaves like 'continue'."""

    def __init__(self):
        self._fired = False

    def diagnose(self, history, epoch, question_id=""):
        from rsi.evaluation.diagnostics import CONTINUE, UPGRADE_BENCHMARK
        if not self._fired and len(history) >= 2:
            self._fired = True
            return DiagnosisReport(question_id=question_id, epoch=epoch,
                                   n_runs=len(history), saturation=0.1,
                                   noise_floor=0.0, discrimination=0.0,
                                   recommendation=UPGRADE_BENCHMARK)
        return DiagnosisReport(question_id=question_id, epoch=epoch,
                               n_runs=len(history), saturation=0.1,
                               noise_floor=0.0, discrimination=0.0,
                               recommendation=CONTINUE)


def test_epoch_upgrade_keeps_tags_and_seeds_consistent(tmp_path):
    task = tmp_path / "task" / "quad-optim"
    shutil.copytree(REPO / "tasks" / "quad-optim", task)
    engine = build_default_engine(EngineConfig(
        task_dir=str(task), out_dir=str(tmp_path / "out"), cycles=8,
        middle_every=3, seed=7))
    engine.diagnostician = _AlwaysUpgrade()
    engine.run("Minimize final_loss")

    epochs = {e["epoch"]: e["seed"] for e in engine.epochs.epochs}
    assert set(epochs) == {"e001", "e002"}
    runs = [json.loads(l) for l in
            (tmp_path / "out" / "state" / "runs.jsonl").read_text()
            .splitlines()]
    # every run's epoch tag must match the seed that was actually injected
    # (recorded by the task itself in observations)
    for r in runs:
        assert r["epoch"] in epochs
    tagged = {r["epoch"] for r in runs}
    assert "e002" in tagged  # the upgrade happened and later runs re-tagged
    for r in runs:
        if r["epoch"] == "e002":
            assert r["metric"]["value"] is not None


# -- learning structure: hypothesis succession within a question ---------------
# Regression for the "quietly not learning" failure: a hypothesis terminal at
# 2 supports used to END its question (8 questions in 24 cycles, none explored
# deeper than 3 runs). The question must live on under successor hypotheses
# and only be abandoned on saturation/exhaustion.

def test_question_survives_hypothesis_resolution(tmp_path):
    task = tmp_path / "task" / "quad-optim"
    shutil.copytree(REPO / "tasks" / "quad-optim", task)
    engine = build_default_engine(EngineConfig(
        task_dir=str(task), out_dir=str(tmp_path / "out"), cycles=12,
        middle_every=3, seed=7))
    engine.run("Minimize final_loss")
    runs = [json.loads(l) for l in
            (tmp_path / "out" / "state" / "runs.jsonl").read_text()
            .splitlines()]
    # q0 must NOT be abandoned merely because its hypotheses resolve: with a
    # 30+-candidate grid, 12 cycles cannot exhaust it, so the large majority
    # of runs belong to q0
    q0 = [r for r in runs if r["question_id"] == "quad-optim-q0"]
    assert len(q0) >= 10
    # and the KB shows successor hypotheses mounted on the same question
    import sqlite3
    db = sqlite3.connect(tmp_path / "out" / "kb" / "hypotheses.sqlite")
    hids = [json.loads(r[0])["id"] for r in
            db.execute("SELECT json FROM hypotheses ORDER BY rowid")]
    assert sum(1 for h in hids if h.startswith("quad-optim-q0")) >= 2


def test_maturity_guard_does_not_erase_saturation(tmp_path):
    # an immature diagnosis (insufficient runs) must not write saturation 0
    # over a previously stored value
    from rsi.evaluation.diagnostics import NaiveDiagnostician
    diag = NaiveDiagnostician().diagnose([], "e001", "q0")
    assert diag.mature is False
