"""Contract tests: every concept dataclass serializes to a payload that
validates against its frozen YAML schema — the two definitions cannot drift.

Port-shaped from project_collab_auto_research_shared_kb/tests/
test_contract_schemas.py (golden-example validation idea).
"""
import jsonschema
import pytest

from rsi.knowledge import schemas
from rsi.knowledge.concepts import (
    ChangeProposal, Hypothesis, Insight, KnowledgeNode, MachineEvidence,
    Question,
)

BENCH = {"name": "quad-optim", "epoch": "e001"}


def test_all_interface_schemas_load():
    for key in schemas.INTERFACES + ("change_proposal", "ik_theory"):
        assert "$schema" in schemas.load_schema(key)


def test_question_roundtrip_validates():
    q = Question(id="q1", statement="s", scope="sc", benchmark_ref=BENCH)
    jsonschema.validate(q.to_dict(), schemas.load_schema("iq"))
    q2 = Question.from_dict(q.to_dict())
    assert q2.to_dict()["edges"] == []


def test_hypothesis_roundtrip_validates():
    h = Hypothesis(
        id="h1", statement="st",
        derived_from=[{"interface": "I-Q", "object_id": "q1"}],
        testability={"procedure": "p", "expected_evidence": "e",
                     "falsifiable_by": "f"})
    jsonschema.validate(h.to_dict(), schemas.load_schema("ih"))


def test_machine_evidence_roundtrip_validates():
    ev = MachineEvidence(
        id="ev1", epoch="e001",
        run_result={
            "run_id": "r1", "env": "quad-optim", "status": "keep",
            "failure_mode": None, "created": "2026-01-01T00:00:00+00:00",
            "duration_s": 1, "hypothesis": "h1",
            "variable_changed": "solver.LR",
            "metric": {"name": "final_loss", "direction": "min",
                       "value": 1.0, "baseline": 2.0, "delta": 1.0},
            "benchmark_ref": {"name": "quad-optim", "epoch": "e001",
                              "task_version": "v1"},
        },
        run={"revision": "r1", "source_hash": "ab" * 32})
    jsonschema.validate(ev.to_dict(), schemas.load_schema("ie"))


def test_failed_run_evidence_value_null_validates():
    ev = MachineEvidence(
        id="ev2", epoch="e001",
        run_result={
            "run_id": "r2", "env": "quad-optim", "status": "fail",
            "failure_mode": "timeout",
            "created": "2026-01-01T00:00:00+00:00",
            "duration_s": 9, "hypothesis": "h1",
            "variable_changed": "solver.LR",
            "metric": {"name": "final_loss", "direction": "min",
                       "value": None, "baseline": 2.0, "delta": None},
            "benchmark_ref": {"name": "quad-optim", "epoch": "e001",
                              "task_version": "v1"},
        },
        failure={"class": "timeout", "diagnosis": "timeout"},
        run={"revision": "r2", "source_hash": "cd" * 32})
    jsonschema.validate(ev.to_dict(), schemas.load_schema("ie"))


def test_insight_roundtrip_validates():
    i = Insight(insight_id="i1", statement="st",
                derived_from=[{"interface": "I-Q", "object_id": "q1"}],
                applicability=["quad-optim"])
    jsonschema.validate(i.to_dict(), schemas.load_schema("is"))


def test_knowledge_node_roundtrip_validates():
    n = KnowledgeNode(node_id="theory:opt", statement="st", domain="opt")
    jsonschema.validate(n.to_dict(), schemas.load_schema("ik_theory"))


def test_change_proposal_validates():
    p = ChangeProposal(
        proposal_id="cp1", actor="engine", op="update",
        idempotency_key="k1",
        targets=[{"interface": "I-Q", "object_id": "q1"}],
        params={"fields": {"saturation": 0.5}})
    jsonschema.validate(p.to_dict(), schemas.load_schema("change_proposal"))


def test_bad_question_rejected_by_schema():
    q = Question(id="Q1!", statement="s", scope="sc", benchmark_ref=BENCH,
                 status="not-a-status")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(q.to_dict(), schemas.load_schema("iq"))
