import pytest

from rsi.knowledge.concepts import EvidenceLink, Hypothesis, MachineEvidence
from rsi.knowledge.errors import PlatformError
from rsi.knowledge.platform import KnowledgePlatform


@pytest.fixture
def kb(tmp_path):
    return KnowledgePlatform(tmp_path / "kb")


def _q(i=0):
    return {"id": f"q{i}", "statement": f"st {i}", "scope": "sc",
            "benchmark_ref": {"name": "b", "epoch": "e001"},
            "status": "open", "version": 1, "saturation": 0.0,
            "uncertainty": [], "edges": []}


def _h(kb, i=0, status="draft"):
    kb.create_question(_q(i))
    return {"id": f"q{i}-h0", "statement": "hyp",
            "derived_from": [{"interface": "I-Q", "object_id": f"q{i}"}],
            "testability": {"procedure": "p", "expected_evidence": "e",
                            "falsifiable_by": "f"},
            "status": status, "version": 1}


def _ev(i, run, to="q0-h0", ltype="supports", value=1.0, strength=0.8,
        confidence=1.0):
    return MachineEvidence(
        id=f"ev{i}", epoch="e001",
        run_result={
            "run_id": run, "env": "t", "status": "keep", "failure_mode": None,
            "created": "2026-01-01T00:00:00+00:00", "duration_s": 1,
            "hypothesis": to, "variable_changed": "solver.LR",
            "metric": {"name": "m", "direction": "min", "value": value,
                       "baseline": 2.0, "delta": 1.0},
            "benchmark_ref": {"name": "b", "epoch": "e001",
                              "task_version": "v1"}},
        links=[EvidenceLink(ltype, to, strength, confidence)],
        run={"revision": run, "source_hash": "ab"}).to_dict()


def test_hypothesis_needs_mount(kb):
    h = _h(kb)
    h["derived_from"] = [{"interface": "I-Q", "object_id": "ghost"}]
    with pytest.raises(PlatformError) as ei:
        kb.create_hypothesis(h)
    assert ei.value.code == "RSI_DANGLING_REF"


def test_hypothesis_testability_all_or_nothing(kb):
    h = _h(kb)
    h["testability"] = {"procedure": "p"}  # missing two of three
    with pytest.raises(PlatformError):
        kb.create_hypothesis(h)


def test_confidence_aggregation_math(kb):
    kb.create_hypothesis(_h(kb))
    # prior 0.5; supports w=0.8*1.0 then w=1.0*1.0 → net +1.8 → clamp 1.0
    kb.push_run(_ev(1, "r1", ltype="supports", value=1.0,
                    strength=0.8, confidence=1.0))
    kb.push_run(_ev(2, "r2", ltype="supports", value=0.5,
                    strength=1.0, confidence=1.0))
    out = kb.update_confidence("q0-h0")
    assert out["confidence"] == 1.0  # 0.5 + 1.8 clamped
    assert out["evidence_counts"] == {"support": 2, "refute": 0,
                                      "inconclusive": 0}
    h = kb.get_hypothesis("q0-h0")
    assert h["confidence_algorithm"].startswith("net-strength")


def test_confidence_refutes_drop(kb, tmp_path):
    kb.create_hypothesis(_h(kb))
    kb.push_run(_ev(1, "r1", ltype="refutes", value=3.0,
                    strength=0.4, confidence=0.5))
    out = kb.update_confidence("q0-h0")
    assert abs(out["confidence"] - (0.5 - 0.2)) < 1e-9


def test_push_run_idempotent_replay(kb):
    kb.create_hypothesis(_h(kb))
    kb.push_run(_ev(1, "r1"))
    with pytest.raises(PlatformError) as ei:
        kb.push_run(_ev(1, "r1"))
    assert ei.value.code == "RSI_IDEMPOTENT_REPLAY"
    assert ei.value.result["evidence_id"] == "ev1"


def test_push_run_dangling_link_rejected(kb):
    kb.create_hypothesis(_h(kb))
    with pytest.raises(PlatformError, match="link target"):
        kb.push_run(_ev(1, "r1", to="ghost"))


def test_supported_needs_two_supports(kb):
    kb.create_hypothesis(_h(kb))
    kb.transition_hypothesis("q0-h0", "active")
    kb.push_run(_ev(1, "r1"))
    kb.update_confidence("q0-h0")
    with pytest.raises(PlatformError, match="support"):
        kb.transition_hypothesis("q0-h0", "supported")
    kb.push_run(_ev(2, "r2"))
    kb.update_confidence("q0-h0")
    h = kb.transition_hypothesis("q0-h0", "supported")
    assert h["status"] == "supported"
    # terminal: no further transitions
    with pytest.raises(PlatformError):
        kb.transition_hypothesis("q0-h0", "active")


def test_submit_envelope_idempotency(kb):
    kb.create_question(_q(0))
    prop = {
        "proposal_id": "cp1", "idempotency_key": "k1", "actor": "engine",
        "targets": [{"interface": "I-Q", "object_id": "q0"}],
        "op": "update", "params": {"fields": {"saturation": 0.3}},
    }
    out = kb.submit(prop)
    assert out["question"]["saturation"] == 0.3
    with pytest.raises(PlatformError) as ei:
        kb.submit(prop)
    assert ei.value.code == "RSI_IDEMPOTENT_REPLAY"
    # base_version conflict via envelope
    with pytest.raises(PlatformError) as e2:
        kb.submit({**prop, "proposal_id": "cp2", "idempotency_key": "k2",
                   "targets": [{"interface": "I-Q", "object_id": "q0",
                                "base_version": 1}]})
    assert e2.value.code == "RSI_VERSION_CONFLICT"


def test_insights_mount_and_query(kb):
    kb.create_question(_q(0))
    kb.add_insight({"insight_id": "i1", "statement": "momentum helps",
                    "derived_from": [{"interface": "I-Q", "object_id": "q0"}],
                    "source": "experiential", "applicability": ["quad-optim"],
                    "status": "active"})
    hits = kb.insights_for("I-Q", "q0")
    assert len(hits) == 1
    # non-empty applicability: the rejection below must come from the
    # dangling mount, not from schema validation
    with pytest.raises(PlatformError, match="mount"):
        kb.add_insight({"insight_id": "i2", "statement": "x",
                        "derived_from": [{"interface": "I-Q",
                                          "object_id": "ghost"}],
                        "source": "experiential",
                        "applicability": ["quad-optim"],
                        "status": "active"})


def test_theory_nodes(kb):
    kb.put_node({"node_id": "theory:opt", "statement": "preconditioning",
                 "status": "established", "domain": "opt", "branch": "core",
                 "level": 0, "version": 1})
    assert kb.get_node("theory:opt")["domain"] == "opt"
    # _h() creates q0; the hypothesis then mounts on I-K instead
    h = Hypothesis.from_dict(_h(kb))
    h.derived_from = [{"interface": "I-K", "object_id": "theory:opt"}]
    kb.create_hypothesis(h.to_dict())
    # a dangling I-K mount is refused
    import dataclasses
    h2 = dataclasses.replace(h, id="q0-h1",
                             derived_from=[{"interface": "I-K",
                                            "object_id": "theory:ghost"}])
    with pytest.raises(PlatformError):
        kb.create_hypothesis(h2.to_dict())
