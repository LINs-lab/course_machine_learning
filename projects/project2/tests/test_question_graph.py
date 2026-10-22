import pytest

from rsi.knowledge import question_graph
from rsi.knowledge.errors import PlatformError
from rsi.knowledge.store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(tmp_path / "q.sqlite", "question")


def _q(i, **kw):
    base = {"id": f"q{i}", "statement": f"statement {i}", "scope": "sc",
            "benchmark_ref": {"name": "b", "epoch": "e001"},
            "status": "open", "version": 1, "saturation": 0.0,
            "uncertainty": [], "edges": []}
    base.update(kw)
    return base


def test_create_and_duplicate_rejected(store):
    question_graph.create_question(store, _q(1))
    with pytest.raises(PlatformError, match="already exists"):
        question_graph.create_question(store, _q(1))


def test_spawn_adds_typed_edge(store):
    question_graph.create_question(store, _q(1))
    child = question_graph.spawn_question(
        store, "q1", _q(2), source={"evidence_id": "ev9"})
    parent = store.get("q1")
    assert parent["edges"] == [{"type": "spawns", "to": "q2",
                                "source": "ev9", "confidence": None}]
    assert child["status"] == "open"


def test_spawn_requires_source(store):
    question_graph.create_question(store, _q(1))
    with pytest.raises(PlatformError, match="source"):
        question_graph.spawn_question(store, "q1", _q(2), source={})


def test_refine_decomposes(store):
    question_graph.create_question(store, _q(1))
    q = question_graph.refine_question(store, "q1", [_q(2), _q(3)])
    assert q["status"] == "refined"
    assert {e["to"] for e in q["edges"]} == {"q2", "q3"}
    assert store.get("q2")["status"] == "open"


def test_merge_bidirectional_edges(store):
    question_graph.create_question(store, _q(1))
    question_graph.create_question(store, _q(2))
    question_graph.merge_questions(store, "q1", "q2")
    assert store.get("q1")["status"] == "merged"
    assert store.get("q2")["edges"][0]["to"] == "q1"


def test_transition_requires_evidence_for_answer(store):
    question_graph.create_question(store, _q(1))
    with pytest.raises(PlatformError, match="conversion_event"):
        question_graph.transition_question(store, "q1", "answered")
    q = question_graph.transition_question(
        store, "q1", "partially_answered",
        conversion_event={"evidence_refs": ["ev1"]})
    assert q["status"] == "partially_answered"


def test_illegal_transition_rejected(store):
    question_graph.create_question(store, _q(1))
    with pytest.raises(PlatformError, match="not a legal transition"):
        question_graph.transition_question(store, "q1", "supported")


def test_base_version_conflict(store):
    question_graph.create_question(store, _q(1))
    with pytest.raises(PlatformError) as ei:
        question_graph.update_question(store, "q1", {"saturation": 0.5},
                                       base_version=99)
    assert ei.value.code == "RSI_VERSION_CONFLICT" and ei.value.retryable


def test_update_field_and_validation(store):
    question_graph.create_question(store, _q(1))
    with pytest.raises(PlatformError, match="non-updatable"):
        question_graph.update_question(store, "q1", {"status": "answered"})
    q = question_graph.update_question(store, "q1",
                                       {"saturation": 0.4,
                                        "best_result": {"evidence_id": "e",
                                                        "score": 1.0}})
    assert q["version"] == 2 and q["saturation"] == 0.4


def test_frontier_orders_by_saturation(store):
    for i, sat in [(1, 0.9), (2, 0.1), (3, 0.5)]:
        question_graph.create_question(store, _q(i, saturation=sat))
    question_graph.transition_question(store, "q3", "retired")
    ids = [q["id"] for q in question_graph.frontier(store)]
    assert ids == ["q2", "q1"]


def test_frontier_max_saturation_filter(store):
    question_graph.create_question(store, _q(1, saturation=0.9))
    question_graph.create_question(store, _q(2, saturation=0.1))
    ids = [q["id"] for q in question_graph.frontier(store,
                                                    max_saturation=0.5)]
    assert ids == ["q2"]
