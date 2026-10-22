
import pytest

from rsi.knowledge.store import EvidenceLog, SQLiteStore


def _question(i=1, sat=0.0, status="open"):
    return {"id": f"q{i}", "statement": f"question {i}", "scope": "s",
            "benchmark_ref": {"name": "b", "epoch": "e001"},
            "status": status, "version": 1, "saturation": sat,
            "uncertainty": [], "edges": []}


def test_put_get_version_bump(tmp_path):
    s = SQLiteStore(tmp_path / "q.sqlite", "question")
    q = s.put(_question())
    assert q["version"] == 1
    q["status"] = "refined"
    q2 = s.put(q, bump=True)
    assert q2["version"] == 2
    assert s.get("q1")["status"] == "refined"


def test_put_no_bump_keeps_version(tmp_path):
    s = SQLiteStore(tmp_path / "q.sqlite", "question")
    q = s.put(_question())
    q["confidence_hint"] = "derived"
    q2 = s.put(q)  # derived update: no bump (source rule)
    assert q2["version"] == 1


def test_fts_or_like_search(tmp_path):
    s = SQLiteStore(tmp_path / "q.sqlite", "question")
    s.put(_question(1))
    s.put(_question(2))
    hits = s.search("question 2")
    assert len(hits) == 1 and hits[0]["id"] == "q2"
    assert s.search("nothing matches this") == []


def test_unknown_kind_rejected(tmp_path):
    with pytest.raises(ValueError):
        SQLiteStore(tmp_path / "x.sqlite", "bogus")


def _evidence(i=1, run_id=None):
    return {"id": f"ev{i}", "evidence_kind": "machine", "epoch": "e001",
            "run_result": {"run_id": run_id or f"r{i}"},
            "links": [{"type": "supports", "to": "h1", "strength": 0.5,
                       "confidence": 0.5}],
            "replay": {"replayable": True, "replay_of": None}}


def test_evidence_append_only_by_run(tmp_path):
    e = EvidenceLog(tmp_path / "e.sqlite")
    e.append(_evidence(1, "r1"))
    e.append(_evidence(2, "r1"))  # correction: same run, new record
    assert [r["id"] for r in e.by_run("r1")] == ["ev1", "ev2"]
    assert e.get("ev1")["epoch"] == "e001"


def test_evidence_by_target(tmp_path):
    e = EvidenceLog(tmp_path / "e.sqlite")
    e.append(_evidence(1))
    e.append(_evidence(2))
    hits = e.by_target("h1")
    assert len(hits) == 2
    refute = e.by_target("h1", direction="refutes")
    assert refute == []


def test_evidence_poll_cursor(tmp_path):
    e = EvidenceLog(tmp_path / "e.sqlite")
    e.append(_evidence(1))
    e.append(_evidence(2))
    items, cursor = e.poll(0)
    assert [i["evidence_id"] for i in items] == ["ev1", "ev2"]
    assert cursor == 2
    again, cursor2 = e.poll(cursor)
    assert again == [] and cursor2 == 2
