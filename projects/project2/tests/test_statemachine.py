from rsi.knowledge.statemachine import (
    HYPOTHESIS_EVIDENCE_REQUIRED, HYPOTHESIS_TRANSITIONS,
    QUESTION_EVIDENCE_REQUIRED, QUESTION_TRANSITIONS, legal)


def test_question_legal_transitions():
    assert legal(QUESTION_TRANSITIONS, "open", "answered")
    assert legal(QUESTION_TRANSITIONS, "open", "refined")
    assert legal(QUESTION_TRANSITIONS, "refined", "open")
    assert not legal(QUESTION_TRANSITIONS, "merged", "open")
    assert not legal(QUESTION_TRANSITIONS, "open", "draft")


def test_question_evidence_required_set():
    assert QUESTION_EVIDENCE_REQUIRED == {"partially_answered", "answered"}


def test_hypothesis_legal_transitions():
    assert legal(HYPOTHESIS_TRANSITIONS, "draft", "active")
    assert legal(HYPOTHESIS_TRANSITIONS, "active", "supported")
    assert legal(HYPOTHESIS_TRANSITIONS, "active", "refuted")
    assert not legal(HYPOTHESIS_TRANSITIONS, "supported", "active")
    assert not legal(HYPOTHESIS_TRANSITIONS, "draft", "supported")


def test_hypothesis_evidence_required_set():
    assert HYPOTHESIS_EVIDENCE_REQUIRED == {"supported", "refuted"}
