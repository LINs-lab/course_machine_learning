# Ported (semantics preserved) from
# project_collab_auto_research_shared_kb/server/question/server.py (_apply op
# bodies + QuestionStore.frontier ordering), with the cross-server MCP
# existence checks replaced by direct store lookups (in-process platform) and
# one additive op: `update` (in-place field writes for the middle loop).
"""Research Question Graph operations: create / spawn / refine / merge /
transition / attach_evidence / update, plus frontier selection.

Semantics preserved from the frozen v0.2 contract:

- ``spawn`` — a new question born from an evidence/hypothesis source; the
  PARENT gains the typed edge ``{type: spawns, to: <child>, source: <E/H id>}``
  (the iq.yaml documented semantics for spawns edges).
- ``refine`` — DECOMPOSITION: replace a question with child sub-questions;
  parent status → ``refined`` and gains ``{type: refines, to: child}`` edges.
- ``merge`` — atomic dual-direction ``merged_into`` edges; both nodes bump.
- ``transition`` — statemachine-gated; ``partially_answered``/``answered``
  require a conversion_event with evidence_refs.
- ``update`` (base extension) — in-place field writes (saturation,
  best_result, uncertainty, benchmark_ref rebind, statement); version bump.
"""
from __future__ import annotations

import jsonschema

from rsi.knowledge import schemas, statemachine
from rsi.knowledge.errors import PlatformError
from rsi.knowledge.store import SQLiteStore

UPDATABLE_FIELDS = ("statement", "scope", "benchmark_ref", "uncertainty",
                    "saturation", "best_result")


def _validate_question(node: dict) -> None:
    try:
        jsonschema.validate(node, schemas.load_schema("iq"))
    except jsonschema.ValidationError as e:
        raise PlatformError("RSI_INVALID_TRANSITION",
                            f"question schema violation: {e.message}") from e


def _check_version(q: dict, base_version: int | None) -> None:
    if base_version is not None and base_version != q["version"]:
        raise PlatformError(
            "RSI_VERSION_CONFLICT",
            f"base_version {base_version} != current {q['version']}",
            retryable=True)


def create_question(store: SQLiteStore, question: dict) -> dict:
    """Validate against I-Q and insert (id must be fresh)."""
    _validate_question(question)
    if store.get(question["id"]):
        raise PlatformError("RSI_INVALID_TRANSITION",
                            f"{question['id']} already exists")
    return store.put(question)


def spawn_question(store: SQLiteStore, parent_id: str, child: dict,
                   source: dict) -> dict:
    """⑦ edge extrapolation: parent --spawns--> child, edge source = the
    evidence/hypothesis that motivated the spawn."""
    src_id = source.get("evidence_id") or source.get("hypothesis_id")
    if not src_id:
        raise PlatformError("RSI_DANGLING_REF", "spawn requires a source ref")
    _validate_question(child)
    if store.get(child["id"]):
        raise PlatformError("RSI_INVALID_TRANSITION",
                            f"{child['id']} already exists")
    parent = store.get(parent_id)
    if not parent:
        raise PlatformError("RSI_DANGLING_REF",
                            f"parent question {parent_id} not found")
    parent.setdefault("edges", []).append({
        "type": "spawns", "to": child["id"], "source": src_id,
        "confidence": source.get("confidence"),
    })
    # one transaction: the parent edge and the child must land together
    store.put(parent, bump=True, commit=False)
    result = store.put(child, commit=False)
    store.commit()
    return result


def refine_question(store: SQLiteStore, question_id: str,
                    children: list[dict], base_version: int | None = None,
                    ) -> dict:
    """Decompose: children must all validate; parent → status refined with
    refines edges. Not atomic-across-processes here (single-writer engine)."""
    q = store.get(question_id)
    if not q:
        raise PlatformError("RSI_DANGLING_REF",
                            f"question {question_id} not found")
    _check_version(q, base_version)
    # refine is a status change (→ refined): it must obey the frozen
    # transition table like any other (a retired question must stay retired)
    if not statemachine.legal(statemachine.QUESTION_TRANSITIONS,
                              q["status"], "refined"):
        raise PlatformError(
            "RSI_INVALID_TRANSITION",
            f"{q['status']} -> refined is not a legal transition")
    if not children:
        raise PlatformError("RSI_INVALID_TRANSITION",
                            "refine requires at least one child question")
    for child in children:
        _validate_question(child)
        if store.get(child["id"]):
            raise PlatformError("RSI_INVALID_TRANSITION",
                                f"{child['id']} already exists")
    for child in children:
        store.put(child, commit=False)
    q["status"] = "refined"
    q.setdefault("edges", []).extend(
        {"type": "refines", "to": c["id"]} for c in children)
    result = store.put(q, bump=True, commit=False)
    store.commit()
    return result


def merge_questions(store: SQLiteStore, question_id: str, into: str,
                    base_version: int | None = None) -> dict:
    q = store.get(question_id)
    other = store.get(into)
    if not q:
        raise PlatformError("RSI_DANGLING_REF",
                            f"question {question_id} not found")
    if not other:
        raise PlatformError("RSI_DANGLING_REF",
                            f"merge target {into} not found")
    if question_id == into:
        raise PlatformError("RSI_INVALID_TRANSITION",
                            "cannot merge a question into itself")
    _check_version(q, base_version)
    # merge is a status change (→ merged, terminal): obey the frozen table
    if not statemachine.legal(statemachine.QUESTION_TRANSITIONS,
                              q["status"], "merged"):
        raise PlatformError(
            "RSI_INVALID_TRANSITION",
            f"{q['status']} -> merged is not a legal transition")
    q["status"] = "merged"
    q.setdefault("edges", []).append({"type": "merged_into", "to": into})
    other.setdefault("edges", []).append({"type": "merged_into", "to": q["id"]})
    store.put(other, bump=True, commit=False)
    result = store.put(q, bump=True, commit=False)
    store.commit()
    return result


def transition_question(store: SQLiteStore, question_id: str, to: str,
                        conversion_event: dict | None = None,
                        evidence_refs: list[str] | None = None,
                        base_version: int | None = None) -> dict:
    q = store.get(question_id)
    if not q:
        raise PlatformError("RSI_DANGLING_REF",
                            f"question {question_id} not found")
    _check_version(q, base_version)
    if not statemachine.legal(statemachine.QUESTION_TRANSITIONS, q["status"],
                              to):
        raise PlatformError("RSI_INVALID_TRANSITION",
                            f"{q['status']} -> {to} is not a legal transition")
    if to in statemachine.QUESTION_EVIDENCE_REQUIRED:
        ev = ((conversion_event or {}).get("evidence_refs")
              or evidence_refs or [])
        if not ev:
            raise PlatformError(
                "RSI_INVALID_TRANSITION",
                f"{to} requires conversion_event with evidence_refs")
        # the justifying evidence must be PERSISTED, not just gate-checked:
        # fold bare evidence_refs into the conversion event and onto the
        # node's evidence chain (an "answered" question with an empty chain
        # is an audit hole)
        if conversion_event is not None:
            conversion_event = {**conversion_event, "evidence_refs": list(ev)}
        else:
            conversion_event = {"evidence_refs": list(ev)}
        chain = q.setdefault("evidence_refs", [])
        for ref in ev:
            if ref not in chain:
                chain.append(ref)
    q["status"] = to
    if conversion_event is not None:
        q.setdefault("conversion_events", []).append(conversion_event)
    return store.put(q, bump=True)


def attach_evidence(store: SQLiteStore, question_id: str,
                    evidence_refs: list[str], base_version: int | None = None,
                    ) -> dict:
    q = store.get(question_id)
    if not q:
        raise PlatformError("RSI_DANGLING_REF",
                            f"question {question_id} not found")
    _check_version(q, base_version)
    chain = q.setdefault("evidence_refs", [])
    for ref in evidence_refs:
        if ref not in chain:      # idempotent: a replayed attach adds nothing
            chain.append(ref)
    return store.put(q, bump=True)


def update_question(store: SQLiteStore, question_id: str, fields: dict,
                    base_version: int | None = None) -> dict:
    """Base-system extension: in-place field updates (see module docstring).
    Status is NOT updatable here — use transition_question."""
    q = store.get(question_id)
    if not q:
        raise PlatformError("RSI_DANGLING_REF",
                            f"question {question_id} not found")
    _check_version(q, base_version)
    if not fields:
        raise PlatformError("RSI_INVALID_TRANSITION",
                            "update requires at least one field")
    unknown = set(fields) - set(UPDATABLE_FIELDS)
    if unknown:
        raise PlatformError("RSI_INVALID_TRANSITION",
                            f"non-updatable fields {sorted(unknown)}; "
                            f"updatable: {UPDATABLE_FIELDS}")
    q.update(fields)
    _validate_question({k: v for k, v in q.items()
                        if k not in ("evidence_refs", "conversion_events")})
    return store.put(q, bump=True)


def frontier(store: SQLiteStore, max_saturation: float = 1.0,
             status: str = "open", limit: int = 50) -> list[dict]:
    """Open, least-saturated questions first (the ① selection anchor)."""
    out = [q for q in store.all()
           if q.get("status") == status
           and float(q.get("saturation") or 0.0) <= max_saturation]
    out.sort(key=lambda q: (float(q.get("saturation") or 0.0), q["id"]))
    return out[:limit]
