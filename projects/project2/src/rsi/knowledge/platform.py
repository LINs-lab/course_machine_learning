# Ported (semantics preserved, in-process) from
# project_collab_auto_research_shared_kb/server/{hypothesis,evidence,insight,
# knowledge}/server.py + server/common/gates.py idempotency: the five MCP
# stdio servers collapse into one facade with direct method calls. Dropped:
# MCP transport, L0/L1/L2 review queue (auto-apply, documented divergence),
# actor enforcement on push (single-writer engine), cross-server polling.
"""KnowledgePlatform: one facade over the entity stores + evidence log.

Implements the 7-step main chain of the engine manual, in-process:

① question create/frontier (question_graph)   ② hypothesis create (mounted,
testability-validated)                        ⑤ push_run (append-only,
idempotent by run_id)                          ⑥ update_confidence (net =
Σsupports − Σrefutes, no version bump) + count-gated transitions
⑦ spawn (edge extrapolation → back to ①).

All write methods validate payloads against the frozen YAML schemas and speak
DICTS (the contract boundary); typed helpers in rsi.knowledge.concepts are
the ergonomic layer above.
"""
from __future__ import annotations

import json
import jsonschema
import sqlite3
from pathlib import Path

from rsi.knowledge import question_graph, schemas, statemachine
from rsi.knowledge.errors import PlatformError
from rsi.knowledge.store import EvidenceLog, SQLiteStore

# Evidence-count thresholds for hypothesis transitions (source server
# constants SUPPORT_THRESHOLD=2 / REFUTE_THRESHOLD=1).
SUPPORT_THRESHOLD = 2
REFUTE_THRESHOLD = 1
CONFIDENCE_ALGORITHM = "net-strength-v0.2"


class KnowledgePlatform:
    def __init__(self, kb_dir: str | Path):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.questions = SQLiteStore(self.kb_dir / "questions.sqlite", "question")
        self.hypotheses = SQLiteStore(self.kb_dir / "hypotheses.sqlite",
                                      "hypothesis")
        self.insights = SQLiteStore(self.kb_dir / "insights.sqlite", "insight")
        self.nodes = SQLiteStore(self.kb_dir / "nodes.sqlite", "node")
        self.evidence = EvidenceLog(self.kb_dir / "evidence.sqlite")
        # ChangeProposal idempotency keys are PERSISTED: an in-memory dict
        # would silently re-apply replayed envelopes after a process restart
        self._idem = sqlite3.connect(str(self.kb_dir / "meta.sqlite"),
                                     check_same_thread=False)
        self._idem.execute(
            "CREATE TABLE IF NOT EXISTS idempotency "
            "(key TEXT PRIMARY KEY, result TEXT NOT NULL)")
        self._idem.commit()

    # -- ① questions -------------------------------------------------------

    def get_question(self, qid: str) -> dict | None:
        return self.questions.get(qid)

    def frontier(self, max_saturation: float = 1.0, status: str = "open",
                 limit: int = 50) -> list[dict]:
        return question_graph.frontier(self.questions, max_saturation,
                                       status, limit)

    def create_question(self, question: dict, idempotency_key: str = "",
                        actor: str = "engine") -> dict:
        return question_graph.create_question(self.questions, question)

    def spawn_question(self, parent_id: str, child: dict, source: dict) -> dict:
        src_id = source.get("evidence_id") or source.get("hypothesis_id")
        found = (self.evidence.get(src_id) is not None
                 if source.get("evidence_id")
                 else self.hypotheses.get(src_id) is not None
                 if source.get("hypothesis_id") else False)
        if not found:
            raise PlatformError("RSI_DANGLING_REF",
                                f"spawn source {src_id!r} not found")
        return question_graph.spawn_question(self.questions, parent_id, child,
                                             source)

    def refine_question(self, question_id: str, children: list[dict],
                        base_version: int | None = None) -> dict:
        return question_graph.refine_question(self.questions, question_id,
                                              children, base_version)

    def merge_questions(self, question_id: str, into: str,
                        base_version: int | None = None) -> dict:
        return question_graph.merge_questions(self.questions, question_id,
                                              into, base_version)

    def transition_question(self, question_id: str, to: str,
                            conversion_event: dict | None = None,
                            evidence_refs: list[str] | None = None,
                            base_version: int | None = None) -> dict:
        # same invariant as attach_evidence: a dangling ref must never enter
        # a question's evidence chain — and an answered transition MUST NOT
        # cite evidence that does not exist
        for ref in (((conversion_event or {}).get("evidence_refs") or [])
                    + list(evidence_refs or [])):
            if self.evidence.get(ref) is None:
                raise PlatformError("RSI_DANGLING_REF",
                                    f"evidence {ref} not found")
        return question_graph.transition_question(
            self.questions, question_id, to, conversion_event, evidence_refs,
            base_version)

    def attach_evidence(self, question_id: str, evidence_refs: list[str],
                        base_version: int | None = None) -> dict:
        # existence check (source: cross-server ep lookup, here in-process):
        # a dangling ref must never enter a question's evidence chain
        for ref in evidence_refs:
            if self.evidence.get(ref) is None:
                raise PlatformError("RSI_DANGLING_REF",
                                    f"evidence {ref} not found")
        return question_graph.attach_evidence(self.questions, question_id,
                                              evidence_refs, base_version)

    def update_question(self, question_id: str, fields: dict,
                        base_version: int | None = None) -> dict:
        return question_graph.update_question(self.questions, question_id,
                                              fields, base_version)

    # -- ② hypotheses ------------------------------------------------------

    def get_hypothesis(self, hid: str) -> dict | None:
        return self.hypotheses.get(hid)

    def create_hypothesis(self, hypothesis: dict) -> dict:
        """Testability all-or-nothing + at least one live mount point."""
        # explicit testability presence check first (source behavior): a
        # wholly missing testability maps to RSI_MISSING_TESTABILITY, which
        # the jsonschema path cannot distinguish (empty absolute_path)
        t = hypothesis.get("testability") or {}
        missing = [k for k in ("procedure", "expected_evidence",
                               "falsifiable_by")
                   if not isinstance(t, dict) or not t.get(k)]
        if missing:
            raise PlatformError(
                "RSI_MISSING_TESTABILITY",
                f"testability is all-or-nothing; missing {missing}")
        try:
            jsonschema.validate(hypothesis, schemas.load_schema("ih"))
        except jsonschema.ValidationError as e:
            raise PlatformError("RSI_INVALID_TRANSITION",
                                f"hypothesis schema violation: {e.message}"
                                ) from e
        for mount in hypothesis["derived_from"]:
            iface, oid = mount["interface"], mount["object_id"]
            found = (self.questions.get(oid) if iface == "I-Q"
                     else self.nodes.get(oid) if iface == "I-K" else None)
            if found is None:
                raise PlatformError(
                    "RSI_DANGLING_REF",
                    f"mount {iface}:{oid} not found — an unmounted hypothesis "
                    f"is mechanically impossible")
        if self.hypotheses.get(hypothesis["id"]):
            raise PlatformError("RSI_INVALID_TRANSITION",
                                f"{hypothesis['id']} already exists")
        return self.hypotheses.put(hypothesis)

    def transition_hypothesis(self, hid: str, to: str,
                              base_version: int | None = None) -> dict:
        """Statemachine + evidence-count thresholds (source: supported needs
        support >= SUPPORT_THRESHOLD; refuted needs refute >= REFUTE_THRESHOLD)."""
        h = self.hypotheses.get(hid)
        if not h:
            raise PlatformError("RSI_DANGLING_REF",
                                f"hypothesis {hid} not found")
        if base_version is not None and base_version != h["version"]:
            raise PlatformError("RSI_VERSION_CONFLICT",
                                f"base_version {base_version} != current "
                                f"{h['version']}", retryable=True)
        if not statemachine.legal(statemachine.HYPOTHESIS_TRANSITIONS,
                                  h["status"], to):
            raise PlatformError("RSI_INVALID_TRANSITION",
                                f"{h['status']} -> {to} is not a legal transition")
        counts = h.get("evidence_counts") or {}
        if to == "supported" and counts.get("support", 0) < SUPPORT_THRESHOLD:
            raise PlatformError(
                "RSI_INVALID_TRANSITION",
                f"supported needs support >= {SUPPORT_THRESHOLD}, "
                f"have {counts.get('support', 0)}")
        if to == "refuted" and counts.get("refute", 0) < REFUTE_THRESHOLD:
            raise PlatformError(
                "RSI_INVALID_TRANSITION",
                f"refuted needs refute >= {REFUTE_THRESHOLD}, "
                f"have {counts.get('refute', 0)}")
        h["status"] = to
        return self.hypotheses.put(h, bump=True)

    def update_confidence(self, hid: str) -> dict:
        """⑥ Aggregate confidence from evidence links: net = Σ(strength×
        confidence | supports) − Σ(... | refutes); confidence = clamp(prior +
        net); counts recomputed. Derived update — put WITHOUT version bump."""
        h = self.hypotheses.get(hid)
        if not h:
            raise PlatformError("RSI_DANGLING_REF",
                                f"hypothesis {hid} not found")
        records = self.evidence.by_target(hid, limit=10 ** 6)
        # a record superseded by a replay_of correction is RETRACTED: it must
        # not count (the correction replaces it — counting both would let a
        # retracted refute still satisfy the refuted threshold)
        superseded = {(r.get("replay") or {}).get("replay_of")
                      for r in records}
        net = 0.0
        counts = {"support": 0, "refute": 0, "inconclusive": 0}
        for rec in records:
            if rec["id"] in superseded:
                continue
            for link in rec.get("links", []):
                if link["to"] != hid:
                    continue
                # explicit 0.0 weights are legal (schema minimum 0) and must
                # not fall back to the 0.5 default
                strength = (link["strength"] if link.get("strength")
                            is not None else 0.5)
                confidence = (link["confidence"] if link.get("confidence")
                              is not None else 0.5)
                w = strength * confidence
                counts[{"supports": "support", "refutes": "refute"}.get(
                    link["type"], "inconclusive")] += 1
                net += (w if link["type"] == "supports"
                        else -w if link["type"] == "refutes" else 0)
        conf = min(1.0, max(0.0, h.get("prior_confidence", 0.5) + net))
        h.update(confidence=conf, evidence_counts=counts,
                 confidence_algorithm=CONFIDENCE_ALGORITHM)
        self.hypotheses.put(h)  # no version bump: system-derived update
        return {"hypothesis_id": hid, "confidence": conf,
                "evidence_counts": counts}

    def merge_hypotheses(self, hid: str, into: str,
                         base_version: int | None = None) -> dict:
        """I-H merge (source hypothesis/server.py): the merged hypothesis is
        terminal; the survivor absorbs its mount points (derived_from union)."""
        h = self.hypotheses.get(hid)
        other = self.hypotheses.get(into)
        if not h:
            raise PlatformError("RSI_DANGLING_REF",
                                f"hypothesis {hid} not found")
        if not other:
            raise PlatformError("RSI_DANGLING_REF",
                                f"merge target {into} not found")
        if base_version is not None and base_version != h["version"]:
            raise PlatformError("RSI_VERSION_CONFLICT",
                                f"base_version {base_version} != current "
                                f"{h['version']}", retryable=True)
        if hid == into:
            raise PlatformError("RSI_INVALID_TRANSITION",
                                "cannot merge a hypothesis into itself")
        # merge is a status change (→ merged, terminal): obey the frozen
        # table — a supported/refuted/draft hypothesis cannot be merged
        if not statemachine.legal(statemachine.HYPOTHESIS_TRANSITIONS,
                                  h["status"], "merged"):
            raise PlatformError(
                "RSI_INVALID_TRANSITION",
                f"{h['status']} -> merged is not a legal transition")
        h["status"] = "merged"
        # mounts are (interface, object_id) pairs — I-Q:x and I-K:x are
        # DIFFERENT mounts (id patterns overlap across kinds)
        seen = {(m["interface"], m["object_id"])
                for m in other.get("derived_from", [])}
        for mount in h.get("derived_from", []):
            key = (mount["interface"], mount["object_id"])
            if key not in seen:
                other.setdefault("derived_from", []).append(mount)
                seen.add(key)
        self.hypotheses.put(other, bump=True, commit=False)
        result = self.hypotheses.put(h, bump=True, commit=False)
        self.hypotheses.commit()
        return result

    # -- ⑤ evidence --------------------------------------------------------

    def push_run(self, evidence: dict) -> str:
        """Append-only X2 write: a finished run becomes machine evidence.

        Idempotent (source gates.py replay semantics): a record whose ``id``
        or ``run_result.run_id`` already exists raises
        RSI_IDEMPOTENT_REPLAY carrying the first record — evidence-before-
        notify means a run counts once, ever. Corrections are the documented
        exception: append a NEW record with ``replay.replay_of`` pointing at
        the prior one."""
        try:
            jsonschema.validate(evidence, schemas.load_schema("ie"))
        except jsonschema.ValidationError as e:
            raise PlatformError("RSI_INVALID_TRANSITION",
                                f"evidence schema violation: {e.message}"
                                ) from e
        is_correction = bool((evidence.get("replay") or {}).get("replay_of"))
        prior = self.evidence.get(evidence["id"])
        rr = evidence.get("run_result") or {}
        if prior is None and rr.get("run_id"):
            by_run = self.evidence.by_run(rr["run_id"])
            prior = by_run[0] if by_run else None
        if prior is not None and not is_correction:
            raise PlatformError(
                "RSI_IDEMPOTENT_REPLAY",
                f"evidence {evidence['id']} / run {rr.get('run_id')} "
                f"already recorded as {prior['id']}",
                result={"evidence_id": prior["id"], "replayed": True})
        for link in evidence.get("links", []):
            if (self.hypotheses.get(link["to"]) is None
                    and self.questions.get(link["to"]) is None):
                raise PlatformError("RSI_DANGLING_REF",
                                    f"link target {link['to']} not found")
        try:
            self.evidence.append(evidence)
        except sqlite3.IntegrityError as e:  # race-free uniqueness backstop
            raise PlatformError("RSI_IDEMPOTENT_REPLAY",
                                f"duplicate evidence id {evidence['id']}"
                                ) from e
        return evidence["id"]

    def get_evidence(self, evidence_id: str) -> dict | None:
        return self.evidence.get(evidence_id)

    def evidence_by_run(self, run_id: str) -> list[dict]:
        return self.evidence.by_run(run_id)

    def evidence_for(self, target_id: str,
                     direction: str | None = None) -> list[dict]:
        return self.evidence.by_target(target_id, direction)

    # -- I-S insights ------------------------------------------------------

    def add_insight(self, insight: dict) -> dict:
        try:
            jsonschema.validate(insight, schemas.load_schema("is"))
        except jsonschema.ValidationError as e:
            raise PlatformError("RSI_INVALID_TRANSITION",
                                f"insight schema violation: {e.message}"
                                ) from e
        for mount in insight["derived_from"]:
            iface, oid = mount["interface"], mount["object_id"]
            found = (self.questions.get(oid) if iface == "I-Q" else
                     self.hypotheses.get(oid) if iface == "I-H" else
                     self.nodes.get(oid) if iface == "I-K" else None)
            if found is None:
                raise PlatformError("RSI_DANGLING_REF",
                                    f"insight mount {iface}:{oid} not found")
        return self.insights.put(insight)

    def insights_for(self, interface: str, object_id: str) -> list[dict]:
        return [i for i in self.insights.all()
                if any(m.get("interface") == interface
                       and m.get("object_id") == object_id
                       for m in i.get("derived_from", []))]

    # -- I-K theory --------------------------------------------------------

    def put_node(self, node: dict) -> dict:
        try:
            jsonschema.validate(node, schemas.load_schema("ik_theory"))
        except jsonschema.ValidationError as e:
            raise PlatformError("RSI_INVALID_TRANSITION",
                                f"theory node schema violation: {e.message}"
                                ) from e
        return self.nodes.put(node)

    def get_node(self, node_id: str) -> dict | None:
        return self.nodes.get(node_id)

    def search_questions(self, text: str, limit: int = 20) -> list[dict]:
        return self.questions.search(text, limit)

    # -- ChangeProposal envelope -------------------------------------------

    def submit(self, proposal: dict) -> dict:
        """The single write envelope (frozen v0.2 §2.3), auto-applied.

        Idempotent by ``idempotency_key``: a replay raises
        RSI_IDEMPOTENT_REPLAY carrying the first result. Risk classification
        is recorded but NOT gated (base divergence: no human review queue)."""
        try:
            jsonschema.validate(proposal, schemas.load_schema("change_proposal"))
        except jsonschema.ValidationError as e:
            raise PlatformError("RSI_INVALID_TRANSITION",
                                f"ChangeProposal schema violation: {e.message}"
                                ) from e
        key = proposal["idempotency_key"]
        row = self._idem.execute(
            "SELECT result FROM idempotency WHERE key = ?", (key,)).fetchone()
        if row is not None:
            raise PlatformError("RSI_IDEMPOTENT_REPLAY",
                                f"idempotency key {key!r} already applied",
                                result=json.loads(row[0]))
        result = self._dispatch(proposal)
        try:
            self._idem.execute(
                "INSERT INTO idempotency (key, result) VALUES (?, ?)",
                (key, json.dumps(result)))
            self._idem.commit()
        except sqlite3.IntegrityError:
            self._idem.rollback()
            row = self._idem.execute(
                "SELECT result FROM idempotency WHERE key = ?",
                (key,)).fetchone()
            raise PlatformError("RSI_IDEMPOTENT_REPLAY",
                                f"idempotency key {key!r} already applied",
                                result=json.loads(row[0])) from None
        return result

    # required params per (interface, op) — a schema-valid envelope with a
    # missing param must fail as PlatformError, never a raw KeyError
    _REQUIRED_PARAMS = {
        ("I-Q", "create"): ("question",),
        ("I-Q", "spawn"): ("question", "parent_id", "source"),
        ("I-Q", "refine"): ("children",),
        ("I-Q", "merge"): ("into",),
        ("I-Q", "transition"): ("to",),
        ("I-H", "transition"): ("to",),
        ("I-H", "merge"): ("into",),
        ("I-S", "create"): ("insight",),
    }

    def _dispatch(self, proposal: dict) -> dict:
        op = proposal["op"]
        target = proposal["targets"][0]
        params = proposal.get("params") or {}
        refs = proposal.get("evidence_refs") or []
        base = target.get("base_version")
        iface = target["interface"]
        required = self._REQUIRED_PARAMS.get((iface, op), ())
        missing = [k for k in required if k not in params]
        if missing:
            raise PlatformError(
                "RSI_INVALID_TRANSITION",
                f"op {op!r} on {iface} requires params {list(required)}; "
                f"missing {missing}")
        if iface == "I-Q":
            if op == "create" or op == "spawn":
                child = dict(params["question"])
                if op == "spawn":
                    return {"question": self.spawn_question(
                        params["parent_id"], child, params["source"])}
                return {"question": self.create_question(child)}
            if op == "refine":
                return {"question": self.refine_question(
                    target["object_id"], params["children"], base)}
            if op == "merge":
                return {"question": self.merge_questions(
                    target["object_id"], params["into"], base)}
            if op == "transition":
                return {"question": self.transition_question(
                    target["object_id"], params["to"],
                    params.get("conversion_event"), refs, base)}
            if op == "attach_evidence":
                return {"question": self.attach_evidence(
                    target["object_id"], refs, base)}
            if op == "update":
                return {"question": self.update_question(
                    target["object_id"], params.get("fields") or {}, base)}
        if iface == "I-H":
            if op == "transition":
                return {"hypothesis": self.transition_hypothesis(
                    target["object_id"], params["to"], base)}
            if op == "merge":
                return {"hypothesis": self.merge_hypotheses(
                    target["object_id"], params["into"], base)}
        if iface == "I-S" and op == "create":
            return {"insight": self.add_insight(params["insight"])}
        raise PlatformError("RSI_INVALID_TRANSITION",
                            f"unsupported op {op!r} for interface {iface}")
