"""Runtime concept types mirroring the frozen YAML schemas 1:1.

Every dataclass here is the ergonomic twin of a schema under ``schemas/``:
``to_dict()`` payloads MUST validate against the schema (contract tests
enforce this — if you add a field, add it to both places or the suite fails).

The knowledge platform speaks DICTS at its boundaries (schema-validated) and
these dataclasses inside the engine (typed). ``from_dict`` is lenient about
extra keys so a newer store can be read by an older reader.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# I-Q -------------------------------------------------------------------------

EDGE_TYPES = ("depends_on", "refines", "challenges", "merged_into", "spawns")
QUESTION_STATUSES = ("open", "partially_answered", "refined", "merged",
                     "retired", "answered")


@dataclass
class Question:
    """A research question node; its ``edges`` form the Research Question
    Graph (typed, directed). Each node binds its applicable scope, benchmark/
    metric version, current best result, open uncertainties, saturation, and
    status — the slow loop's unit of state."""

    id: str
    statement: str
    scope: str
    benchmark_ref: dict                       # {name, epoch}
    best_result: dict | None = None           # {evidence_id, score}
    uncertainty: list[str] = field(default_factory=list)
    saturation: float = 0.0
    status: str = "open"
    edges: list[dict] = field(default_factory=list)  # {type, to, source?, confidence?}
    version: int = 1
    evidence_refs: list[str] = field(default_factory=list)
    conversion_events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {
            "id": self.id, "statement": self.statement, "scope": self.scope,
            "benchmark_ref": self.benchmark_ref, "status": self.status,
            "version": self.version,
        }
        # Optional-but-populated fields travel; iq.yaml allows them all.
        out["best_result"] = self.best_result
        out["uncertainty"] = self.uncertainty
        out["saturation"] = self.saturation
        out["edges"] = self.edges
        if self.evidence_refs:
            out["evidence_refs"] = self.evidence_refs
        if self.conversion_events:
            out["conversion_events"] = self.conversion_events
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(
            id=d["id"], statement=d["statement"], scope=d["scope"],
            benchmark_ref=d["benchmark_ref"],
            best_result=d.get("best_result"),
            uncertainty=list(d.get("uncertainty") or []),
            saturation=float(d.get("saturation") or 0.0),
            status=d["status"],
            edges=list(d.get("edges") or []),
            version=int(d["version"]),
            evidence_refs=list(d.get("evidence_refs") or []),
            conversion_events=list(d.get("conversion_events") or []),
        )


# I-H -------------------------------------------------------------------------

HYPOTHESIS_STATUSES = ("draft", "active", "supported", "refuted", "merged")


@dataclass
class Hypothesis:
    """A testable hypothesis. ``derived_from`` mounts it on at least one
    question (I-Q) or theory node (I-K) — an unmounted hypothesis is
    mechanically impossible in the platform. ``testability`` is
    all-or-nothing: procedure, expected_evidence, falsifiable_by."""

    id: str
    statement: str
    derived_from: list[dict]                  # >=1 {interface: "I-Q"|"I-K", object_id}
    testability: dict                         # {procedure, expected_evidence, falsifiable_by}
    prior_confidence: float = 0.5
    confidence: float = 0.5
    confidence_algorithm: str | None = None
    status: str = "draft"
    evidence_counts: dict = field(default_factory=lambda: {
        "support": 0, "refute": 0, "inconclusive": 0})
    version: int = 1

    def to_dict(self) -> dict:
        return {
            "id": self.id, "statement": self.statement,
            "derived_from": self.derived_from, "testability": self.testability,
            "prior_confidence": self.prior_confidence,
            "confidence": self.confidence,
            "confidence_algorithm": self.confidence_algorithm,
            "status": self.status,
            "evidence_counts": self.evidence_counts,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        return cls(
            id=d["id"], statement=d["statement"],
            derived_from=list(d["derived_from"]),
            testability=d["testability"],
            prior_confidence=float(d.get("prior_confidence", 0.5)),
            confidence=float(d.get("confidence", 0.5)),
            confidence_algorithm=d.get("confidence_algorithm"),
            status=d["status"],
            evidence_counts=dict(d.get("evidence_counts") or {
                "support": 0, "refute": 0, "inconclusive": 0}),
            version=int(d["version"]),
        )


# I-E -------------------------------------------------------------------------

LINK_TYPES = ("supports", "refutes", "inconclusive")


@dataclass
class EvidenceLink:
    type: str                                 # supports | refutes | inconclusive
    to: str                                   # hypothesis or question id
    strength: float = 0.5
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {"type": self.type, "to": self.to,
                "strength": self.strength, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceLink":
        # lenient: ignore unknown keys rather than raising (module contract)
        return cls(**{k: d[k] for k in
                     ("type", "to", "strength", "confidence") if k in d})


@dataclass
class MachineEvidence:
    """Append-only machine evidence: the projection of one finished run.
    Corrections may only append a new record with ``replay.replay_of`` set."""

    id: str
    epoch: str
    run_result: dict                          # RunResult lite (see schemas/ie.yaml)
    links: list[EvidenceLink] = field(default_factory=list)
    run: dict | None = None                   # replay pair {revision, source_hash}
    failure: dict | None = None               # {class, diagnosis}
    replay: dict = field(default_factory=lambda: {
        "replayable": True, "replay_of": None})

    def to_dict(self) -> dict:
        out = {
            "id": self.id, "evidence_kind": "machine", "epoch": self.epoch,
            "run_result": self.run_result,
            "links": [l.to_dict() for l in self.links],
            "replay": self.replay,
        }
        if self.run is not None:
            out["run"] = self.run
        if self.failure is not None:
            out["failure"] = self.failure
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "MachineEvidence":
        return cls(
            id=d["id"], epoch=d["epoch"], run_result=d["run_result"],
            links=[EvidenceLink.from_dict(l) for l in d.get("links") or []],
            run=d.get("run"), failure=d.get("failure"),
            replay=dict(d.get("replay") or {"replayable": True,
                                            "replay_of": None}),
        )


# I-S -------------------------------------------------------------------------

@dataclass
class Insight:
    """Experiential knowledge attached to questions/hypotheses/theories.
    Insights never enter the evidence chain — they bias future proposals.
    ``applicability`` is required by the schema (minItems 1): an insight that
    applies nowhere is not an insight."""

    insight_id: str
    statement: str
    derived_from: list[dict]                  # mounts {interface, object_id}
    applicability: list[str]                  # schema: at least one scope
    source: object = "experiential"           # "experiential" | [evidence_id, ...]
    status: str = "active"

    def to_dict(self) -> dict:
        return {
            "insight_id": self.insight_id, "statement": self.statement,
            "derived_from": self.derived_from, "source": self.source,
            "applicability": self.applicability, "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Insight":
        return cls(
            insight_id=d["insight_id"], statement=d["statement"],
            derived_from=list(d["derived_from"]),
            source=d.get("source", "experiential"),
            applicability=list(d.get("applicability") or []),
            status=d.get("status", "active"),
        )


# I-K -------------------------------------------------------------------------

THEORY_EDGE_TYPES = ("contains", "rests_on", "specializes", "tensions")


@dataclass
class KnowledgeNode:
    """A theory-graph node (draft/established statements with typed edges)."""

    node_id: str
    statement: str
    status: str = "draft"                     # draft | established
    domain: str = "general"                   # schema requires non-empty
    branch: str = "core"
    level: int = 0                            # 0-2 caps drill depth
    provenance: dict = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> dict:
        out = {
            "node_id": self.node_id, "statement": self.statement,
            "status": self.status, "domain": self.domain,
            "branch": self.branch, "level": self.level,
            "edges": self.edges, "version": self.version,
        }
        # provenance is optional in ik_theory but REQUIRES {source} when
        # present — omit it entirely when the caller has nothing to cite.
        if self.provenance:
            out["provenance"] = self.provenance
        if self.aliases:
            out["aliases"] = self.aliases
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeNode":
        return cls(
            node_id=d["node_id"], statement=d["statement"],
            status=d.get("status", "draft"),
            domain=d.get("domain") or "general",
            branch=d.get("branch") or "core",
            level=int(d.get("level", 0)),
            provenance=dict(d.get("provenance") or {}),
            edges=list(d.get("edges") or []),
            aliases=list(d.get("aliases") or []),
            version=int(d.get("version", 1)),
        )


# ChangeProposal ---------------------------------------------------------------

OPS = ("create", "refine", "merge", "transition", "attach_evidence", "spawn",
       "update")


@dataclass
class ChangeProposal:
    """The single write envelope for engine/human writes into I-Q/I-H/I-S.

    ``targets[].base_version`` gives optimistic concurrency (a mismatch is
    RSI_VERSION_CONFLICT, retryable); ``idempotency_key`` replays the first
    result instead of double-applying. The source system classified risk
    L0/L1/L2 with a human review queue for L2 — the base system auto-applies
    everything (documented divergence; see docs/architecture.md)."""

    proposal_id: str
    actor: str                                # engine | human | service
    targets: list[dict]                       # [{interface, object_id, base_version?}]
    op: str
    idempotency_key: str
    params: dict = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    risk_level: str = "L0"
    status: str = "pending"

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id, "actor": self.actor,
            "targets": self.targets, "op": self.op, "params": self.params,
            "idempotency_key": self.idempotency_key,
            "evidence_refs": self.evidence_refs,
            "risk_level": self.risk_level, "status": self.status,
        }
