"""Core fast-loop data types.

Vocabulary is frozen from the source systems: keep|discard|fail (finalize /
I-E run_result.status), and EfficiencyReport mirrors the shape of the source
``graph/efficiency.py`` waste diagnosis (keep_rate, wasted fraction, waste
patterns) that triggers self-improvement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rsi.evaluation.gates import MetricReading

BASELINE_VARIABLE = "baseline"


@dataclass
class Proposal:
    """One single-variable experiment proposal (the fast loop's unit)."""
    proposal_id: str
    question_id: str
    hypothesis_id: str
    variable_changed: str             # e.g. "solver.LR"; "baseline" = the
                                      # baseline-establishing run
    params: dict                      # knob -> value applied to the editable
    rationale: str
    benchmark_ref: dict               # BenchmarkRef.to_dict() at submission

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "question_id": self.question_id,
            "hypothesis_id": self.hypothesis_id,
            "variable_changed": self.variable_changed,
            "params": self.params,
            "rationale": self.rationale,
            "benchmark_ref": self.benchmark_ref,
        }


@dataclass
class RunRecord:
    """The finalized outcome of one cycle — the L0 record that becomes
    machine evidence (I-E) via the platform's push_run."""

    run_id: str                       # "cycle-003": cycle-derived, no clock
    proposal_id: str
    hypothesis_id: str
    question_id: str
    task: str
    epoch: str                        # epoch at SUBMISSION (freeze/archive)
    status: str                       # keep | discard | fail
    metric: MetricReading
    variable_changed: str
    created: str                      # ISO, informational only
    duration_s: float
    params: dict = field(default_factory=dict)  # knobs actually applied
    verdict_reasons: list[str] = field(default_factory=list)
    workdir: str | None = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "proposal_id": self.proposal_id,
            "hypothesis_id": self.hypothesis_id,
            "question_id": self.question_id, "task": self.task,
            "epoch": self.epoch, "status": self.status,
            "metric": {
                "name": self.metric.name, "direction": self.metric.direction,
                "value": self.metric.value, "baseline": self.metric.baseline,
                "delta": self.metric.delta,
                "failure_mode": self.metric.failure_mode,
            },
            "variable_changed": self.variable_changed,
            "created": self.created, "duration_s": self.duration_s,
            "params": self.params,
            "verdict_reasons": self.verdict_reasons,
            "workdir": self.workdir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        m = d["metric"]
        return cls(
            run_id=d["run_id"], proposal_id=d["proposal_id"],
            hypothesis_id=d["hypothesis_id"], question_id=d["question_id"],
            task=d["task"], epoch=d["epoch"], status=d["status"],
            metric=MetricReading(
                name=m["name"], direction=m["direction"], value=m["value"],
                baseline=m.get("baseline"), failure_mode=m.get("failure_mode")),
            variable_changed=d["variable_changed"], created=d["created"],
            duration_s=d["duration_s"],
            params=dict(d.get("params") or {}),
            verdict_reasons=list(d.get("verdict_reasons") or []),
            workdir=d.get("workdir"))


@dataclass
class EfficiencyReport:
    """Waste-signal diagnosis over the run history (source: efficiency.py
    Phase-A aggregation shape). ``waste_patterns`` is what the improver
    triggers on: {pattern, surface, support, est_cost}."""

    n_cycles: int
    keep: int
    discard: int
    fail: int
    keep_rate: float
    wasted_fraction: float
    n_new_bests: int
    waste_patterns: list[dict] = field(default_factory=list)


def build_efficiency_report(runs: list[RunRecord]) -> EfficiencyReport:
    n = len(runs)
    keep = sum(1 for r in runs if r.status == "keep")
    discard = sum(1 for r in runs if r.status == "discard")
    fail = sum(1 for r in runs if r.status == "fail")
    # new bests: keeps whose value strictly beat all earlier values (naive:
    # direction-aware scan over non-fail runs)
    def _better(a, b, direction):
        return a < b if direction == "min" else a > b
    n_new_bests, cur_best = 0, None
    for r in runs:
        v = r.metric.value
        if r.status == "keep" and v is not None:
            if cur_best is None or _better(v, cur_best, r.metric.direction):
                n_new_bests += 1
                cur_best = v
    # waste patterns from verdict reasons (naive classification)
    counts: dict[str, int] = {}
    for r in runs:
        for reason in r.verdict_reasons:
            key = reason.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
    patterns = [{"pattern": k, "surface": "proposer", "support": c,
                 "est_cost": c}
                for k, c in sorted(counts.items(), key=lambda kv: -kv[1])
                if k not in ("no gate objected",)]
    return EfficiencyReport(
        n_cycles=n, keep=keep, discard=discard, fail=fail,
        keep_rate=round(keep / n, 4) if n else 0.0,
        wasted_fraction=round((discard + fail) / n, 4) if n else 0.0,
        n_new_bests=n_new_bests, waste_patterns=patterns)
