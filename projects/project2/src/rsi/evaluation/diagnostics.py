"""Diagnosis: aggregate multi-round evidence into a middle-loop report.

The DiagnosisReport is what drives epoch arbitration (upgrade the benchmark?)
and slow-loop decisions (saturate the question?), and its ``saturation`` is
written back to the question node (I-Q).

This module deliberately depends on NOTHING from rsi.knowledge — diagnostics
consume plain RunRecord sequences passed in (interface frozen for the
instructor-arranged evaluation tasks; see docs/evaluation.md).

TODO(homework): the naive implementations below are deliberately crude:
- saturation: recent-window best improvement vs early-window best improvement
- noise_floor: mean |delta| among non-improving runs (a placeholder for a
  real seed-variance estimate)
- discrimination: (best keep − median) / noise_floor
Real estimators (seed replication, paired tests, per-knob response surfaces)
are the assignment (docs/homework.md, task E-3).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Protocol, Sequence

CONTINUE = "continue"
UPGRADE_BENCHMARK = "upgrade_benchmark"
SATURATE_QUESTION = "saturate_question"


@dataclass
class DiagnosisReport:
    question_id: str
    epoch: str
    n_runs: int
    saturation: float                 # 0..1, written back to the I-Q node
    noise_floor: float
    discrimination: float
    recommendation: str               # continue | upgrade_benchmark | saturate_question
    mature: bool = True               # False = too little evidence to say
                                      # anything; callers must NOT write the
                                      # immature numbers back (they would
                                      # erase what the question already
                                      # learned)
    details: dict = field(default_factory=dict)


class Diagnostician(Protocol):
    def diagnose(self, history: Sequence, epoch: str,
                 question_id: str = "") -> DiagnosisReport: ...


class NaiveDiagnostician:
    """Sliding-window heuristics; deterministic in inputs only."""

    def __init__(self, window: int = 5, upgrade_saturation: float = 0.8,
                 min_runs: int = 3):
        self.window = max(1, window)
        self.upgrade_saturation = upgrade_saturation
        self.min_runs = min_runs

    def diagnose(self, history: Sequence, epoch: str,
                 question_id: str = "") -> DiagnosisReport:
        # history: RunRecord-like objects with .status and .metric
        # (.value/.direction/.delta). A record WITHOUT an epoch belongs to
        # no epoch (defaulting to the queried one would count it everywhere).
        runs = [r for r in history
                if getattr(r, "epoch", None) is not None
                and r.epoch == epoch]
        direction = next((r.metric.direction for r in runs
                          if getattr(r.metric, "direction", None)), "min")
        best = min if direction == "min" else max
        # values: non-fail runs only, and catastrophic outliers excluded (a
        # diverged discard is collapse evidence, not a saturation statistic).
        # Exclusion is MEDIAN-anchored, not best-anchored: a solved task can
        # score exactly 0.0, and a best-anchored scale of ~0 would exclude
        # every informative run — the diagnostics go blind precisely when the
        # task is solved.
        raw = [r.metric.value for r in runs
               if r.status != "fail" and r.metric.value is not None]
        if raw:
            center = statistics.median(raw)
            scale = max(abs(center), 1e-12)
            values = [v for v in raw if abs(v - center) <= 10.0 * scale]
        else:
            values = []
        n = len(runs)
        if n < self.min_runs or len(values) < 2:
            return DiagnosisReport(
                question_id=question_id, epoch=epoch, n_runs=n,
                saturation=0.0, noise_floor=0.0, discrimination=0.0,
                recommendation=CONTINUE, mature=False,
                details={"reason": "insufficient runs"})

        early = values[:max(2, len(values) // 2)]
        recent = values[-self.window:]
        best_recent = best(recent)
        # improvement of the recent window vs the early half, normalized by
        # the total observed improvement (0 = full headroom, 1 = nothing left).
        early_gain = abs(early[0] - best(early))
        recent_gain = abs(max(recent) - min(recent))
        total = max(early_gain + recent_gain, 1e-12)
        saturation = 1.0 - min(recent_gain / total, 1.0)

        # noise floor: mean |delta| among non-improving runs in-epoch, capped
        # at 10% of the incumbent scale — a |delta| beyond that is collapse
        # territory (a genuinely worse/diverged candidate), not measurement
        # noise, and must not poison the gate. Crude by design: a real
        # estimator replicates seeds (docs/homework.md, task E-3).
        deltas = [r.metric.delta for r in runs
                  if getattr(r.metric, "delta", None) is not None]
        worse = [abs(d) for d in deltas if d <= 0]
        scale_cap = 0.1 * max(abs(best_recent), 1e-12)
        noise_floor = min(statistics.fmean(worse), scale_cap) if worse else 0.0

        median = statistics.median(values)
        spread = (median - best_recent) if direction == "min" else (best_recent - median)
        discrimination = spread / max(noise_floor, 1e-12)

        if saturation >= self.upgrade_saturation:
            # Two different saturated states need two different responses:
            # low discrimination means the benchmark cannot tell candidates
            # apart (observation noise ≈ spread) → harden the OBSERVER
            # (upgrade epoch, same question continues); high discrimination
            # means the best keep is cleanly separated and nothing recent
            # improved → the QUESTION is exhausted (slow loop abandons it).
            recommendation = (SATURATE_QUESTION if discrimination >= 1.0
                              else UPGRADE_BENCHMARK)
        else:
            recommendation = CONTINUE
        return DiagnosisReport(
            question_id=question_id, epoch=epoch, n_runs=n,
            saturation=round(saturation, 4),
            noise_floor=round(noise_floor, 8),
            discrimination=round(discrimination, 4),
            recommendation=recommendation,
            details={"best": best_recent, "n_values": len(values)})
