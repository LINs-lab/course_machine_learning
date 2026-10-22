"""Verdict gates: the finalize decision as an ordered, substitutable chain.

The trichotomy is the frozen contract (source finalize vocabulary, I-E
``run_result.status``): **keep | discard | fail**. A gate may ABSTAIN
(verdict None) — the chain's first decisive verdict wins; all-abstain on a
valid run means keep (a real improvement needs no gate's permission, only no
gate's veto — mirroring the source's "gates reject, they do not bless").

Shipped naive gates:

- ``NoChangeGate`` — delta <= 0 (worse or flat) → discard "no_change"
- ``CollapseGate`` — failure_mode or non-finite value → fail "collapse"
- ``NoiseGate``    — 0 < delta <= noise_floor → discard "below_noise"

TODO(homework): significance testing against seed variance, shortcut /
reward-hacking detectors (metric improvements not explained by the changed
variable), held-out E_test gates (see docs/homework.md, task E-2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

KEEP, DISCARD, FAIL = "keep", "discard", "fail"


@dataclass
class MetricReading:
    name: str
    direction: str                # min | max
    value: float | None           # None = failure_mode set
    baseline: float | None        # incumbent value at proposal time; None on
                                  # the first (baseline-establishing) run
    failure_mode: str | None = None

    @property
    def delta(self) -> float | None:
        """Signed improvement (positive = better) or None when undefined."""
        if self.value is None or self.baseline is None:
            return None
        return ((self.baseline - self.value) if self.direction == "min"
                else (self.value - self.baseline))


@dataclass
class RunContext:
    baseline: float | None
    noise_floor: float | None
    epoch: str


@dataclass
class GateResult:
    verdict: str | None           # None = abstain
    reason: str = ""


class Gate(Protocol):
    def check(self, reading: MetricReading, ctx: RunContext) -> GateResult: ...


@dataclass
class Verdict:
    status: str                   # keep | discard | fail
    reasons: list[str]


class GateChain:
    """Ordered gates; first decisive verdict wins; all-abstain ⇒ keep."""

    def __init__(self, gates: list[Gate]):
        self.gates = list(gates)

    def check(self, reading: MetricReading, ctx: RunContext) -> Verdict:
        reasons: list[str] = []
        for gate in self.gates:
            res = gate.check(reading, ctx)
            if res.verdict is None:
                if res.reason:
                    reasons.append(res.reason)
                continue
            return Verdict(res.verdict, [res.reason] if res.reason else [])
        return Verdict(KEEP, reasons or ["no gate objected"])


class CollapseGate:
    """A failed / non-finite run fails outright — a divergence is the
    clearest discard and must never masquerade as a win."""

    def check(self, reading: MetricReading, ctx: RunContext) -> GateResult:
        if reading.failure_mode is not None:
            return GateResult(FAIL, f"collapse:{reading.failure_mode}")
        if reading.value is None or not math.isfinite(reading.value):
            return GateResult(FAIL, "collapse:non-finite")
        return GateResult(None)


class NoChangeGate:
    def check(self, reading: MetricReading, ctx: RunContext) -> GateResult:
        delta = reading.delta
        if delta is not None and delta <= 0:
            return GateResult(DISCARD, "no_change")
        return GateResult(None)


class NoiseGate:
    """A gain smaller than the noise floor is indistinguishable from noise."""

    def __init__(self, default_floor: float = 1e-9):
        self.default_floor = default_floor

    def check(self, reading: MetricReading, ctx: RunContext) -> GateResult:
        delta = reading.delta
        floor = ctx.noise_floor
        if floor is None:
            floor = self.default_floor
        if delta is not None and 0 < delta <= floor:
            return GateResult(DISCARD, "below_noise")
        return GateResult(None)
