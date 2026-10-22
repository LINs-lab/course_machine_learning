"""The Proposer seam: turn knowledge state into the next single-variable
proposal (the fast loop's Think/Propose stages, folded from the source
system's think → propose pair).

The shipped ``HeuristicProposer`` is deliberately naive: it derives knobs
from the task's editable file (module-level numeric assignments) and walks a
FIXED deterministic value grid per knob — gentle perturbations first, then
scale jumps — in the strategy file's priority order, skipping (knob, value)
pairs already tried for this question. It never looks at evidence beyond the
incumbent value, never consults theory, and its "rationale" is a template —
that whole gap is TODO(homework) (docs/homework.md, task C-1): evidence-driven
proposal ranking, smarter value grids, hypothesis-driven search, LLM-composed
rewrites of the editable file (the source system's implement stage), novelty
gating, dead-letter avoidance.

The strategy artifact (``state/strategy.py``) is the self-editable surface:
it is re-read EVERY cycle, so a promoted self-improvement re-enters here —
``load_strategy`` is the observable re-entry point of the whole RSI loop.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Protocol

from rsi.core.types import Proposal
from rsi.selfedit import sentinels
from rsi.tasks.spec import TaskSpec

STRATEGY_TEMPLATE = '''"""Proposer strategy — the SELF-EDITABLE surface.

Everything between the sentinel markers may be rewritten by the
self-improvement loop (guard-checked: statement shape locked, scorer
untouchable). The proposer re-reads this file every cycle.
"""
# <<<SELF-IMPROVE-MUTABLE
# Knob exploration priority + per-knob candidate width. Values are plain
# python literals. PRIORITY entries that no longer match the task's knobs
# are ignored; remaining knobs follow file order.
PRIORITY = ["MOMENTUM", "LR", "CLIP"]
EXPLORE_WIDTH = 6
# SELF-IMPROVE-MUTABLE>>>
'''


def derive_knobs(source: str) -> dict[str, float]:
    """Module-level numeric assignments of an editable file, in file order."""
    out: dict[str, float] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = float(value)
    return out


def value_grid(v: float) -> list[float]:
    """Deterministic candidate values around the current knob value:
    gentle perturbations first (x2, /2, +0.5, -0.5 for unit-scale knobs),
    then scale jumps (x20, /20, x5, /5). Positive values only."""
    cands: list[float] = []
    for c in (v * 2.0, v * 0.5):
        cands.append(c)
    if 0.5 <= v <= 4.0:
        cands.extend([v + 0.5, max(0.05, v - 0.5)])
    cands.extend([v * 20.0, v * 0.05, v * 5.0, v * 0.2])
    if v == 0.0:
        # a zero-valued knob has no multiplicative grid; offer sensible
        # unit-scale alternatives (deterministic)
        cands = [0.9, 0.5, 1.0]
    seen: set[float] = set()
    out: list[float] = []
    for c in cands:
        c = round(c, 10)
        if c > 0 and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def ensure_strategy_file(path: str | Path) -> Path:
    """Create the strategy artifact with the default sentinel region if it
    does not exist; return the path (idempotent)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(STRATEGY_TEMPLATE)
    sentinels.split_regions(path.read_text())  # integrity check on load
    return path


def load_strategy(path: str | Path) -> dict:
    """Read the mutable region's top-level assignments (name -> literal).

    Only the sentinel-bounded region is parsed; statements outside it are
    invisible to strategy reads. A guarded write keeps the file parseable
    and shape-locked, so this never needs a try/except.
    """
    _, mutable, _ = sentinels.split_regions(Path(path).read_text())
    out: dict = {}
    for node in ast.parse(mutable).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
    return out


class Proposer(Protocol):
    def propose(self, question: dict, spec: TaskSpec,
                incumbent_params: dict | None, incumbent_value: float | None,
                history: list, strategy: dict) -> Proposal | None: ...


class HeuristicProposer:
    """Deterministic grid walk over the task's own knobs; None = grid
    exhausted for this question (the engine's slow-loop trigger)."""

    def __init__(self, counter_start: int = 0):
        self._n = counter_start

    def propose(self, question: dict, spec: TaskSpec,
                incumbent_params: dict | None, incumbent_value: float | None,
                history: list, strategy: dict) -> Proposal | None:
        # Baseline cycle: establish the incumbent before any change. Per
        # (question, EPOCH): an epoch upgrade re-baselines the incumbent
        # solution under the new seed, so a baseline re-run is due whenever
        # this question has no baseline run IN THE CURRENT epoch. A baseline
        # that FAILED (e.g. an always-crashing task) must NOT be re-proposed
        # forever — fall through to the grid walk instead.
        epoch = question["benchmark_ref"].get("epoch")
        if incumbent_value is None and not any(
                r.question_id == question["id"]
                and r.variable_changed == "baseline"
                and getattr(r, "epoch", None) == epoch for r in history):
            self._n += 1
            return Proposal(
                proposal_id=f"p{self._n:03d}",
                question_id=question["id"],
                hypothesis_id=_active_hypothesis_id(question, history),
                variable_changed="baseline", params={},
                rationale="baseline-establishing run (pristine editable)",
                benchmark_ref=question["benchmark_ref"])

        source = ((spec.editable_file).read_text()
                  if incumbent_params is None else None)
        knobs = (dict(incumbent_params) if incumbent_params is not None
                 else derive_knobs(source))
        if not knobs:
            return None  # task has no numeric knobs — nothing to walk

        # tried pairs are scoped to THIS question. A pair whose single
        # attempt failed is deferred, not blocking: a crash may be transient,
        # but a diverging first value must never head-of-line block the whole
        # grid. ONE retry at most — a twice-failed pair is dead (otherwise
        # the grid walk ends in an infinite retry loop, never exhausting).
        qid = question["id"]
        tried_ok = {(r.variable_changed, _param_key(r.params))
                    for r in history
                    if r.question_id == qid and r.status != "fail"}
        fail_counts: dict = {}
        for r in history:
            if r.question_id == qid and r.status == "fail":
                key = (r.variable_changed, _param_key(r.params))
                fail_counts[key] = fail_counts.get(key, 0) + 1
        priority = [k for k in (strategy.get("PRIORITY") or [])
                    if k in knobs]
        ordered = priority + [k for k in knobs if k not in priority]
        width = max(1, int(strategy.get("EXPLORE_WIDTH", 6)))
        prefix = spec.editable[0].rsplit(".", 1)[0]
        deferred: tuple[str, float, dict] | None = None
        for knob in ordered:
            for value in value_grid(knobs[knob])[:width]:
                params = {knob: value}
                key = (f"{prefix}.{knob}", _param_key(params))
                if key in tried_ok:
                    continue
                fails = fail_counts.get(key, 0)
                if fails:
                    if fails == 1 and deferred is None:
                        deferred = (knob, value, params)
                    continue
                self._n += 1
                return Proposal(
                    proposal_id=f"p{self._n:03d}",
                    question_id=question["id"],
                    hypothesis_id=_active_hypothesis_id(question, history),
                    variable_changed=f"{prefix}.{knob}", params=params,
                    rationale=f"grid walk: try {knob}={value!r} on the "
                              f"incumbent (priority order)",
                    benchmark_ref=question["benchmark_ref"])
        if deferred is not None:
            knob, value, params = deferred
            self._n += 1
            return Proposal(
                proposal_id=f"p{self._n:03d}",
                question_id=question["id"],
                hypothesis_id=_active_hypothesis_id(question, history),
                variable_changed=f"{prefix}.{knob}", params=params,
                rationale=f"grid walk: retry {knob}={value!r} (previous "
                          f"attempts failed; all fresh candidates exhausted)",
                benchmark_ref=question["benchmark_ref"])
        return None  # exhausted — the engine's slow-loop trigger


def _active_hypothesis_id(question: dict, history: list) -> str:
    if history:
        return history[-1].hypothesis_id
    return f"{question['id']}-h0"


def _param_key(params: dict) -> str:
    return ",".join(f"{k}={v!r}" for k, v in sorted(params.items()))
