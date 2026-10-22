# Ported (semantics preserved, five stages) from
# project_collab_auto_research/graph/cli/{think,propose,init,implement,run,
# finalize,analyze}.py: think folds into the proposer's context, init folds
# into task load + workdir staging. One cycle = one single-variable
# experiment through the full pipeline; keep/discard is decided by the gate
# chain (the scorer), never by the proposer — the trust boundary in action.
"""One experiment cycle: propose → implement → run → finalize → analyze.

Pure orchestration over injected seams (Proposer, TaskRunner, GateChain) —
the engine and tests both call ``run_cycle``. The knowledge-platform writes
(evidence, confidence) stay in the engine: this function returns the
RunRecord and leaves persistence to the caller.
"""
from __future__ import annotations

import datetime
from pathlib import Path

from rsi.core.types import Proposal, RunRecord
from rsi.evaluation.gates import GateChain, MetricReading, RunContext
from rsi.tasks.runner import LocalRunner, stage_workdir
from rsi.tasks.spec import TaskSpec, read_result


def failed_run(spec: TaskSpec, proposal: Proposal, cycle: int,
               out_dir: str | Path, reason: str) -> RunRecord:
    """Synthesize a ``fail`` RunRecord for a cycle that never dispatched
    (staging/scope-guard crash — e.g. a student proposer named a knob that
    does not exist). The failure enters the record like any other failed
    experiment instead of killing the engine."""
    return RunRecord(
        run_id=f"cycle-{cycle:03d}",
        proposal_id=proposal.proposal_id,
        hypothesis_id=proposal.hypothesis_id,
        question_id=proposal.question_id,
        task=spec.name,
        epoch=proposal.benchmark_ref.get("epoch", ""),
        status="fail",
        metric=MetricReading(
            name=spec.metric.name, direction=spec.metric.direction,
            value=None, baseline=None, failure_mode="crash"),
        variable_changed=proposal.variable_changed,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        duration_s=0.0,
        params=dict(proposal.params),
        verdict_reasons=[f"pre-dispatch failure: {reason}"],
        workdir=str(Path(out_dir) / "runs" / f"cycle-{cycle:03d}"),
    )


def run_cycle(spec: TaskSpec, proposal: Proposal, cycle: int,
              runner: LocalRunner, gates: GateChain,
              incumbent_bytes: bytes | None, incumbent_value: float | None,
              benchmark_seed: int, out_dir: str | Path,
              noise_floor: float | None = None,
              ) -> tuple[RunRecord, Path]:
    """Stage → dispatch → read → gate. Returns (run_record, workdir)."""
    workdir = Path(out_dir) / "runs" / f"cycle-{cycle:03d}"
    stage_workdir(spec, workdir, incumbent_bytes, proposal.params)
    staged_editable = (workdir / spec.editable[0]).read_bytes()

    dispatch = runner.dispatch(
        spec, workdir,
        env_extra={"RSI_EPOCH": proposal.benchmark_ref.get("epoch", ""),
                   "RSI_BENCHMARK_SEED": str(benchmark_seed)})
    result = read_result(spec, workdir)

    # post-run integrity: the task must not rewrite its own editable file
    # (the incumbent ratchet and the evidence source_hash trust these bytes;
    # the pre-run scope guard cannot see this). A DELETED editable is the
    # same violation — fail the run, never crash the engine.
    try:
        mutated = ((workdir / spec.editable[0]).read_bytes()
                   != staged_editable)
    except OSError:
        mutated = True
    if mutated:
        failure_mode = "editable_mutated"
        value = None
    else:
        value = result["metric"]
        failure_mode = result["failure_mode"]
        if dispatch.timed_out and failure_mode is None:
            failure_mode = "timeout"
        elif dispatch.exit_code != 0 and failure_mode is None:
            failure_mode = "crash"

    reading = MetricReading(
        name=spec.metric.name, direction=spec.metric.direction,
        value=value, baseline=incumbent_value, failure_mode=failure_mode)
    ctx = RunContext(baseline=incumbent_value, noise_floor=noise_floor,
                     epoch=proposal.benchmark_ref.get("epoch", ""))
    verdict = gates.check(reading, ctx)

    run = RunRecord(
        run_id=f"cycle-{cycle:03d}",
        proposal_id=proposal.proposal_id,
        hypothesis_id=proposal.hypothesis_id,
        question_id=proposal.question_id,
        task=spec.name,
        epoch=proposal.benchmark_ref.get("epoch", ""),
        status=verdict.status,
        metric=reading,
        variable_changed=proposal.variable_changed,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        duration_s=round(dispatch.duration_s, 3),
        params=dict(proposal.params),
        verdict_reasons=list(verdict.reasons),
        workdir=str(workdir),
    )
    return run, workdir
