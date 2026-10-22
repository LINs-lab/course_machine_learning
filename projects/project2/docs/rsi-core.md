# The RSI Core

This page covers the engine (`rsi.core`): the experiment pipeline, the proposer seam, and above all the self-improvement machinery — the trust boundary, the edit lifecycle, and the re-entry that makes the loop recursive.

## The five-stage pipeline (`rsi.core.pipeline`)

```
propose → implement → run → finalize → analyze
```

(The source system's seven stages fold: `think` becomes the proposer's context assembly, `init` becomes task load + workdir staging.) One cycle:

1. **propose** — the Proposer turns (question, incumbent, history, strategy) into a `Proposal`: one `variable_changed`, one param set, a benchmark_ref.
2. **implement** — `stage_workdir` copies the pristine task bundle into an isolated `runs/cycle-NNN/`, applies the params to the editable file **on top of the incumbent bytes** (keeps accumulate), and runs the scope guard (only the editable file may differ; the file must compile).
3. **run** — `LocalRunner` dispatches the entrypoint with the benchmark identity injected (`RSI_EPOCH`, `RSI_BENCHMARK_SEED`), a wall-clock timeout, and a captured run.log.
4. **finalize** — the metric is read from the result file (`failure_mode` vocabulary: `missing_result | invalid_metric | missing_metric | timeout | crash`); the `GateChain` decides `keep | discard | fail`. Keep ratchets the incumbent and the question's `best_result`.
5. **analyze** — the run becomes machine evidence (push_run), confidence is aggregated, and count-gated transitions may fire (hypothesis → supported; question → partially_answered with a conversion event).

## The self-editable surface: `state/strategy.py`

On first run the engine materializes:

```python
# <<<SELF-IMPROVE-MUTABLE
PRIORITY = ["MOMENTUM", "LR", "CLIP"]
EXPLORE_WIDTH = 6
# SELF-IMPROVE-MUTABLE>>>
```

`HeuristicProposer` re-reads this file **every cycle** — that read is the observable re-entry point of the whole RSI loop.

## The guard (`rsi.selfedit`)

Every self-edit candidate passes `check_candidate`, layers 0–4:

0. **surface allowlist** — `EDITABLE_SURFACES = {"state/strategy.py"}` and nothing else. This is the trust boundary made executable: the scorer, the gates, the platform can never be edited by the improver;
1. **sentinel integrity** — exactly one well-formed mutable region; no smuggled boundary tokens;
2. **byte-identity** — everything outside the region is unchanged;
3. **statement-shape lock** — the region's top-level statement structure (kind + ALL assignment targets + the RHS value TYPE) must match the canonical: values may change within their type, statements may not be added/removed/retyped, and non-Name targets are locked by their dumped form (no smuggled `_hijack = lambda: ...`, no `PRIORITY = os.system = [...]` multi-target smuggle, no `EXPLORE_WIDTH = "lots"` type confusion);
4. **override scan** — an optional independent judge answering `override: yes|no`; unparseable or failing ⇒ reject (default-reject).

Note the honest boundary: layers 0–3 are STRUCTURAL — a knob's right-hand side may be any expression of the same type, so without a real layer-4 judge the guard cannot contain semantics (e.g. a value expression that calls out to `os` at import time). Until homework C-3 lands, treat the mutable region as data-only by convention and audit kept edits in the ledger.

Layer 4 ships as a seam (homework task C-3).

## The edit lifecycle (`rsi.core.ledger`)

Append-only JSONL, folded by edit id (last row wins):

```
applying → observing → kept        (kept == promoted: the edit compounds)
                     → reverted    (prior bytes restored from the ledger)
applying → failed | rejected
```

Content-addressed edit ids (same candidate text ⇒ same id) keep the whole system deterministic.

## Stage2-lite arbitration (`rsi.core.improver`)

While an edit is `observing`, the middle loop eventually calls `arbitrate(edit_id, baseline_runs, window_runs)`:

- keep-rate(window) < 0.5 × keep-rate(baseline) ⇒ **reverted** (bytes restored, terminal);
- otherwise ⇒ **kept** (stays live, terminal).

The source system's full Stage-2 (paired rubric judges, offline replay filters, degradation monitors, a meltdown brake) is deliberately out of scope — see homework tasks C-2/C-4.

## Terminal-transition policy (engine-level)

The platform enforces frozen count thresholds (supported ≥ 2 supports; refuted ≥ 1 refute — source constants). The ENGINE decides when to attempt a terminal transition, with a net-refutation rule: ``refuted`` is attempted only when refute ≥ 2 AND refute > support. Without the net rule, a single flat discard terminalizes a hypothesis that also has supporting keeps, and the slow loop churns questions without accumulating evidence (a failure mode observed and fixed during the first full review).

## Hypothesis succession and the slow-loop cadence

A terminal hypothesis does **not** end its question. When a hypothesis resolves (supported/refuted), the fast loop continues under a SUCCESSOR hypothesis mounted on the same question (`<qid>-h0`, `-h1`, …) — one question hosts many hypotheses over its life, which is the research-process shape and what makes the fast loop "repeatedly propose experiments on ONE research question". The question itself is abandoned (partial progress recorded, child spawned, frontier advanced) only on:

- **saturation** ≥ `engine.saturation_threshold` (the middle loop's verdict — a *mature* diagnosis routing to `saturate_question`; an `upgrade_benchmark` routing instead resets the question's saturation and the question CONTINUES on the new epoch, its incumbent value re-baselined by a fresh per-(question, epoch) baseline run), or
- **proposer exhaustion** (the grid walk returns None for this question).

Letting the 2-support threshold switch questions directly was the second quiet-failure found by measurement: 8 questions in 24 cycles, none explored deeper than 3 runs, and the epoch upgrade never fired (per-question runs never reached the diagnostician's minimum). With succession, the same runs produce 1–3 deep questions and multiple evaluation epochs.

## What ships vs what is homework

Shipped (naive, walking): `HeuristicProposer` (derives knobs from the task's editable file, walks a fixed deterministic value grid), `NoopImprover` + `NaiveImprover` (rotate strategy priority when no_change waste dominates), `SeedBumpUpgrader`, three verdict gates, sliding-window diagnostics, template question spawning.

Homework (`TODO(homework)` in code; inventory in `docs/homework.md`): evidence/LLM-driven proposers, real self-improvers (including sentinel-region code edits beyond value rotation), significance and shortcut/reward-hacking gates, saturation estimators, benchmark hardening, question-graph evolution strategies.

## Engine wiring

```python
from rsi.core.engine import EngineConfig, build_default_engine

engine = build_default_engine(EngineConfig(
    task_dir="tasks/quad-optim", out_dir="out",
    cycles=6, middle_every=3, seed=7, improver="naive"))
summary = engine.run("Minimize final_loss on quad-optim")
```

`build_default_engine` is the composition root: replace the naive components there (or register your own) — `RSIEngine.run` itself should not need to change.
