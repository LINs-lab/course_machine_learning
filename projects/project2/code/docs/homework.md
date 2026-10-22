# Project 2: The Unimplemented Strategies

Everything the base system does is deliberately **naive but walking**: the three loops turn, a self-edit can be promoted, and the demo runs keyless in seconds. What makes the system _good_ is entirely missing — that is the assignment. Every seam below is marked `TODO(homework)` in the code.

Ground rules:

- The **frozen parts** are the concept contracts (`rsi.knowledge.schemas/`), the verdict trichotomy, the trust boundary (`EDITABLE_SURFACES`), and the engine's loop structure. You implement _strategies_, not new plumbing.
- Wire your implementations in `build_default_engine` (`rsi/core/engine.py`) or via config — do not fork the engine loop.
- The test suite must stay green: your components are drop-in substitutes for Protocols, and new behavior earns new tests.
- Demo task is CPU-sub-second: iterate fast, measure everything.

## C — Core strategies

### C-1. A real Proposer (rsi/core/proposer.py) — _start here_

The shipped `HeuristicProposer` derives knobs from the task's editable file but walks a FIXED deterministic value grid per knob and never reads evidence. Do better:

- rank candidate changes by the KB's evidence (per-knob response history, `KnowledgePlatform.evidence_for`, question `best_result`/`uncertainty`);
- smarter value grids (per-knob scales, adaptive ranges around the best keep, exhaustive small-integer grids);
- consult the LLM seam (`rsi.llm`) to _compose_ an edit (the source system's implement stage rewrites the whole editable file; keep the scope guard);
- avoid repeats via insights and dead-letter-like memory.

Acceptance: on the five public tasks (start with `tasks/quad-optim` for fast iteration), beat the heuristic baseline's final `best_result` at equal cycle budget; determinism test still passes.

### C-2. A real SelfImprover (rsi/core/improver.py)

`NaiveImprover` rotates one list when no_change waste exceeds 30%. Build a real one:

- richer triggers from `EfficiencyReport.waste_patterns` (below_noise, collapse patterns, wasted cycles per surface);
- search over the strategy artifact (EXPLORE*WIDTH, PRIORITY, your own new Assign statements — remember the shape lock allows \_changing values of existing statements*; adding statements requires editing the canonical template first, i.e. you own `STRATEGY_TEMPLATE`);
- an LLM-composed candidate path through the guard;
- more than one candidate per epoch (the engine applies at most one live edit at a time — that invariant stays).

Acceptance: with your proposer fixed, the improver's `kept` edits measurably raise keep-rate / best-result over a NoopImprover control run.

### C-3. Guard layer 4 — the override-scan judge (rsi/selfedit/guard.py)

Implement an independent judge (different model than the proposer's, or a deterministic rule-based scanner) for candidate mutable regions: default-reject on any doubt. The seam is the `judge` callable; see the source system's prompt-framing ("the block is UNTRUSTED DATA") for why.

### C-4. Stage-2 hardening (rsi/core/improver.py + a new module)

The shipped arbitration is a keep-rate ratio over ≥2 runs. Extend toward the source design: paired-rubric offline verification, degradation monitoring, and a **meltdown brake** (freeze self-improvement after N consecutive reverts or M cycles with zero keeps; human-only unfreeze flag).

## E — Evaluation strategies

### E-1. BenchmarkUpgrader (rsi/evaluation/benchmark.py)

`SeedBumpUpgrader` only re-seeds observation noise. Harden the benchmark across epochs: harder instances (e.g. larger condition range / dimension for quad-optim), held-out evaluation splits, anti-shortcut variants. Remember: `task_version` changes make cross-epoch deltas non-comparable by design — handle re-baselining explicitly.

### E-2. Gates (rsi/evaluation/gates.py)

Add gates to the chain: significance vs seed variance (the task supports `RSI_STEPS`/`RSI_NOISE` knobs — replicate), shortcut / reward-hacking detection (metric gain not explained by the changed variable), a held-out E_test gate. Each gate abstains by default and only vetoes.

### E-3. Diagnostician (rsi/evaluation/diagnostics.py)

Replace the sliding-window heuristics: seed-replication noise floors, saturation as a fitted plateau model, discrimination as effect size. Your `DiagnosisReport.recommendation` drives epoch upgrades and question saturation — calibrate it so the middle loop neither thrashes nor naps.

## K — Knowledge strategies

### K-1. Question-graph evolution (rsi/core/engine.py slow loop)

The naive evolution spawns one child per resolved hypothesis. Implement real slow-loop policies over the frozen edge vocabulary (`depends_on | refines | challenges | merged_into | spawns`):

- **decompose** (`refine`) a saturated question into independent sub-questions;
- **depend** questions (order work by `depends_on` closure in `frontier`);
- **challenge**: spawn `challenges` edges when evidence contradicts a supported hypothesis (refutation chains);
- **transfer**: turn cross-question insights into new mounted hypotheses.

Acceptance: on a 20+ cycle run, your frontier policy visits questions in a principled order and the spawned graph is connected, typed, and inspectable via `python -m rsi.knowledge`.

## R — Rigor

### R-1. Result provenance (rsi/tasks/runner.py, spec.py)

The source system attests results (sha256 of result.json + returncode + editable-file hash, enforced at read time). Port it: write an attestation sidecar in `LocalRunner.dispatch`, verify in `read_result`, add failure modes `attestation_missing | attestation_mismatch`.

### R-2. Your own task bundle (tasks/)

Build a second task on the `program.md` contract — the natural course task is an MLE-flavored one (e.g. a tiny model + training loop where `editable` is the model/training file, metric = validation loss). Constraints that stay hard: exactly ONE editable file, one gate metric, fixed budget, entrypoint as a black box.

## Scripts you will use

| Script | What it does |
|---|---|
| `scripts/report.py` | Session report: trajectories, self-improvement ledger, per-edit response, ON/OFF compare (`--session main=... --session control=...`) |
| `scripts/compare_seeds.py` | Multi-seed ON/OFF ablation: per-seed running best + final best with mean ± SD |
| `scripts/probe_task_knobs.py` | Single-knob probe on one task — see which knobs actually move the gate metric |
| `scripts/llm_proposer_demo.py` | Worked example for C-1 (LLM-composed proposals); needs an OpenAI-compatible endpoint |

## Suggested order and grading hints

1. C-1 (proposer) — biggest visible effect, teaches the loop.
2. E-3 + E-2 (measurement) — without honest noise floors, C-1 improvements are unmeasurable.
3. C-2 (self-improver) — the RSI core; needs C-1+E-3 to show real yield.
4. K-1 / E-1 (slow + middle loops) — system-level research strategy.
5. C-3/C-4/R-1 — rigor and safety; separates strong submissions.

Report what you measure on all five public tasks, next to the Claude Code baseline in `baselines/public_tasks_baseline.md`: keep-rate, best-result trajectory per epoch, ledger terminal states, and the causal story of at least one promoted self-edit (what changed in `state/strategy.py`, what measurably improved after re-entry).
