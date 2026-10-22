# Architecture & Functional Reference

Autoresearch-Base is a recursive self-improvement (RSI) research system in miniature: a *harness* that runs single-variable experiments against a task bundle, a *knowledge platform* that accumulates what those experiments prove, and a *self-improvement* mechanism that lets the harness rewrite a guarded part of its own strategy — with the edit re-entering the very next cycle. This page is the code-level map: what each module is, what the key types do, and how one cycle of data flows through them.

## The three loops

```
                 ┌──────────────────────────────────────────────────────┐
                 │ RSIEngine.run()  (rsi/core/engine.py)               │
                 └──────────────────────────────────────────────────────┘
  FAST loop (every cycle)
  ┌───────────────────────────────────────────────────────────────────┐
  │ strategy ← load_strategy(state/strategy.py)   ← RE-ENTRY POINT    │
  │ proposal ← Proposer.propose(question, history, strategy)          │
  │ stage → dispatch (RSI_EPOCH, RSI_BENCHMARK_SEED) → read_result    │
  │ verdict  ← GateChain (Collapse → NoChange → Noise)                │
  │ keep ⇒ incumbent ratchet + question.best_result                   │
  │ evidence ← platform.push_run (append-only, idempotent by run_id)  │
  │ confidence ← platform.update_confidence (Σsupports − Σrefutes)    │
  └───────────────────────────────────────────────────────────────────┘
  MIDDLE loop (every `middle_every` cycles)
  ┌───────────────────────────────────────────────────────────────────┐
  │ DiagnosisReport ← Diagnostician (saturation / noise / discrimination)
  │ mature ⇒ question.saturation ← report                             │
  │ upgrade? ⇒ BenchmarkUpgrader + EpochManager.upgrade ⇒ re-baseline │
  │ efficiency ← build_efficiency_report (waste patterns)             │
  │ live edit? ⇒ SelfImprover.arbitrate → kept (= promoted) | reverted│
  │ else     ⇒ SelfImprover.propose + apply (guard-checked) → observing│
  └───────────────────────────────────────────────────────────────────┘
  SLOW loop (saturation ≥ threshold | proposer exhausted)
  ┌───────────────────────────────────────────────────────────────────┐
  │ hypothesis terminal ⇒ successor hypothesis on the SAME question   │
  │ question abandoned ⇒ transition + spawn child (typed edge)        │
  │ next ← frontier (open, least saturated) → new mounted hypothesis  │
  └───────────────────────────────────────────────────────────────────┘
```

**The RSI essence.** Improving the *task* solution (the editable file) is the fast loop's job; that alone is auto-research, not self-improvement. The system improves *itself* when the improver edits `state/strategy.py` under guard: the proposer **re-reads the strategy every cycle**, so a promoted edit literally shapes the next proposal. Arbitration (Stage2-lite keep-rate comparison) decides `kept` vs `reverted`; only a `kept` edit compounds.

**The trust boundary (C/F split).** The improver may rewrite only the proposer-side strategy artifact (`rsi.selfedit.guard.EDITABLE_SURFACES`). The gates, diagnostics, platform, and task scorer — the parts that *measure* success — are frozen by construction. The loop can never rewrite its own grader; a test asserts an edit targeting `rsi/evaluation/gates.py` is rejected.

## Package map (what each module is and does)

| Module | Responsibility | Key types / functions |
|---|---|---|
| `rsi.core.engine` | The three-loop orchestrator + composition root | `RSIEngine`, `EngineConfig`, `build_default_engine()` |
| `rsi.core.pipeline` | One experiment cycle, pure orchestration | `run_cycle()` (stage → dispatch → read → gate) |
| `rsi.core.types` | Fast-loop data types | `Proposal`, `RunRecord`, `EfficiencyReport`, `build_efficiency_report()` |
| `rsi.core.proposer` | The Proposer seam + strategy artifact IO | `Proposer` (Protocol), `HeuristicProposer`, `derive_knobs()`, `value_grid()`, `ensure_strategy_file()`, `load_strategy()` |
| `rsi.core.improver` | The SelfImprover seam (system-level half of RSI) | `SelfImprover` (Protocol), `NoopImprover`, `NaiveImprover`, `ImprovementProposal` |
| `rsi.core.ledger` | Append-only self-edit lifecycle log | `EditLedger`, `EditRow`, transitions `applying→observing→kept\|reverted` |
| `rsi.knowledge.platform` | In-process facade over the entity stores (the 7-step write chain) | `KnowledgePlatform` |
| `rsi.knowledge.question_graph` | RQ-graph ops + frontier | `create/spawn/refine/merge/transition/attach/update`, `frontier()` |
| `rsi.knowledge.store` | Sqlite storage | `SQLiteStore` (versioned entities, FTS/LIKE), `EvidenceLog` (append-only) |
| `rsi.knowledge.concepts` | Dataclass twins of the frozen schemas | `Question`, `Hypothesis`, `MachineEvidence`, `Insight`, `KnowledgeNode`, `ChangeProposal` |
| `rsi.knowledge.schemas/` | The frozen concept contracts (JSON-Schema YAML) | I-Q, I-H, I-E, I-S, I-K, I-X, ChangeProposal |
| `rsi.knowledge.statemachine` | Frozen transition tables | `QUESTION_TRANSITIONS`, `HYPOTHESIS_TRANSITIONS`, `legal()` |
| `rsi.knowledge.errors` | Frozen error-code vocabulary | `PlatformError`, `ERROR_CODES` |
| `rsi.evaluation.benchmark` | Epoch freeze/upgrade (middle loop anchor) | `Benchmark`, `BenchmarkRef`, `EpochManager`, `BenchmarkUpgrader` (Protocol), `SeedBumpUpgrader` |
| `rsi.evaluation.gates` | Verdict gates (frozen trichotomy) | `Gate` (Protocol), `GateChain`, `CollapseGate`, `NoChangeGate`, `NoiseGate`, `MetricReading`, `Verdict` |
| `rsi.evaluation.diagnostics` | Middle-loop diagnosis | `Diagnostician` (Protocol), `DiagnosisReport` (with `mature`), `NaiveDiagnostician` |
| `rsi.tasks.spec` | Task-bundle contract | `TaskSpec`, `TaskMetric`, `load_task()`, `read_result()` |
| `rsi.tasks.runner` | Staging + subprocess dispatch | `TaskRunner` (Protocol), `LocalRunner`, `stage_workdir()` + scope guard, `apply_params()`, `parse_params()` |
| `rsi.llm.base` | Provider abstraction (verbatim from source) | `LLMProvider` (Protocol), `NullProvider`, `load_provider()` |
| `rsi.llm.echo` | Keyless deterministic provider | `EchoProvider` |
| `rsi.llm.openai_compat` | Optional real provider skeleton | `OpenAICompatProvider` |
| `rsi.selfedit.sentinels` | Mutable-region markers + parser (trust-boundary mechanics) | `split_regions()`, `replace_mutable()` |
| `rsi.selfedit.guard` | Edit pre-filter, layers 0–4 | `check_candidate()`, `EDITABLE_SURFACES`, `GuardVerdict` |
| `rsi.config` | Optional YAML config (registry allowlist, no credentials) | `get()`, `set_config_path()` |

## One cycle, as data

1. **Propose** — `HeuristicProposer.propose()` reads `state/strategy.py` (the self-editable surface), derives knobs from the task's editable file (`derive_knobs` / `parse_params`), walks `value_grid()` in strategy priority order, and emits a `Proposal`: one `variable_changed` (e.g. `model.CLIP`), one `params` dict, the `benchmark_ref` at submission.
2. **Implement** — `stage_workdir()` copies the pristine task into `out/runs/cycle-NNN/`, applies params on top of the incumbent bytes (keeps accumulate), enforces the scope guard (only the editable file may differ; it must compile).
3. **Run** — `LocalRunner.dispatch()` runs the entrypoint subprocess with `RSI_EPOCH` + `RSI_BENCHMARK_SEED` injected (the epoch-freeze carrier), wall-clock timeout, captured `run.log`. A post-run check fails the cycle if the task rewrote its own editable file.
4. **Finalize** — `read_result()` parses the result file into a metric or a `failure_mode`; the `GateChain` decides `keep | discard | fail` (gates reject, they do not bless). Keep ratchets the incumbent (`incumbent_bytes/value`) and the question's `best_result`.
5. **Analyze** — the engine projects the `RunRecord` into machine evidence (`MachineEvidence` with links from the frozen link policy), pushes it append-only + idempotent, aggregates hypothesis confidence (Σsupports − Σrefutes, no version bump), and fires count-gated transitions (supported / net-refuted).

Middle-loop ticks add: per-(question, epoch) diagnosis → saturation write-back (mature reports only) → epoch upgrade + re-baseline on recommendation → efficiency report → self-improvement arbitration or a new guarded edit.

## Runtime artifacts (under `--out`)

| Path | Contents |
|---|---|
| `state/runs.jsonl` | One `RunRecord` per cycle (`run_id`, `epoch`, `status`, `metric{name,direction,value,baseline,delta,failure_mode}`, `variable_changed`, `params`, `verdict_reasons`) |
| `state/strategy.py` | The SELF-EDITABLE strategy artifact (sentinel region; re-read every cycle) |
| `state/ledger.jsonl` | Append-only self-edit rows (status lifecycle + `applied_at_cycle` / `terminated_at` + prior mutable bytes for byte-revert) |
| `state/epochs.json` | Epoch history: `{current, epochs: [{name, epoch, task_version, seed}]}` |
| `kb/questions.sqlite` | I-Q nodes (statement, scope, benchmark_ref, best_result, uncertainty, saturation, status, typed edges, version) |
| `kb/hypotheses.sqlite` | I-H nodes (mounts, testability, confidence + algorithm, evidence_counts, status, version) |
| `kb/evidence.sqlite` | Append-only I-E records (machine evidence + links + replay identity) |
| `kb/insights.sqlite`, `kb/nodes.sqlite` | I-S insights and I-K theory nodes (schema-defined surfaces for homework) |
| `kb/meta.sqlite` | Persisted ChangeProposal idempotency keys (restart-safe replays) |
| `runs/cycle-NNN/` | Staged workdir: the task copy as run + `run.log` + `result.json` |
| `summary.json` | Engine end-of-run summary (counts, improvements, questions, epochs, best) |

## Entry points & configuration

```
python -m rsi.core --task <dir> --out <dir> [--cycles N] [--middle-every N]
                   [--seed N] [--improver naive|noop] [--question "..."]
python -m rsi.knowledge --kb <dir> questions|frontier [--max-saturation F]|
                                 question <id>|hypothesis <id>|
                                 evidence [--run id|--target id]|insights
python -m rsi.evaluation --runs <out>/state/runs.jsonl report
python scripts/report.py --session main=<out> [--session control=<out>] \
       --task <name> --metric <name> --direction min|max --dest <dir>
```

Configuration: engine runs are driven by CLI flags / `EngineConfig` (the authoritative path). `config.yaml` is optional and currently consulted only by `llm.load_provider` (`llm.provider`); the registry in `rsi/config.py` declares every key the framework may ever read, and credential-shaped keys are refused by construction.

## Layering and dependency rule

```
config ← llm ← selfedit ← tasks ← knowledge ← evaluation ← core
```

- `rsi.evaluation` library code imports **nothing** from `rsi.knowledge` or `rsi.core`: its interfaces are frozen and consume plain `RunRecord` sequences, so instructor-arranged evaluation tasks can drive them independently of storage. (`rsi.evaluation.__main__` is a CLI adapter.)
- `rsi.core` is the only composition point: `build_default_engine()` wires concrete (naive) implementations into `RSIEngine`, which holds protocols only. Swap strategies by replacing that wiring — never the loop.
- Pluggable seams are `typing.Protocol`s with a `load_*`/registry shape (OCP); providers/runners are substitutable (LSP); Gate / Diagnostician / Upgrader are separate small interfaces (ISP); the engine depends on abstractions only (DIP).

## Task bundles

A research task is a directory with a `program.md` manifest (YAML frontmatter) declaring `entrypoint`, `metric {name, direction, result_file, result_key}`, **exactly one `editable` file**, optional `readonly`/`validate`/`time_budget`. The harness never imports task code: it stages an isolated copy per cycle, runs the entrypoint as a subprocess with the benchmark identity injected, and reads the metric from the result file only. A pre-run scope guard asserts that only the editable file differs from the pristine bundle; a post-run check fails the cycle if the task mutated its own editable.

## Intentional divergences from the source systems

| Divergence | Source behavior | Base behavior | Why |
|---|---|---|---|
| No MCP transport | five stdio MCP servers, cross-server validation edges | one in-process `KnowledgePlatform` facade | teaching size; tool semantics preserved |
| No L0/L1/L2 risk gates | ChangeProposal risk classification + human review queue | risk recorded, everything auto-applies | single-writer course setting |
| Replay pair vs triple | evidence requires `run: {commit, docker_image, config_hash}` | `run: {revision, source_hash}` | no container fleet; replay concept preserved |
| Byte-revert vs git-revert | self-edits committed to a local branch, reverted via git | prior mutable-region bytes in the ledger | no git dependency; edits confined to one region of one file |
| No attestation | executor-attested result provenance | untrusted self-reported result.json | homework task R-1 |
| `update` op added | ChangeProposal ops: create/refine/merge/transition/attach_evidence/spawn | + `update` (in-place field writes) | the middle loop must write diagnosis back to I-Q |
| Evidence metric.value nullable | number required | `number \| null` | a failed run carries no value |
| TaskSpec vs EnvSpec | `envs/<name>/program.md`, `share`, regimes, multi_editable… | `tasks/<name>/program.md`, single-variable core fields | course scope |
| I-Q records schema-complete | stored records carried extra unvalidated keys | `iq.yaml` declares `evidence_refs`/`conversion_events`; records are strictly schema-valid | strict readers/exporters |

## Determinism

Run ids are cycle-derived (no clock in identities), task noise/seeds come from `RSI_BENCHMARK_SEED`, `EchoProvider` is a pure function of the prompt, edit ids are content-addressed, epoch ids are sequential. The e2e test asserts two same-seed runs produce identical summaries and metric sequences. (Timestamps exist only in informational fields.)
