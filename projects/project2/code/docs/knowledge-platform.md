# The Knowledge Platform

The knowledge platform is the system's long-term memory: a typed graph of research questions, hypotheses mounted on them, append-only machine evidence, experiential insights, and a theory layer. It is a definition-only skeleton — **no data ships with the package**; a session's knowledge lives under `<out>/kb/*.sqlite`.

The concept contracts are frozen JSON-Schema (draft 2020-12) files in `src/rsi/knowledge/schemas/` — they are the single source of truth. The dataclasses in `concepts.py` mirror them 1:1, and contract tests assert the two never drift.

## Concepts

### I-Q — Question (`schemas/iq.yaml`)

The node of the **Research Question Graph**. Each question binds:

- `statement`, `scope` — what is being asked and where it applies;
- `benchmark_ref {name, epoch}` — the frozen benchmark version it is asked under (rebound on epoch upgrades);
- `best_result {evidence_id, score, epoch}` — the current best, with provenance; `epoch` (base extension) makes the ratchet epoch-aware — cross-epoch scores are never compared;
- `uncertainty []` — open uncertainties (spawn material for the slow loop);
- `saturation` ∈ [0,1] — written by the middle loop's diagnostician;
- `status`: `open → partially_answered → answered`, plus `refined | merged | retired` (frozen transition tables in `statemachine.py`; the evidence-requiring transitions must carry a `conversion_event` with `evidence_refs`);
- `edges []` — the typed directed edges of the question graph: `depends_on | refines | challenges | merged_into | spawns`.

### I-H — Hypothesis (`schemas/ih.yaml`)

- `derived_from` — ≥1 mount on `I-Q` (question) or `I-K` (theory); **an unmounted hypothesis is mechanically impossible** (dangling-mount writes are rejected);
- `testability {procedure, expected_evidence, falsifiable_by}` — all-or-nothing; a hypothesis without a falsifier cannot be created;
- `confidence` — aggregated from evidence (below), never hand-set;
- `status`: `draft → active → supported | refuted | merged`, where `supported` needs ≥2 supporting and `refuted` ≥1 refuting evidence links (count-gated through the state machine).

### I-E — Evidence (`schemas/ie.yaml`)

Append-only. Two kinds; the engine produces **machine** evidence: the projection of one finished run (`run_result`: run_id, env, status `keep|discard|fail`, metric, benchmark_ref, failure diagnosis) plus:

- `links []` — `{type: supports | refutes | inconclusive, to: <hypothesis or question id>, strength, confidence}`. The engine's link policy:

  | verdict | metric relation | link |
  |---|---|---|
  | keep | delta > noise floor | `supports` (strength ∝ relative gain) |
  | discard | delta ≤ 0 | `refutes` (0.5) |
  | discard | 0 < delta ≤ noise floor | `inconclusive` |
  | fail | any | `inconclusive` + failure.class |
  | baseline run (no incumbent yet) | — | `inconclusive` (informs, does not test) |

- `run {revision, source_hash}` — the replay identity of the code that produced the number (base adaptation of the source replay triple);
- `replay {replayable, replay_of}` — corrections append a new record pointing back; **no update/delete paths exist by design**.

Idempotency: pushing the same `run_id` again raises `RSI_IDEMPOTENT_REPLAY` carrying the first evidence id — a run counts once, ever (evidence-before-notify).

### I-S — Insight (`schemas/is.yaml`)

Experiential knowledge mounted on questions/hypotheses/theory (`applicability` scopes where it applies). Insights bias future proposals but **never enter the evidence chain**.

### I-K — Knowledge / theory (`schemas/ik_theory.yaml`)

A theory graph: `draft | established` statements with typed edges (`contains | rests_on | specializes | tensions`) and a 0–2 depth level. Hypotheses may mount on theory nodes (`derived_from: I-K`).

### ChangeProposal (`schemas/change_proposal.yaml`)

The single write envelope for engine/human writes into I-Q/I-H/I-S: `actor`, `targets [{interface, object_id, base_version?}]`, `op`, `params`, `idempotency_key`. `base_version` gives optimistic concurrency (`RSI_VERSION_CONFLICT`, retryable); the idempotency key replays the first result instead of double-applying, and keys are PERSISTED (`kb/meta.sqlite`) so a restart cannot re-apply an envelope. The base system auto-applies (see architecture.md, divergences).

## The 7-step write chain (engine manual, in-process)

```
① create question (or pick from frontier)        platform.create_question / frontier
② create hypothesis (mounted, testability)       platform.create_hypothesis
   draft → active                                 platform.transition_hypothesis
③ submit run proposal (fast loop picks it up)    engine internals
④ execute via the task runner                    rsi.tasks.runner
⑤ push evidence (append-only, idempotent)        platform.push_run
⑥ update confidence + transitions                platform.update_confidence /
                                                   transition_hypothesis/question
⑦ spawn child question (edge extrapolation)      platform.spawn_question → back to ①
```

Step ⑦ returning to ① is the recursion at the knowledge level.

## Storage

One sqlite file per entity kind (`questions/hypotheses/evidence/insights/nodes .sqlite`): JSON blob (schema-validated at write time) + a few indexed columns; entity `put(bump=True)` performs the optimistic version bump; derived updates (confidence aggregation) put **without** bumping. FTS5 full-text search when the sqlite build has it, LIKE fallback otherwise. Evidence is a separate append-only log with a `seq` cursor (`poll`).

## Inspection CLI

```bash
python -m rsi.knowledge --kb out/kb questions
python -m rsi.knowledge --kb out/kb frontier --max-saturation 0.5
python -m rsi.knowledge --kb out/kb question quad-optim-q0
python -m rsi.knowledge --kb out/kb hypothesis quad-optim-q0-h0
python -m rsi.knowledge --kb out/kb evidence --target quad-optim-q0-h0
python -m rsi.knowledge --kb out/kb insights
```
