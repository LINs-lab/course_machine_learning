# Course Assignment: Build the Strategies of a Recursive Self-Improvement System

This repository is a **working** recursive self-improvement (RSI) research system — with deliberately **naive** strategies. The machinery is real and verified; the intelligence is missing. **Your assignment is to build that intelligence** and to prove, with experiments, that your system genuinely improves itself.

## 1. The system you are given

An engine that, given a research task bundle (a folder describing a task and the code to improve), runs three nested loops:

```
FAST   propose → implement → run → finalize → analyze single-variable experiments on ONE research question, under a FROZEN benchmark epoch
MIDDLE every N cycles: diagnose saturation/noise, freeze the benchmark within an epoch, upgrade it across epochs, and arbitrate self-improvement edits
SLOW   research questions as a typed-edge graph: hypotheses succeed each other on one question; saturated/exhausted questions spawn children
```

…plus the RSI half: the system may rewrite its **own proposer strategy** inside a guarded sentinel region of `state/strategy.py`. A guarded edit goes `applying → observing → kept|reverted`; a `kept` (promoted) edit **re-enters immediately**, because the proposer re-reads the strategy every cycle. That round trip — waste signal → guarded self-edit → re-entry → measured yield → promotion — is what "recursive self-improvement" means in this course.

Everything runs **without API keys**, deterministically, on CPU, in seconds:

```bash
conda env create -f environment.yml && conda activate autoresearch-base
python -m rsi.core --task tasks/iris-ood --out out --cycles 16 --seed 7
```

## 2. What you get vs what you build

**Given (works, but naive):**

- The full three-loop engine and the self-improvement lifecycle (`applying → observing → kept|reverted` + byte-level revert)
- The knowledge platform: frozen concept contracts (I-Q/I-H/I-E/I-S/I-K), the question graph, append-only evidence, confidence aggregation
- The frozen evaluation interfaces: epoch freeze/upgrade, the `keep | discard | fail` gate chain, the diagnosis report
- **Five public evaluation tasks** — `tasks/battle-of-sexes`, `tasks/prisoners-dilemma`, `tasks/colonel-blotto`, `tasks/house-price`, `tasks/breakout-lite` — each a self-contained bundle with one editable file, one frozen gate metric and a fixed budget; the course grade is measured on these five
- Two development task bundles: `tasks/quad-optim` (seconds) and `tasks/iris-ood` (OOD generalization with a real headroom gradient)
- A report generator and a **reference experiment** with analysis: `experiments/2026-10-11-iris-ood/` — study it first; it shows the machinery working and exactly where the naive strategies plateau
- A **public baseline exhibit** (`baselines/public_tasks_baseline.md` / `.pdf`): pristine baseline, RSI self-editing ON/OFF, and a **Claude Code strong baseline** — 3 independent 20-minute sessions per task, frozen seed 7, harness-verified scores. The shipped RSI arms show **no measurable ON/OFF gap**; Claude Code leads on all five tasks. Your system must show a real self-editing gain and, ideally, close part of that gap.
- Naive-but-walking implementations of every strategy seam

**Yours to build** (the assignment's core, `TODO(homework)` in code, catalog in `docs/homework.md`):

| Area               | What a real strategy must do (naive status quo)                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| C-1 Proposer       | rank/explore using KB evidence and hypotheses; smarter value grids; optionally LLM-composed edits (today: fixed grid walk, ignores evidence) |
| C-2 SelfImprover   | turn waste signals into strategy edits that measurably raise research yield (today: rotates one list)                                        |
| E-3/E-2 Evaluation | honest noise floors via seed replication, significance and anti-shortcut gates, calibrated saturation (today: sliding-window heuristics)     |
| K-1 Slow loop      | principled question-graph evolution: decompose/depend/challenge/transfer (today: template spawn)                                             |
| E-1 Benchmarks     | harden across epochs: harder instances, held-out splits (today: seed bump)                                                                   |
| C-3/C-4/R-1 Rigor  | guard judge, hardened arbitration (meltdown brake), result provenance                                                                        |

The **RSI design is the deliverable**: your own approach to *how the system improves its own strategy* (proposal policy, self-edit search, arbitration), built on the given knowledge-graph architecture. The *evaluation* of the five public tasks — scorers, metrics, seeds, budgets — is frozen and shared by every submission, which is what makes results comparable across students and against the Claude Code baseline.

## 3. The rules (frozen — violating them invalidates the work)

1. **The trust boundary is enforced, not advisory.** Your improver may edit only `state/strategy.py`'s sentinel region. The guard (layers 0–4) and the test suite check this; a strategy that touches the scorer, gates, platform, or task files **by design or by accident fails loudly** — and deliberately circumventing it is academic misconduct.
2. **Frozen contracts.** The YAML schemas, the state machines, the `keep | discard | fail` trichotomy, and the `rsi.evaluation` interfaces do not change. You implement _strategies_ behind existing seams (`typing.Protocol`s), wired in `build_default_engine()` — you do not fork the engine loop or re-spec the knowledge platform.
3. **Evaluation is frozen; the strategy is yours.** The five public tasks *are* the evaluation: their `run.py` scorers, metrics, seeds and budgets are not yours to change (graded runs use pristine copies). Each bundle keeps exactly ONE editable file, one gate metric, a fixed time budget and a black-box entrypoint. You change *how the system searches and improves itself* — the strategies behind the `typing.Protocol` seams — never *how it is scored*.
4. **Determinism is a feature.** Same seed ⇒ same results. Your strategies must keep this property (no wall-clock/randomness in identities or decisions without a seeded source).
5. The demo must stay **keyless**. LLM providers may be wired via the `rsi.llm` seam with credentials in environment variables only.

## 4. How to work

1. **Setup & reproduce.** Install, run the suite (`pytest tests/ -q`, all green), then reproduce the reference experiment — `experiments/2026-10-11-iris-ood/ANALYSIS.md` §1 has the exact commands (two `python -m rsi.core` runs at `--cycles 292 --seed 7`, improver `naive` vs `noop`, then `scripts/report.py`) — and read `ANALYSIS.md` fully: it names every weakness you are about to attack.
2. **Pick a seam and baseline it.** Start with C-1 (the proposer — the biggest visible effect). Measure the naive baseline first on the five public tasks, keep the summaries.
3. **Implement behind the Protocol.** Wire in `build_default_engine` (or add a config-driven registry). Keep the public interfaces; add tests for your behavior — the existing suite must stay green.
4. **Experiment, don't assert.** Use `scripts/report.py` with a `--session control=` of the naive strategy at the same seed. A strategy is better only if the numbers say so: final best, keep-rate, best-over- time curve, epochs/questions dynamics.
5. **Compare against the strong baseline.** `baselines/public_tasks_baseline.md` fixes the protocol (task seed 7, first 20 minutes of each session, harness-verified best); report your system next to the pristine baseline, the RSI ON/OFF ablation and the Claude Code baseline on all five public tasks.
6. **Iterate across seams.** The recommended order and acceptance hints are in `docs/homework.md`.

## 5. Deliverables

1. **Code**: your self-improving RSI (at minimum a real Proposer and a real SelfImprover), wired, tested, and runnable on the five public tasks.
2. **Experiments on all five public tasks**: for each task, your RSI (self-editing ON) and the same system with self-editing OFF — same seed, same budget — with session reports (`scripts/report.py`) and the naive-strategy control; results tabulated next to the Claude Code baseline from `baselines/public_tasks_baseline.md`.
3. **Report** (`submission/`, the NeurIPS-format template, ≤ 9 pages of body): your design, the measurements on the five tasks, and the causal story of at least one promoted self-edit — what changed in `state/strategy.py`, what the proposer did differently afterwards, and how the yield changed (keep-rate/best/over time). The report is graded by the rubric printed on page 1 of `submission/main.pdf` (survey 10%, reasoning & insight 30%, experiments 30%, result analysis 30%).

## 6. How it is evaluated

- **Interfaces**: your strategies are driven through the frozen seams on the five public tasks, and may also be evaluated on held-out bundles you have not seen. Hard-coding task-specific answers is visible and scored accordingly.
- **Metrics**: final best result per task at the frozen budgets; research efficiency (keep-rate, wasted cycles); — the RSI core — measurable yield from promoted self-edits relative to a no-self-editing control; and the gap to the Claude Code baseline.
- **Report**: graded on the four rubric components (survey 10% / reasoning & insight 30% / experiments 30% / result analysis 30%), as printed on page 1 of `submission/main.pdf`.
- **Integrity**: the trust boundary, the frozen scorers and determinism hold under your code; the full test suite passes.

## 7. FAQ

**Can I edit `RSIEngine.run`?** No — the loop structure is the frozen frame. Change behavior through the Protocol seams.

**Can my improver edit files other than `state/strategy.py`?** No. The sentinel surface is the only self-editable file; guard rejects anything else. Richer self-edit spaces live _inside_ the sentinel region (you own `STRATEGY_TEMPLATE` — shape-locked statements, any values).

**Can I add dependencies?** Prefer stdlib. A new dependency needs a clear justification and must keep the keyless path working (extras are OK).

**May I change a public task's `run.py`, metric or budget?** No. The five public tasks are the evaluation, and graded runs use pristine copies; changing a scorer to look better is reward hacking and scores zero. You may *add* your own task bundle for development (next question).

**Can I make my own task bundle?** Yes — see homework R-2 and `docs/architecture.md` §Task bundles. It is excellent practice and a likely bonus direction.

**What does "naive works but isn't effective" mean concretely?** See `experiments/2026-10-11-iris-ood/ANALYSIS.md` §3: the machinery (re-entry, arbitration, protection) demonstrably functions, while the main/control trajectories coincide. Your grade is largely the distance you open between those two lines — honestly measured.
