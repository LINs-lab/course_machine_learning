# Autoresearch-Base

A minimal **recursive self-improvement (RSI) research harness** with a **knowledge platform**, distilled from two production systems into one small, teachable Python package. It is the starting point for a course project: the skeleton runs end-to-end (keyless, CPU, seconds) — the actual improvement **strategies are yours to write** (see `docs/homework.md`).

## What it is

A system that, given one **research task bundle** (a folder with a task manifest and code to improve, e.g. an ML model or an optimizer), repeatedly:

1. **Fast loop** — proposes, implements, runs, and analyzes single-variable experiments against a _frozen_ benchmark (`propose → implement → run → finalize → analyze`);
2. **Middle loop** — aggregates evidence across rounds, diagnoses saturation / noise / discrimination, freezes the benchmark within an evaluation epoch, and upgrades it across epochs;
3. **Slow loop** — evolves a typed **Research Question Graph** (refine / decompose / spawn / merge / challenge) from evidence;
4. **System-level self-improvement** — improves its _own_ proposal strategy through guarded, revertible edits. An edit is only real when it is **promoted and re-enters** the next round's proposals — that recursion is what makes it RSI rather than a pile of experiments. The scorer is never editable (trust boundary).

## Install (conda)

```bash
conda env create -f environment.yml      # python 3.11 + pip install -e .[dev,report]
conda activate autoresearch-base
pytest tests/ -q                         # 173 tests, keyless
```

Pip fallback: `python -m venv .venv && . .venv/bin/activate && pip install -e .[dev,report]`.

## Quickstart (no API keys needed)

Run these from the repo root (the `--task` paths are relative):

```bash
# 6 research cycles on the demo task (ill-conditioned quadratic, CPU, seconds)
python -m rsi.core --task tasks/quad-optim --out out --cycles 6 --seed 7

# the OOD-generalization research task (iris under a seeded feature warp)
python -m rsi.core --task tasks/iris-ood --out out-iris --cycles 16 --seed 7

# the five public evaluation tasks (frozen scorers; each bundle runs in seconds)
for t in battle-of-sexes prisoners-dilemma colonel-blotto house-price breakout-lite; do
  python -m rsi.core --task tasks/$t --out out-$t --cycles 24 --seed 7
done

# inspect what the system learned
python -m rsi.knowledge --kb out/kb frontier
python -m rsi.knowledge --kb out/kb evidence
python -m rsi.evaluation --runs out/state/runs.jsonl report
cat out/summary.json

# a full visual report (trajectory, self-improvement ledger, question
# graph, optional with/without-improver comparison). Figures are vector
# PDF only (Times New Roman, fonttype 42 — the academic figure-production
# convention; purple is reserved for the RSI system). The question graph
# uses graphviz's hierarchical layout when `dot` is available (it is in
# environment.yml) and a built-in fallback otherwise.
python scripts/report.py --session main=out-iris --task iris-ood \
    --metric ood_accuracy --direction max --dest report-iris
```

A reference long run (292 cycles + a no-improver control, with analysis) is committed under `experiments/2026-10-11-iris-ood/`.

The five public tasks each run in seconds; the fixed comparison protocol — pristine baseline, RSI self-editing ON/OFF, and the Claude Code strong baseline — is tabulated in `baselines/public_tasks_baseline.md` (figure: `public_tasks_baseline.pdf`).

### What you should see (self-check)

The first demo ends with a JSON summary on stdout and `out/summary.json`:

```json
{
  "task": "quad-optim", "cycles": 6, "keep": 3, "discard": 3, "fail": 0,
  "improvements": {"ed-…": "kept"},
  "questions": ["quad-optim-q0", …],
  "best": {"quad-optim-q0": {"evidence_id": "…", "score": 2.48e-4, "epoch": "e001"}, …}
}
```

If `keep >= 1`, `improvements` shows at least one edit reaching a terminal state, and the two inspection CLIs print rows, the system is healthy.

### Troubleshooting

- `--out already holds a session`: session dirs are single-use — use a fresh `--out` or delete the old one.
- `ModuleNotFoundError: matplotlib` when running `scripts/report.py`: install the report extra (`pip install -e .[report]`). On a machine where the conda env was created by an older version of this repo, refresh it instead of reinstalling from scratch: `conda env update -f environment.yml --prune`.
- `TaskError: task at tasks/quad-optim has no program.md`: you are not in the repo root — the quickstart paths are relative.
- Everything is deterministic: same `--seed` ⇒ same summary and metric sequence. If two runs differ, check that the task bundle wasn't edited.
- Requires Python ≥ 3.10. Only `pyyaml` and `jsonschema` are runtime deps; the demo never needs network access or API keys.

Everything lands under `--out`:

```
out/
├── state/strategy.py    # the SELF-EDITABLE strategy artifact (sentinel region)
├── state/runs.jsonl     # one RunRecord per cycle
├── state/ledger.jsonl   # self-improvement ledger (applying→observing→kept|reverted)
├── state/epochs.json    # benchmark epoch freeze/upgrade history
├── kb/*.sqlite          # the knowledge platform (questions, hypotheses, evidence, insights, theory)
└── runs/cycle-NNN/      # staged, isolated per-cycle workdirs
```

## Package map

| Module                      | Responsibility                                                                                                                                                 |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rsi.core`                  | `RSIEngine` — the three loops + self-improvement re-entry; pipeline, proposer, improver, ledger                                                                |
| `rsi.knowledge`             | Knowledge platform: concept contracts (I-Q/I-H/I-E/I-S/I-K), question graph, sqlite stores, `KnowledgePlatform` facade                                         |
| `rsi.evaluation`            | Frozen evaluation interfaces: benchmark epoch freeze/upgrade, verdict gate chain, diagnostics                                                                  |
| `rsi.tasks`                 | Task-bundle contract (`program.md` manifest) + local runner with scope guard                                                                                   |
| `rsi.llm`                   | `LLMProvider` protocol; `EchoProvider` (deterministic, keyless) and an optional OpenAI-compatible skeleton                                                     |
| `rsi.selfedit`              | Sentinel regions + the edit guard (the executable trust boundary)                                                                                              |
| `tasks/quad-optim`          | Demo task: optimizer design on an ill-conditioned quadratic                                                                                                    |
| `tasks/iris-ood`            | Research task: OOD generalization on iris (seeded monotone feature warp)                                                                                       |
| `tasks/<five public tasks>` | The public evaluation tasks (`battle-of-sexes`, `prisoners-dilemma`, `colonel-blotto`, `house-price`, `breakout-lite`); their scorers are frozen               |
| `baselines/`                | Public baseline exhibit: pristine vs RSI ON/OFF vs Claude Code strong baseline (`public_tasks_baseline.md` / `.pdf`)                                           |
| `scripts/`                  | `report.py` (session report), `compare_seeds.py` (multi-seed ablation), `probe_task_knobs.py` (single-knob probe), `llm_proposer_demo.py` (C-1 worked example) |

## Where to go next

- **`ASSIGNMENT.md` — the course assignment brief: start here if you are a student**
- `docs/architecture.md` — the three-loop design + module-level functional reference
- `docs/knowledge-platform.md` — the concept definitions and the 7-step write chain
- `docs/rsi-core.md` — the C/F split, trust boundary, edit lifecycle
- `docs/evaluation.md` — the frozen evaluation interfaces (the five public tasks are the evaluation; scorers never move)
- `docs/homework.md` — the strategy task catalog: every `TODO(homework)` seam, with acceptance hints
- `baselines/public_tasks_baseline.md` — the fixed comparison protocol: pristine vs RSI ON/OFF vs the Claude Code strong baseline on the five public tasks
- `submission/` — the NeurIPS-format report template; its page 1 carries the grading rubric

## Provenance

Ported and unified (semantics preserved) from two larger systems: `project_collab_auto_research` (the experiment harness and self-improvement machinery) and `project_collab_auto_research_shared_kb` (the knowledge-platform contracts). Every ported file carries a `# Ported (semantics preserved) from ...` header; intentional divergences are listed in `docs/architecture.md`.
