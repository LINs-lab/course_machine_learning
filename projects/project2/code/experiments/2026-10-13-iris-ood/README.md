# iris-ood self-editing ablation — multi-seed (3 seeds x 2 arms, 8 h each)

The multi-seed follow-up to `2026-10-11-iris-ood/` (single seed 7): does the
self-editing ON/OFF contrast survive seed variation? Six fresh sessions,
all run in parallel against the same endpoint and task.

- arms: **self-editing ON** (`--improver naive`) vs **OFF** (`--improver noop`)
- seeds: **7, 42, 123** (7 matches the archived single-seed run)
- budget: **8 h wall-clock** per session

## Reproducibility & resume

- Every LLM call is made with `temperature: 0` + `seed: <session seed>` in
  the request body, and thinking disabled (Qwen3 chat-template kwarg).
- The full agent context — complete prompt + raw response of every call —
  is persisted twice: `<out>/llm_log.jsonl` (per proposal)
  and `llm_cache.sqlite` (content-addressed, sha256 of
  endpoint+model+method+payload).
- The engine checkpoints loop state at every cycle
  (`state/checkpoint.json`, atomic write). To resume an interrupted
  session, re-run the same `scripts/llm_proposer_demo.py --out <dir> ...`
  command — the checkpoint restores the loop and the LLM cache replays
  identical calls. Verified:
  kill-at-cycle-4 + resume reproduces the uninterrupted 10-cycle trajectory
  bit-for-bit (runs, prompts, responses).

## Run (keyless reproduction)

The archived session directories are not shipped (they are large runtime
state); `reports/` and `figures/` are the final outputs. To run your own
equivalent with the keyless engine, use the deterministic grid-walk
proposer at the same three seeds:

```bash
for s in 7 42 123; do
  python -m rsi.core --task tasks/iris-ood --out out-naive-s$s --cycles 292 --seed $s
  python -m rsi.core --task tasks/iris-ood --out out-noop-s$s  --cycles 292 --seed $s --improver noop
done
```

## Compare (after the runs finish)

```bash
python scripts/compare_seeds.py \
    --on  out-naive-s7  --on  out-naive-s42  --on  out-naive-s123 \
    --off out-noop-s7   --off out-noop-s42   --off out-noop-s123 \
    --task iris-ood --metric ood_accuracy --direction max --dest .
```

(also regenerate each session's own report with `scripts/report.py` —
`main=out-naive-s<N>`, `control=out-noop-s<N>`)
