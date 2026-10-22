---
schema_version: 1
name: colonel-blotto
entrypoint: ["python3", "run.py"]
time_budget: 60
metric:
  name: value_rate
  direction: max
  result_file: result.json
  result_key: value_rate
editable: ["strategy.py"]
readonly: ["run.py"]
validate: ["python3", "run.py", "--self-check"]
---
# colonel-blotto v2 — value-weighted concentration against a random opponent

Colonel Blotto with 12 battlefields of **unequal, published value**
(1,1,2,2,2,3,3,3,4,4,5,5 — total 35) and a budget of 120 soldiers. The
winner of each field takes its value (ties split it). The opponent's
allocation procedure is **published in `run.py`** (READONLY): mid-spread
random weights (sd ~4.1 per field, mean 10).

## Task protocol

- Optimize **`value_rate`** (higher is better): fraction of the total field
  value won, averaged over 300 seeded games (scale [0, 1]).
- The **only** editable file is `strategy.py` (contract: `allocate(rng) ->
  list of 12 non-negative numbers summing to <= 120`). Everything else is
  read-only.
- One coherent change per run (single-variable control).
- `time_budget` is fixed; a contract violation or non-finite outcome scores 0.0.

## Research direction hints (not requirements)

- baseline (even split, 10 per field): **≈0.51** — coin flips everywhere;
- concentration knobs: **≈0.62** — the v3 opponent counters the agent's
  top value-weighted fields after a 40-game warmup, so static concentration
  gets punished; rotation (`ROTATE_EVERY`) only pays off in combination,
  and several knobs are decoys;
- structural rewrite of `strategy.py`: the opponent's per-field distribution
  is published, so a value-density allocation (force ∝ field value × win
  probability at margin) is computable directly and beats any knob sweep.
