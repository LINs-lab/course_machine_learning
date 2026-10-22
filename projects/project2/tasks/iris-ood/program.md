---
schema_version: 1
name: iris-ood
entrypoint: ["python3", "run.py"]
time_budget: 120
metric:
  name: ood_accuracy
  direction: max
  result_file: result.json
  result_key: ood_accuracy
editable: ["model.py"]
readonly: ["run.py", "iris.csv"]
validate: ["python3", "run.py", "--self-check"]
---
# iris-ood — OOD generalization on iris

Plain iid accuracy on iris is saturated (≈100% for decades) and has no discrimination. This task measures **out-of-distribution generalization**: the model trains on an in-distribution split and is scored on a held-out split whose features have been warped by a seeded monotone per-feature transformation (power + scale). The warp family is published; its seeded parameters are not, and they change with the benchmark epoch.

## Task protocol

- Optimize **`ood_accuracy`** (higher is better). The entrypoint writes `result.json[ood_accuracy]`; `id_accuracy` is reported for context (the id/ood gap is the interesting quantity).
- The **only** editable file is `model.py` (contract: class `Model` with `fit(X, y)` / `predict(X)`, pure python, no imports).
- The model must never receive the warp parameters — robustness must come from the model's own structure (preprocessing, invariances, architecture), not from fitting the evaluator's seed.
- One coherent change per run (single-variable control).
- `time_budget` is fixed; a crash or non-finite outcome scores 0.0.
- Split: 90 train / 15 id / 45 ood (stratified, seeded by the benchmark).

## Research direction hints (not requirements)

The response surface has three regimes (empirical, seed-7):

- baseline (raw features): **≈0.33** — chance level; the warp fully breaks the linear model;
- preprocessing knobs (single-variable control): **≈0.87–0.91** — robust statistics (winsorization) and power-matching each recover a large part;
- structural rewrite of `model.py`: **≈0.93–0.96** — note `predict(X)` receives the whole test batch at once, so transductive normalization (per-batch rank/quantile features) is invariant to ANY strictly monotone per-feature warp. This is the fast-loop-knob vs full-rewrite distinction the course studies, made measurable.
