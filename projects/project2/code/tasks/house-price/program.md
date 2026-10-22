---
schema_version: 1
name: house-price
entrypoint: ["python3", "run.py"]
time_budget: 120
metric:
  name: rmse
  direction: min
  result_file: result.json
  result_key: rmse
editable: ["model.py"]
readonly: ["run.py", "houses.csv"]
validate: ["python3", "run.py", "--self-check"]
---
# house-price — tabular regression with three published pain points

Predict price on a frozen 900-row housing dataset (9 features — 3 are traps). The
train/test split (70/30) is seeded by the harness (`RSI_BENCHMARK_SEED`) and
changes across evaluation epochs. The data has three published pain points:
a heavy-tailed `area` (lognormal with 5% mansion outliers), a piecewise
`dist_cbd` response (urban cliff at 8 km), and
a saturating `age` depreciation. The noise floor of the generative process is
σ ≈ 10k; `junk1` is pure noise, `junk2` is a collinear duplicate of `area`,
`junk3` is heavy-tailed — they are traps, not signal.

## Task protocol

- Optimize **`rmse`** (lower is better) on the held-out 30% split; crashes
  and non-finite outcomes score 1e12 (worst possible).
- The **only** editable file is `model.py` (contract: class `Model` with
  `fit(X, y)` / `predict(X)`, pure Python, stdlib imports allowed).
  `houses.csv` and `run.py` are read-only.
- One coherent change per run (single-variable control).
- `time_budget` is fixed.

## Research direction hints (not requirements)

- baseline (raw features, degree 1, no caps): **≈14.5k RMSE** — the convex
  distance term, the saturating age effect, and the area×bedrooms
  interaction are fit by a line; four junk columns are traps;
- representation knobs (single-variable control): polynomial degree is the
  lever — degree 2 reaches **≈12.8k**, degree 3 **≈11.7k**; beware: capping
  (`WINSOR_Q`) and `LOG_AREA` HURT here (the response keeps rising in area —
  evidence should reject them);
- structural rewrite of `model.py`: the response is additive in transformed
  features — a rewrite that models `1/(dist+1)` and saturating age explicitly
  can approach the noise floor (≈10k) and beats any polynomial order on the
  tail behavior.
