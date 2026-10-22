---
schema_version: 1
name: breakout-lite
entrypoint: ["python3", "run.py"]
time_budget: 60
metric:
  name: bricks_cleared
  direction: max
  result_file: result.json
  result_key: bricks_cleared
editable: ["policy.py"]
readonly: ["run.py"]
validate: ["python3", "run.py", "--self-check"]
---
# breakout-lite — anticipation under a reactive paddle

A compact Breakout (12 columns × 5 brick rows = 60 bricks, 2 balls per
episode, 600-step cap): the ball moves at constant speed 1.25 cell/step with
unit-cell physics; the paddle is 3 cells wide and moves at most 1 cell per
step toward the commanded target. The full physics — including the
hit-offset bounce rule — is **published in `run.py`** (READONLY). The
research surface is the paddle controller in `policy.py`.

## Task protocol

- Optimize **`bricks_cleared`** (higher is better): mean bricks destroyed per
  episode over 12 seeded episodes (scale [0, 60]).
- The **only** editable file is `policy.py` (contract: `init() -> state`,
  `paddle_target(ball_x, ball_y, ball_vx, ball_vy, paddle_cx, state, t) ->
  desired paddle center x`). Everything else is read-only.
- One coherent change per run (single-variable control).
- `time_budget` is fixed; a contract violation or non-finite outcome scores 0.0.

## Research direction hints (not requirements)

- baseline (lazy tracker, `LEAD_GAIN = 0`, slow `GAIN`): **≈6** — the v3
  ball is faster and the paddle narrower; chasing current x loses almost
  every rally;
- anticipation knobs: `GAIN = 1` alone reaches **≈33** and `LEAD_GAIN ≈ 1`
  **≈21** — paddle speed binds before anticipation does; decoys abound;
- structural rewrite of `policy.py`: wall bounces are cheap to simulate in
  closed form — a landing predictor that reflects off the side walls instead
  of the naive dead-reckoning covers the launch-angle range exactly.
