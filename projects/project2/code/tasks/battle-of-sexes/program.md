---
schema_version: 1
name: battle-of-sexes
entrypoint: ["python3", "run.py"]
time_budget: 60
metric:
  name: mean_payoff
  direction: max
  result_file: result.json
  result_key: mean_payoff
editable: ["strategy.py"]
readonly: ["run.py"]
validate: ["python3", "run.py", "--self-check"]
---
# battle-of-sexes v2 — coordination against an ADAPTIVE opponent

Repeated Battle of the Sexes: both players want to coordinate, but the agent
prefers Opera (payoff 3) and tolerates Football (payoff 2); mismatch scores
0. The opponent's policy is **published in `run.py`** (READONLY) — and it
adapts: it tracks the agent's recent modal move over a sliding window and
drifts away when the agent over-commits, so a static exploit (always-O)
collapses mid-episode.

## Task protocol

- Optimize **`mean_payoff`** (higher is better): the agent's average per-round
  payoff over 50 episodes × 25 rounds against seeded opponent draws
  (payoff scale [0, 3]).
- The **only** editable file is `strategy.py` (contract: `init() -> state`,
  `move(hist_own, hist_opp, rng, state, t) -> "O"|"F"`). Everything else is
  read-only.
- The opponent code is part of the task statement: reading it and reasoning
  about its adaptation is the intended research skill.
- One coherent change per run (single-variable control).
- `time_budget` is fixed; a contract violation or non-finite outcome scores 0.0.

## Research direction hints (not requirements)

- baseline (coin flip): **≈1.34** — half the rounds mismatch;
- single-knob exploits: **≈1.42** — playing the opponent's modal action works
  only briefly: v3's faster adaptation (window 4, drift 0.28) collapses it
  quickly; `TRACK_W` and `EXPLOIT` **interact**, and several knobs are
  decoys that must be rejected by evidence;
- structural rewrite of `strategy.py`: the opponent's equilibrium drift rate
  is computable from the published rule — a tracker that holds the opponent
  near its neutral point (≈0.55–0.6 Opera) collects **≈1.7**, the flat
  portion of the adaptation surface.
