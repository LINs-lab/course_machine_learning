---
schema_version: 1
name: prisoners-dilemma
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
# prisoners-dilemma — cooperation timing against a trust-then-grim opponent

Iterated Prisoner's Dilemma (T=5, R=3, P=1, S=0), 20 rounds per episode. The
opponent's full policy is **published in `run.py`** (READONLY): it cooperates
with probability 0.90 until the agent's **first defection**, then drops to
0.10 for the rest of the episode (a noisy grim trigger).

## Task protocol

- Optimize **`mean_payoff`** (higher is better): average per-round payoff over
  60 episodes × 20 rounds (payoff scale [0, 5]).
- The **only** editable file is `strategy.py` (contract: `init() -> state`,
  `move(hist_own, hist_opp, rng, state, t) -> "C"|"D"`). Everything else is
  read-only.
- The opponent code is part of the task statement: reading it and reasoning
  about WHEN defection pays is the intended research skill.
- One coherent change per run (single-variable control).
- `time_budget` is fixed; a contract violation or non-finite outcome scores 0.0.

## Research direction hints (not requirements)

- baseline (always defect, `DEFECT_PERIOD = 1`): **≈1.68** — the trust
  collapses and the episode decays to (D,D);
- scheduling knobs: **≈2.0–2.7** — v3's slower trust rebuild (gain 0.08)
  makes naive 'defect once, exploit' fail; the surface rewards a controlled
  spend/repay schedule, and several knobs are decoys;
- structural rewrite of `strategy.py`: defection has a **time value** — the
  trigger is per-episode and rounds are finite, so there is an optimal
  end-game moment to defect (classic backward induction); harvesting it
  cleanly requires reading the published horizon and trigger, not tuning a
  rate.
