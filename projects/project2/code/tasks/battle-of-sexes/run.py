#!/usr/bin/env python3
"""battle-of-sexes scoring driver v2 (READONLY contract).

Repeated Battle of the Sexes against an ADAPTIVE published opponent. The
researcher never touches this file — the research surface is the strategy in
``strategy.py``. Pure Python, zero dependencies, sub-second.

Benchmark identity is harness-injected (the epoch-freeze carrier):

- ``RSI_BENCHMARK_SEED``: base seed for the per-episode opponent draws.
  Within one evaluation epoch the harness freezes it, so every run in the
  epoch is comparable; across epochs the seed changes, so cross-epoch scores
  are NOT directly comparable.
- ``RSI_EPOCH``: informational echo (recorded in result.json observations).

Metric: ``mean_payoff`` — the agent's average per-round payoff over
EPISODES episodes of ROUNDS rounds each. Payoff scale is [0, 3].

Non-finite outcomes and contract violations score 0.0 (worst possible).
"""
import json
import os
import random
import sys

import strategy  # the editable strategy

EPISODES = 50
ROUNDS = 25

# Agent payoff (the opponent's payoff is irrelevant to the metric):
#   both O(pera) -> 3, both F(ootball) -> 2, mismatch -> 0.
PAYOFF = {("O", "O"): 3.0, ("F", "F"): 2.0,
          ("O", "F"): 0.0, ("F", "O"): 0.0}

# The opponent, in plain sight (this file is READONLY — reading it IS the
# research skill being measured). v2: the opponent ADAPTS — it tracks the
# agent's recent modal move over a sliding window and drifts toward its own
# preferred equilibrium (Football) when the agent over-commits to Opera. A
# static exploit (always-O) therefore collapses mid-episode; only a strategy
# that tracks the opponent's drifting frequency keeps the coordination
# payoff.
OPP_BASE_OPERA = 0.65     # P(opponent plays O) at neutral history
OPP_ADAPT = 0.28          # drift magnitude per unit of agent over-commitment (v3: faster)
OPP_WINDOW = 4            # sliding window (v3: shorter = twitchier)


def opponent_p_opera(hist_own):
    if len(hist_own) < OPP_WINDOW:
        return OPP_BASE_OPERA
    window = hist_own[-OPP_WINDOW:]
    frac_opera = sum(1 for m in window if m == "O") / OPP_WINDOW
    return max(0.15, min(0.95, OPP_BASE_OPERA - OPP_ADAPT * (frac_opera - 0.5) * 2))


def run_once(seed):
    rng = random.Random(seed)
    total = 0.0
    n = 0
    for ep in range(EPISODES):
        state = strategy.init()
        hist_own, hist_opp = [], []
        ep_rng = random.Random(seed * 1000 + ep)  # frozen opponent stream
        for t in range(ROUNDS):
            opp = "O" if ep_rng.random() < opponent_p_opera(hist_own) else "F"
            own = strategy.move(hist_own, hist_opp, rng, state, t)
            if own not in ("O", "F"):
                return 0.0  # contract violation: worst possible
            total += PAYOFF[(own, opp)]
            n += 1
            hist_own.append(own)
            hist_opp.append(opp)
    return total / n


def _self_check() -> int:
    """Correctness battery for the validate gate: exit 0 = usable strategy."""
    try:
        state = strategy.init()
        rng = random.Random(0)
        hist_own, hist_opp = [], []
        for t in range(8):
            own = strategy.move(hist_own, hist_opp, rng, state, t)
            assert own in ("O", "F"), f"move returned {own!r}"
            hist_own.append(own)
            hist_opp.append("O" if t % 2 else "F")
    except Exception as e:  # noqa: BLE001 — the battery reports, never raises
        print(f"self-check failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    if "--self-check" in sys.argv:
        return _self_check()
    seed = int(os.getenv("RSI_BENCHMARK_SEED", "7"))
    val = run_once(seed)
    if val != val or val in (float("inf"), float("-inf")):
        val = 0.0
    result = {
        "mean_payoff": val,
        "observations": {
            "seed": seed,
            "episodes": EPISODES,
            "rounds": ROUNDS,
            "opp_base_opera": OPP_BASE_OPERA,
            "opp_adapt": OPP_ADAPT,
            "opp_window": OPP_WINDOW,
            "epoch": os.getenv("RSI_EPOCH", ""),
        },
    }
    with open("result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
