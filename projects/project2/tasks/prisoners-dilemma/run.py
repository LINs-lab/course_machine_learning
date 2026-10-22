#!/usr/bin/env python3
"""prisoners-dilemma scoring driver v2 (READONLY contract).

Iterated Prisoner's Dilemma against an opponent with a REPAIRABLE trust
level: defection halves its trust, cooperation slowly rebuilds it. The
researcher never touches this file — the research surface is the strategy in
``strategy.py``. Pure Python, zero dependencies, sub-second.

Benchmark identity is harness-injected (the epoch-freeze carrier):

- ``RSI_BENCHMARK_SEED``: base seed for the per-episode opponent noise.
  Within one evaluation epoch the harness freezes it; across epochs the seed
  changes and scores are NOT directly comparable.
- ``RSI_EPOCH``: informational echo (recorded in result.json observations).

Metric: ``mean_payoff`` — the agent's average per-round payoff over
EPISODES episodes of ROUNDS rounds. Payoff scale is [0, 5] (standard PD:
T=5, R=3, P=1, S=0).

Non-finite outcomes and contract violations score 0.0 (worst possible).
"""
import json
import os
import random
import sys

import strategy  # the editable strategy

EPISODES = 50
ROUNDS = 24

# Standard Prisoner's Dilemma payoffs for the agent row (C/D = agent move):
#   (C, C) -> 3   (C, D) -> 0   (D, C) -> 5   (D, D) -> 1
PAYOFF = {("C", "C"): 3.0, ("C", "D"): 0.0, ("D", "C"): 5.0, ("D", "D"): 1.0}

# The opponent, in plain sight (READONLY — reading it IS the skill). v2: its
# trust is a state, not a switch — a defection halves it, each cooperation
# rebuilds it a little. Exploiting trust without repaying collapses the
# episode; cooperating forever leaves the temptation payoff unharvested.
OPP_TRUST_LOSS = 0.65    # trust multiplier after observing a defection (v3: harsher)
OPP_TRUST_GAIN = 0.08   # trust increment per observed cooperation (v3: slower rebuild)
OPP_P0 = 0.10            # cooperation probability at trust 0
OPP_P1 = 0.80            # trust-scaled cooperation probability range


def opponent_move(rng, trust):
    p = OPP_P0 + OPP_P1 * trust
    return "C" if rng.random() < p else "D"


def run_once(seed):
    rng = random.Random(seed)
    total = 0.0
    n = 0
    for ep in range(EPISODES):
        state = strategy.init()
        hist_own, hist_opp = [], []
        trust = 1.0
        ep_rng = random.Random(seed * 1000 + ep)  # frozen opponent stream
        for t in range(ROUNDS):
            opp = opponent_move(ep_rng, trust)
            own = strategy.move(hist_own, hist_opp, rng, state, t)
            if own not in ("C", "D"):
                return 0.0  # contract violation: worst possible
            if own == "D":
                trust *= 1.0 - OPP_TRUST_LOSS
            else:
                trust = min(1.0, trust + OPP_TRUST_GAIN)
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
            assert own in ("C", "D"), f"move returned {own!r}"
            hist_own.append(own)
            hist_opp.append("C" if t % 2 else "D")
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
            "trust_loss": OPP_TRUST_LOSS,
            "trust_gain": OPP_TRUST_GAIN,
            "epoch": os.getenv("RSI_EPOCH", ""),
        },
    }
    with open("result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
