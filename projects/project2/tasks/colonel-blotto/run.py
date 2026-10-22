#!/usr/bin/env python3
"""colonel-blotto scoring driver v3 (READONLY contract).

Colonel Blotto with VALUE-WEIGHTED battlefields and a COUNTERING opponent:
after a warmup, the opponent reallocates part of its budget onto the fields
the agent contested most over recent games — a static concentration gets
punished. The researcher never touches this file; the research surface is the
allocation strategy in ``strategy.py``.

Benchmark identity is harness-injected (the epoch-freeze carrier):

- ``RSI_BENCHMARK_SEED``: base seed for the per-game opponent allocations.
  Within one evaluation epoch the harness freezes it; across epochs the seed
  changes and scores are NOT directly comparable.
- ``RSI_EPOCH``: informational echo (recorded in result.json observations).

Metric: ``value_rate`` — fraction of the TOTAL battlefield value won,
averaged over GAMES seeded games (a tie splits the field's value).

Non-finite outcomes and contract violations score 0.0 (worst possible).
"""
import json
import os
import random
import sys

import strategy  # the editable allocation strategy

N_FIELDS = 12
FIELD_VALUES = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5]   # published, unequal
TOTAL_VALUE = sum(FIELD_VALUES)
BUDGET = 120
GAMES = 300
WARMUP = 40          # games before the opponent starts countering
COUNTER_SHARE = 0.18 # fraction of budget the opponent reallocates onto the
                     # agent's top-2 recently-contested fields
COUNTER_WINDOW = 8   # recent games tracked


def opponent_allocation(rng, agent_history):
    """Published opponent. Warmup: mid-spread random weights. After WARMUP
    games: random weights PLUS COUNTER_SHARE of the budget concentrated on
    the agent's two most-contested fields over the last COUNTER_WINDOW games.
    """
    weights = [rng.uniform(0.5, 1.5) for _ in range(N_FIELDS)]
    scale = BUDGET / sum(weights)
    raw = [w * scale for w in weights]
    if len(agent_history) >= WARMUP:
        recent = agent_history[-COUNTER_WINDOW:]
        contested = [0.0] * N_FIELDS
        for alloc in recent:
            for i, v in enumerate(alloc):
                contested[i] += v * FIELD_VALUES[i]  # value-weighted contest
        top2 = sorted(range(N_FIELDS), key=lambda i: contested[i],
                      reverse=True)[:2]
        reserve = BUDGET * COUNTER_SHARE
        for i in range(N_FIELDS):
            raw[i] *= (1.0 - COUNTER_SHARE)
        for i in top2:
            raw[i] += reserve / 2.0
    alloc = [int(x) for x in raw]
    order = sorted(range(N_FIELDS), key=lambda i: raw[i] - alloc[i], reverse=True)
    for i in range(BUDGET - sum(alloc)):
        alloc[order[i % N_FIELDS]] += 1
    return alloc


def run_once(seed):
    rng = random.Random(seed)
    total = 0.0
    history = []
    for g in range(GAMES):
        alloc = strategy.allocate(rng, g)
        if (not isinstance(alloc, (list, tuple)) or len(alloc) != N_FIELDS
                or not all(isinstance(v, (int, float)) for v in alloc)
                or any(v < 0 for v in alloc) or sum(alloc) > BUDGET + 1e-9):
            return 0.0  # contract violation: worst possible
        opp = opponent_allocation(random.Random(seed * 1000 + g), history)
        for a, b, v in zip(alloc, opp, FIELD_VALUES):
            if a > b:
                total += v
            elif a < b:
                total += 0.0
            else:
                total += v / 2
        history.append(alloc)
    return total / (GAMES * TOTAL_VALUE)


def _self_check() -> int:
    """Correctness battery for the validate gate: exit 0 = usable strategy."""
    try:
        alloc = strategy.allocate(random.Random(0), 0)
        assert isinstance(alloc, (list, tuple)) and len(alloc) == N_FIELDS
        assert all(isinstance(v, (int, float)) and v >= 0 for v in alloc)
        assert sum(alloc) <= BUDGET + 1e-9
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
        "value_rate": val,
        "observations": {
            "seed": seed,
            "n_fields": N_FIELDS,
            "total_value": TOTAL_VALUE,
            "budget": BUDGET,
            "games": GAMES,
            "warmup": WARMUP,
            "counter_share": COUNTER_SHARE,
            "epoch": os.getenv("RSI_EPOCH", ""),
        },
    }
    with open("result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
