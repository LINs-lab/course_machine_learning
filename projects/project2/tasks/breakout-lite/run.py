#!/usr/bin/env python3
"""breakout-lite scoring driver (READONLY contract).

A compact Breakout: a 12x4 brick field, a 3-cell paddle, 3 balls per
episode. The physics, episode protocol, and scoring are fixed here; the
research surface is the paddle controller in ``policy.py``. Pure Python,
sub-second.

Benchmark identity is harness-injected (the epoch-freeze carrier):

- ``RSI_BENCHMARK_SEED``: base seed for the per-episode launch angles.
  Within one evaluation epoch the harness freezes it; across epochs the seed
  changes and scores are NOT directly comparable.
- ``RSI_EPOCH``: informational echo (recorded in result.json observations).

Metric: ``bricks_cleared`` — mean number of bricks destroyed per episode
(out of 48 = 12 columns x 4 rows), averaged over EPISODES episodes.

Non-finite outcomes and contract violations score 0.0 (worst possible).
"""
import json
import math
import os
import random
import sys

import policy  # the editable paddle controller

W = 12.0              # field width (x in [0, W])
BRICK_ROWS = 5
BRICK_Y0, BRICK_Y1 = 6.0, 11.0
PADDLE_Y = 1.0
PADDLE_HALF = 0.95    # v3: narrower
BALLS = 2
EPISODES = 12
MAX_STEPS = 600
SPEED = 1.35


class Sim:
    """Fixed breakout physics (published, READONLY)."""

    def __init__(self, rng):
        self.rng = rng
        self.bricks = [[True] * W_int() for _ in range(BRICK_ROWS)]
        self.cx = W / 2.0
        self.cleared = 0
        self.launch()

    def launch(self):
        ang = math.radians(self.rng.uniform(30.0, 150.0))
        self.bx, self.by = W / 2.0, 2.0
        self.vx = SPEED * math.cos(ang)
        self.vy = SPEED * math.sin(ang)  # upward launch

    def step(self, target_x):
        # paddle: move at most 1 cell/step toward the commanded target
        d = max(-1.0, min(1.0, target_x - self.cx))
        self.cx = max(PADDLE_HALF, min(W - PADDLE_HALF, self.cx + d))
        lost = False
        for _ in range(2):  # half-steps to avoid tunneling
            self.bx += self.vx / 2.0
            self.by += self.vy / 2.0
            if self.bx < 0.0 or self.bx > W:
                self.vx = -self.vx
                self.bx = max(0.0, min(W, self.bx))
            row = int(self.by - BRICK_Y0)
            col = int(self.bx)
            if 0 <= row < BRICK_ROWS and 0 <= col < W_int() and self.bricks[row][col]:
                self.bricks[row][col] = False
                self.cleared += 1
                self.vy = -self.vy
            if self.by >= BRICK_Y1:
                self.by = BRICK_Y1 - 1e-6
                self.vy = -abs(self.vy)
            if self.by <= PADDLE_Y and self.vy < 0.0:
                if abs(self.bx - self.cx) <= PADDLE_HALF + 0.2:
                    self.vy = abs(self.vy)
                    self.vx = self.vx * 0.5 + 0.35 * (self.bx - self.cx)
                    n = math.hypot(self.vx, self.vy)
                    self.vx *= SPEED / n
                    self.vy *= SPEED / n
                elif self.by < 0.0:
                    lost = True
                    break
        return lost


def W_int():
    return int(W)


def run_once(seed):
    total = 0
    for ep in range(EPISODES):
        rng = random.Random(seed * 1000 + ep)
        sim = Sim(rng)
        state = policy.init()
        balls = BALLS
        for t in range(MAX_STEPS):
            target = policy.paddle_target(
                sim.bx, sim.by, sim.vx, sim.vy, sim.cx, state, t)
            if not isinstance(target, (int, float)) or target != target \
                    or abs(target) == float("inf"):
                return 0.0  # contract violation: worst possible
            if sim.step(float(target)):
                balls -= 1
                if balls <= 0:
                    break
                sim.launch()
        total += sim.cleared
    return total / EPISODES


def _self_check() -> int:
    """Correctness battery for the validate gate: exit 0 = usable policy."""
    try:
        state = policy.init()
        for t in range(3):
            x = policy.paddle_target(6.0, 5.0, 0.3, -0.9, 6.0, state, t)
            assert isinstance(x, (int, float)) and x == x and abs(x) != float("inf")
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
        "bricks_cleared": val,
        "observations": {
            "seed": seed,
            "episodes": EPISODES,
            "bricks_total": BRICK_ROWS * W_int(),
            "balls": BALLS,
            "max_steps": MAX_STEPS,
            "epoch": os.getenv("RSI_EPOCH", ""),
        },
    }
    with open("result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
