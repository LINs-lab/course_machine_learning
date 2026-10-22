#!/usr/bin/env python3
"""house-price scoring driver (READONLY contract).

Tabular regression on a FROZEN housing dataset (``houses.csv``, shipped in
the bundle): the harness owns the split, the metric, and the budget; the
research surface is the featurizer + regressor in ``model.py``. Pure Python,
zero dependencies, a couple of seconds.

Benchmark identity is harness-injected (the epoch-freeze carrier):

- ``RSI_BENCHMARK_SEED``: seed for the train/test split. Within one
  evaluation epoch the harness freezes it; across epochs the seed changes
  and scores are NOT directly comparable.
- ``RSI_EPOCH``: informational echo (recorded in result.json observations).

Metric: ``rmse`` — root mean squared error of predicted price on the held-out
test rows (30% of the dataset). Lower is better.

Non-finite outcomes, exceptions, and contract violations score 1e12 (worst
possible).
"""
import csv
import json
import math
import os
import random
import sys

import model  # the editable featurizer + regressor

TRAIN_FRAC = 0.7
WORST = 1e12
FEATURES = ["area", "bedrooms", "age", "dist_cbd", "income_zip",
            "garage", "junk1", "junk2", "junk3", "junk4"]


def load_rows():
    with open("houses.csv", newline="") as fh:
        return list(csv.reader(fh))


def run_once(seed):
    rows = load_rows()[1:]  # skip header
    data = [[float(v) for v in r] for r in rows]
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    n_train = int(len(data) * TRAIN_FRAC)
    train = [data[i] for i in idx[:n_train]]
    test = [data[i] for i in idx[n_train:]]

    m = model.Model()
    m.fit([r[:-1] for r in train], [r[-1] for r in train])
    preds = m.predict([r[:-1] for r in test])
    if len(preds) != len(test):
        return WORST
    se = 0.0
    for p, r in zip(preds, test):
        p = float(p)
        if p != p or abs(p) == float("inf"):
            return WORST
        se += (p - r[-1]) ** 2
    return math.sqrt(se / len(test))


def _self_check() -> int:
    """Correctness battery for the validate gate: exit 0 = usable model."""
    try:
        m = model.Model()
        X = [[100.0, 3, 10.0, 5.0, 60.0, 1, 42.0, 99.0, 9.0, 9.0],
             [80.0, 2, 2.0, 12.0, 70.0, 0, 7.0, 81.0, 4.0, 0.0]]
        y = [150000.0, 120000.0]
        m.fit(X, y)
        out = m.predict(X)
        assert len(out) == 2 and all(isinstance(v, float) for v in out)
    except Exception as e:  # noqa: BLE001 — the battery reports, never raises
        print(f"self-check failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    if "--self-check" in sys.argv:
        return _self_check()
    seed = int(os.getenv("RSI_BENCHMARK_SEED", "7"))
    try:
        val = run_once(seed)
    except Exception as e:  # noqa: BLE001 — a crash is the worst outcome
        print(f"model crashed: {type(e).__name__}: {e}", file=sys.stderr)
        val = WORST
    if val != val or val in (float("inf"), float("-inf")):
        val = WORST
    result = {
        "rmse": val,
        "observations": {
            "seed": seed,
            "n_train": int(900 * TRAIN_FRAC),
            "n_test": 900 - int(900 * TRAIN_FRAC),
            "features": FEATURES,
            "epoch": os.getenv("RSI_EPOCH", ""),
        },
    }
    with open("result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
