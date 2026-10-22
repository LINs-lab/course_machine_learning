#!/usr/bin/env python3
"""iris-ood scoring driver (READONLY contract).

Task: train a classifier on the in-distribution iris split; the gate metric
is accuracy on an OUT-OF-DISTRIBUTION (OOD) split whose features have been
warped by a seeded MONOTONE per-feature transformation (power + scale).
Plain iid accuracy on iris is saturated — this task has headroom only for
models that are robust to covariate shift.

Benchmark identity is harness-injected:
- ``RSI_BENCHMARK_SEED``  seeds the split AND the warp parameters (per-feature
  scale factors). Frozen within an evaluation epoch; changes across epochs.
- ``RSI_EPOCH``          informational echo.

The warp family (not its parameters) is part of the published protocol:
x_ood[f] = b_f * x[f] ** W, with W fixed and b_f ~ seeded in [0.5, 2.0].
The model never sees b_f; a robust model must neutralize monotone per-feature
warp (e.g. via its own feature preprocessing) rather than memorize scales.

Zero dependencies (pure python). Writes result.json:
    {"ood_accuracy": ..., "id_accuracy": ..., "observations": {...}}
"""
import csv
import json
import math
import os
import random
import sys

import model  # the editable classifier

WARP_EXPONENT = 2.0          # published: the warp's power
N_TRAIN_PER_CLASS = 30       # 90 train / 15 id-val folded into train report / 45 ood
N_ID_PER_CLASS = 5
PENALTY = 0.0                # non-finite / crash => worst possible accuracy


def load_iris():
    X, y = [], []
    with open(os.path.join(os.path.dirname(__file__), "iris.csv")) as fh:
        for row in csv.DictReader(fh):
            X.append([float(row[k]) for k in
                      ("sepal_length", "sepal_width", "petal_length",
                       "petal_width")])
            y.append(int(row["class"]))
    return X, y


def split_and_warp(seed):
    """Seeded stratified split; OOD features warped by the shift family."""
    X, y = load_iris()
    rng = random.Random(seed)
    scales = [rng.uniform(0.5, 2.0) for _ in range(4)]  # b_f, unknown to model
    train_X, train_y, id_X, id_y, ood_X, ood_y = [], [], [], [], [], []
    for c in (0, 1, 2):
        idx = [i for i, label in enumerate(y) if label == c]
        rng.shuffle(idx)
        for i in idx[:N_TRAIN_PER_CLASS]:
            train_X.append(X[i]); train_y.append(c)
        for i in idx[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_ID_PER_CLASS]:
            id_X.append(X[i]); id_y.append(c)
        for i in idx[N_TRAIN_PER_CLASS + N_ID_PER_CLASS:]:
            ood_X.append([scales[f] * X[i][f] ** WARP_EXPONENT
                          for f in range(4)])
            ood_y.append(c)
    return train_X, train_y, id_X, id_y, ood_X, ood_y


def accuracy(preds, gold):
    if len(gold) == 0:
        return 0.0
    return sum(1 for p, g in zip(preds, gold) if p == g) / len(gold)


def _self_check() -> int:
    try:
        m = model.Model()
        m.fit([[0.0, 0.0, 0.0, 0.0], [9.9, 4.4, 6.9, 2.5], [5.0, 2.0, 3.5, 1.0]],
              [0, 2, 1])
        out = m.predict([[9.9, 4.4, 6.9, 2.5], [0.0, 0.0, 0.0, 0.0]])
        assert len(out) == 2 and all(isinstance(v, int) and 0 <= v <= 2
                                     for v in out)
    except Exception as e:  # noqa: BLE001 — the battery reports, never raises
        print(f"self-check failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    if "--self-check" in sys.argv:
        return _self_check()
    seed = int(os.getenv("RSI_BENCHMARK_SEED", "7"))
    train_X, train_y, id_X, id_y, ood_X, ood_y = split_and_warp(seed)
    m = model.Model()
    try:
        m.fit(train_X, train_y)
        ood_acc = accuracy(m.predict(ood_X), ood_y)
        id_acc = accuracy(m.predict(id_X), id_y)
        if not (math.isfinite(ood_acc) and math.isfinite(id_acc)):
            raise ValueError("non-finite accuracy")
    except Exception as e:  # noqa: BLE001 — a crash scores PENALTY, never NaN
        print(f"model failed: {type(e).__name__}: {e}", file=sys.stderr)
        ood_acc, id_acc = PENALTY, PENALTY
    result = {
        "ood_accuracy": ood_acc,
        "id_accuracy": id_acc,
        "observations": {
            "seed": seed,
            "warp_exponent": WARP_EXPONENT,
            "n_train": len(train_X), "n_id": len(id_X), "n_ood": len(ood_X),
            "epoch": os.getenv("RSI_EPOCH", ""),
            "ood_minus_id": round(ood_acc - id_acc, 4),
        },
    }
    with open("result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
