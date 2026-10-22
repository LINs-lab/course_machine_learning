"""Iris classifier — the ONLY file the researcher (or the RSI system) edits.

Baseline: a softmax-regression linear classifier trained by gradient descent
on raw features. On the OOD split (monotone per-feature warp) this baseline
degrades badly — the headroom is in the preprocessing knobs and beyond:

- POWER : feature exponent applied before fitting/predicting,
  x[f] <- x[f] ** POWER. The OOD warp composes with this: matching the warp
  (partially) neutralizes the shift.
- CLIP  : z-score winsorization bound applied AFTER train-statistic
  standardization. The warp inflates OOD features far outside the train
  range; clipping them back is the classic robust-statistics response.
- REG   : L2 regularization strength.
- LR    : gradient-descent learning rate.
- STEPS : training iterations.

Contract (keep it):
- class Model with fit(X: list[list[float]], y: list[int]) and
  predict(X: list[list[float]]) -> list[int]; standard library only
  (``math`` is imported; no third-party packages — sklearn/numpy would
  trivialize the research problem).
- The model never sees the warp parameters — robustness must come from the
  model's own structure, not from peeking at the evaluation.
"""
import math

POWER = 1.0
CLIP = 10.0
REG = 1e-3
LR = 0.5
STEPS = 200


class Model:
    def _prep(self, row):
        # power transform, affine standardization by TRAIN statistics, then
        # winsorize at CLIP sigma (the warp inflates OOD features far outside
        # the train range — affine stats alone cannot undo that)
        out = []
        for f in range(len(row)):
            z = (row[f] ** POWER - self._mean[f]) / self._std[f]
            out.append(max(-CLIP, min(CLIP, z)))
        return out

    def fit(self, X, y):
        d = len(X[0])
        powered = [[v ** POWER for v in r] for r in X]
        self._mean = [sum(r[f] for r in powered) / len(powered)
                      for f in range(d)]
        self._std = [math.sqrt(sum((r[f] - self._mean[f]) ** 2
                                   for r in powered) / len(powered))
                     or 1.0 for f in range(d)]
        X = [self._prep(r) for r in X]
        d = len(X[0])
        self.W = [[0.0] * d for _ in range(3)]
        self.b = [0.0] * 3
        n = len(X)
        for _ in range(int(STEPS)):
            grads = [[0.0] * d for _ in range(3)]
            gbias = [0.0] * 3
            for row, label in zip(X, y):
                scores = [sum(w * v for w, v in zip(self.W[c], row))
                          + self.b[c] for c in range(3)]
                mx = max(scores)
                exps = [math.exp(s - mx) for s in scores]
                z = sum(exps)
                for c in range(3):
                    p = exps[c] / z
                    g = p - (1.0 if c == label else 0.0)
                    gbias[c] += g
                    for j in range(d):
                        grads[c][j] += g * row[j]
            for c in range(3):
                for j in range(d):
                    self.W[c][j] -= LR * (grads[c][j] / n
                                          + REG * self.W[c][j])
                self.b[c] -= LR * (gbias[c] / n)

    def predict(self, X):
        preds = []
        for row in X:
            row = self._prep(row)
            scores = [sum(w * v for w, v in zip(self.W[c], row)) + self.b[c]
                      for c in range(3)]
            preds.append(max(range(3), key=lambda c: scores[c]))
        return preds
