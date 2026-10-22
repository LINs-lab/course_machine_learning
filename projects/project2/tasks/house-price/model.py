"""Featurizer + regressor — the ONLY file the researcher (or the RSI
system) edits.

Baseline: plain ridge regression on ALL nine raw features (degree-1, no
transform) — including the three junk columns (pure noise, a collinear
duplicate of area, and a heavy-tailed outlier feature) that the dataset ships
as traps. The published pain points: a heavy-tailed ``area``, a piecewise
``dist_cbd`` response (urban cliff at 8 km), a saturating ``age`` effect, and
the junk columns. The knobs are the single-variable control surface; a
structural rewrite may replace any of this file freely as long as the Model
contract holds.
"""
import math

WINSOR_Q = 0.0     # two-sided area quantile cap (0.0 = off, e.g. 0.02)
LOG_AREA = 0.0     # 1 = log-transform area (0/1)
POLY_DEGREE = 1.0  # polynomial degree for dist_cbd and age (1..3)
ALPHA_L2 = 1.0     # ridge penalty
STANDARDIZE = 0.0  # 1 = z-score features before the solve (0/1)
DROP_JUNK = 0.0    # 1 = drop the three junk columns (0/1)
ROBUST_HUBER = 0.0  # decoy: not wired into the loss
LOG_INCOME = 0.0    # decoy: not wired into the featurizer



class Model:
    def fit(self, X, y):
        areas = sorted(r[0] for r in X)
        n = len(areas)
        q = WINSOR_Q
        self.a_lo = areas[max(0, int(n * q))] if q > 0 else -math.inf
        self.a_hi = areas[min(n - 1, int(n * (1.0 - q)))] if q > 0 else math.inf

        F = [self._featurize(r) for r in X]
        k = len(F[0])
        if STANDARDIZE >= 0.5:
            self.mu = [sum(f[j] for f in F) / len(F) for j in range(k)]
            self.sd = [(sum((f[j] - self.mu[j]) ** 2 for f in F) / len(F))
                       ** 0.5 or 1.0 for j in range(k)]
        else:
            self.mu = [0.0] * k
            self.sd = [1.0] * k
        Z = [[(f[j] - self.mu[j]) / self.sd[j] for j in range(k)] for f in F]

        # ridge via normal equations (with intercept; intercept unpenalized)
        A = [[0.0] * (k + 1) for _ in range(k + 1)]
        b = [0.0] * (k + 1)
        for z, t in zip(Z, y):
            row = z + [1.0]
            for i in range(k + 1):
                for j in range(i, k + 1):
                    A[i][j] += row[i] * row[j]
                    if j != i:
                        A[j][i] = A[i][j]
                b[i] += row[i] * t
        for i in range(k):
            A[i][i] += ALPHA_L2
        self.w = self._solve(A, b)

    def predict(self, X):
        out = []
        for r in X:
            f = self._featurize(r)
            z = [(f[j] - self.mu[j]) / self.sd[j] for j in range(len(f))]
            out.append(sum(wi * zi for wi, zi in zip(self.w, z)) + self.w[-1])
        return out

    # -- internals ------------------------------------------------------
    def _featurize(self, row):
        area, bedrooms, age, dist, income, garage = row[:6]
        junk = row[6:10] if DROP_JUNK < 0.5 else []
        a = min(max(area, self.a_lo), self.a_hi)
        if LOG_AREA >= 0.5:
            a = math.log(max(a, 1.0))
        deg = max(1, min(3, int(POLY_DEGREE)))
        feats = [a, float(bedrooms)]
        for power in range(1, deg + 1):
            feats.append(dist ** power)
            feats.append(age ** power)
        feats.extend([income, float(garage)])
        feats.extend(float(v) for v in junk)
        return feats

    @staticmethod
    def _solve(A, b):
        """Gaussian elimination with partial pivoting."""
        n = len(b)
        M = [A[i][:] + [b[i]] for i in range(n)]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(M[r][col]))
            if abs(M[piv][col]) < 1e-12:
                continue
            M[col], M[piv] = M[piv], M[col]
            inv = 1.0 / M[col][col]
            for r in range(col + 1, n):
                f = M[r][col] * inv
                if f:
                    for c in range(col, n + 1):
                        M[r][c] -= f * M[col][c]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
            x[i] = s / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0
        return x
