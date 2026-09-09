"""Numerical primitives shared by the Lecture 2 interactive stations."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


class NumericalError(RuntimeError):
    """Raised when a required numerical linear-algebra operation fails."""


def _as_design_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one row")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one column")
    if not np.isfinite(X).all():
        raise ValueError("X must contain finite values")
    return X


def _as_matrix_vector_weights(X, y, w):
    X = _as_design_matrix(X)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.ndim != 1 or w.ndim != 1:
        raise ValueError("y and w must be one-dimensional vectors")
    if X.shape[0] != y.size:
        raise ValueError("X and y must have the same number of rows")
    if X.shape[1] != w.size:
        raise ValueError("X columns must match the number of weights")
    if not (np.isfinite(y).all() and np.isfinite(w).all()):
        raise ValueError("X, y, and w must contain finite values")
    return X, y, w


def _as_matrix_vector(X, y):
    X = _as_design_matrix(X)
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("X and y must have compatible rows")
    if not np.isfinite(y).all():
        raise ValueError("X and y must contain finite values")
    return X, y


def design_matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional vector")
    if not np.isfinite(x).all():
        raise ValueError("x must contain finite values")
    return np.column_stack([np.ones_like(x), x])


def predict(X, w):
    X = _as_design_matrix(X)
    w = np.asarray(w, dtype=float)
    if X.ndim != 2 or w.ndim != 1 or X.shape[1] != w.size:
        raise ValueError("X columns must match the number of weights")
    if not np.isfinite(w).all():
        raise ValueError("X and w must contain finite values")
    return X @ w


def residuals(X, y, w):
    X, y, w = _as_matrix_vector_weights(X, y, w)
    return y - X @ w


def objective(X, y, w, *, kind="mse", delta=1.0):
    r = residuals(X, y, w)
    rho, _ = loss_rho_psi(r, kind=kind, delta=delta)
    return float(np.mean(rho))


def loss_rho_psi(r, *, kind="mse", delta=1.0):
    """Return a loss contribution rho(r) and its residual influence psi(r).

    MAE uses psi(0) = 0, a stated member of its subgradient interval.
    """
    r = np.asarray(r, dtype=float)
    if r.ndim != 1:
        raise ValueError("r must be a one-dimensional vector")
    if not np.isfinite(r).all():
        raise ValueError("r must contain finite values")
    if kind == "mse":
        return 0.5 * r**2, r
    if kind == "mae":
        return np.abs(r), np.sign(r)
    if kind == "huber":
        if (
            isinstance(delta, (bool, np.bool_))
            or not isinstance(delta, (int, float, np.integer, np.floating))
            or not np.isfinite(delta)
            or delta <= 0
        ):
            raise ValueError("delta must be positive and finite")
        magnitude = np.abs(r)
        rho = np.where(
            magnitude <= delta,
            0.5 * r**2,
            delta * (magnitude - 0.5 * delta),
        )
        psi = np.where(magnitude <= delta, r, delta * np.sign(r))
        return rho, psi
    raise ValueError("kind must be one of: mse, mae, huber")


def gradient_mse(X, y, w):
    X, y, w = _as_matrix_vector_weights(X, y, w)
    return X.T @ (X @ w - y) / y.size


def hessian_half_mse(X):
    """Return the constant Hessian of J(w) = ||y - Xw||^2 / (2N)."""
    X = _as_design_matrix(X)
    return X.T @ X / X.shape[0]


def stable_step_limit(X):
    """Return the strict-GD stability boundary 2 / lambda_max for half-MSE."""
    try:
        eigenvalues = np.linalg.eigvalsh(hessian_half_mse(X))
    except np.linalg.LinAlgError as exc:
        raise NumericalError("stable_step_limit failed during eigendecomposition") from exc
    lambda_max = float(eigenvalues[-1])
    return np.inf if lambda_max == 0.0 else 2.0 / lambda_max


def gd_path(X, y, initial_w, *, learning_rate, steps):
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    X, y, w = _as_matrix_vector_weights(X, y, initial_w)
    path = [w.copy()]
    for _ in range(steps):
        w = w - learning_rate * gradient_mse(X, y, w)
        path.append(w.copy())
    return np.asarray(path)


def batch_gradient_mse(X, y, w, *, batch_size, seed):
    X, y, w = _as_matrix_vector_weights(X, y, w)
    _validate_batch_size(batch_size, y.size)
    if batch_size == y.size:
        indices = np.arange(y.size)
    else:
        indices = np.sort(np.random.default_rng(seed).choice(y.size, batch_size, replace=False))
    gradient = gradient_mse(X[indices], y[indices], w)
    return gradient, indices


def enumerate_batch_gradients(X, y, w, *, batch_size):
    """Enumerate every fixed-size mini-batch and its mean half-MSE gradient."""
    X, y, w = _as_matrix_vector_weights(X, y, w)
    _validate_batch_size(batch_size, y.size)
    batches = np.asarray(list(combinations(range(y.size), batch_size)), dtype=int)
    gradients = np.asarray(
        [gradient_mse(X[batch], y[batch], w) for batch in batches], dtype=float
    )
    return gradients, batches


def batch_gradient_summary(X, y, w, *, batch_size):
    """Compare enumerated batch-gradient dispersion with its finite-population law."""
    X, y, w = _as_matrix_vector_weights(X, y, w)
    _validate_batch_size(batch_size, y.size)
    gradients, batches = enumerate_batch_gradients(X, y, w, batch_size=batch_size)
    full_gradient = gradient_mse(X, y, w)
    deviations = gradients - full_gradient
    covariance = deviations.T @ deviations / len(gradients)
    expected_squared_error = float(np.mean(np.sum(deviations**2, axis=1)))

    individual_gradients = np.asarray(
        [gradient_mse(X[[index]], y[[index]], w) for index in range(y.size)]
    )
    individual_deviations = individual_gradients - full_gradient
    if y.size == 1:
        finite_population_covariance = np.zeros((w.size, w.size))
    else:
        finite_population_covariance = (
            (y.size - batch_size) / (batch_size * y.size * (y.size - 1))
        ) * (individual_deviations.T @ individual_deviations)
    finite_population_expected_squared_error = float(
        np.trace(finite_population_covariance)
    )
    return BatchGradientSummary(
        gradients=gradients,
        batches=batches,
        full_gradient=full_gradient,
        mean_gradient=gradients.mean(axis=0),
        covariance=covariance,
        finite_population_covariance=finite_population_covariance,
        expected_squared_error=expected_squared_error,
        finite_population_expected_squared_error=finite_population_expected_squared_error,
    )


def _validate_batch_size(batch_size, sample_count):
    if (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or not 1 <= batch_size <= sample_count
    ):
        raise ValueError("batch_size must be an integer between 1 and N")


def least_squares_summary(X, y):
    X, y = _as_matrix_vector(X, y)
    try:
        weights, _, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise NumericalError("least_squares_summary failed during least-squares solve") from exc
    if singular_values.size == 0:
        condition_number = np.inf
    elif singular_values[-1] == 0:
        condition_number = np.inf
    else:
        condition_number = float(singular_values[0] / singular_values[-1])
    return LeastSquaresSummary(
        weights=weights,
        rank=int(rank),
        singular_values=singular_values,
        condition_number=condition_number,
        rank_deficient=rank < X.shape[1],
    )


def projection_summary(X, y):
    """Summarize the least-squares projection of y onto col(X)."""
    X, y = _as_matrix_vector(X, y)
    summary = least_squares_summary(X, y)
    fitted_values = X @ summary.weights
    return ProjectionSummary(
        weights=summary.weights,
        fitted_values=fitted_values,
        residual=y - fitted_values,
    )


def null_space_family(X, weights):
    """Return a parameter family whose coordinate shifts leave Xw unchanged."""
    X = _as_design_matrix(X)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or X.shape[1] != weights.size:
        raise ValueError("X columns must match the number of weights")
    if not np.isfinite(weights).all():
        raise ValueError("weights must contain finite values")
    try:
        _, singular_values, right_vectors = np.linalg.svd(X, full_matrices=True)
    except np.linalg.LinAlgError as exc:
        raise NumericalError("null_space_family failed during singular-value decomposition") from exc
    tolerance = np.finfo(float).eps * max(X.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    return NullSpaceFamily(base_weights=weights, basis=right_vectors[rank:].T)


@dataclass(frozen=True)
class LeastSquaresSummary:
    weights: np.ndarray
    rank: int
    singular_values: np.ndarray
    condition_number: float
    rank_deficient: bool


@dataclass(frozen=True)
class ProjectionSummary:
    weights: np.ndarray
    fitted_values: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class NullSpaceFamily:
    base_weights: np.ndarray
    basis: np.ndarray

    def weights_for(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 1 or coordinates.size != self.basis.shape[1]:
            raise ValueError("coordinates must match the null-space dimension")
        if not np.isfinite(coordinates).all():
            raise ValueError("coordinates must contain finite values")
        return self.base_weights + self.basis @ coordinates


@dataclass(frozen=True)
class BatchGradientSummary:
    gradients: np.ndarray
    batches: np.ndarray
    full_gradient: np.ndarray
    mean_gradient: np.ndarray
    covariance: np.ndarray
    finite_population_covariance: np.ndarray
    expected_squared_error: float
    finite_population_expected_squared_error: float
