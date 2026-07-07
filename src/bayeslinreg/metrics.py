"""Evaluation helpers for point and probabilistic regression outputs."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _as_1d_float(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - max_values)
    return np.squeeze(max_values, axis=axis) + np.log(np.sum(shifted, axis=axis))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage_95": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(upper - lower)),
    }


def normal_prediction_interval(
    mean: np.ndarray,
    std: np.ndarray,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Central prediction interval for independent normal predictive laws."""

    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")

    mean = _as_1d_float(mean, name="mean")
    std = _as_1d_float(std, name="std")
    if mean.shape != std.shape:
        raise ValueError("mean and std must have the same shape")
    if np.any(std <= 0):
        raise ValueError("std must be positive")

    z_value = ndtri(0.5 + level / 2.0)
    return mean - z_value * std, mean + z_value * std


def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Mean central prediction interval score.

    Lower values are better. The score rewards narrow intervals but applies a
    penalty when observations fall outside the interval.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    y_true = _as_1d_float(y_true, name="y_true")
    lower = _as_1d_float(lower, name="lower")
    upper = _as_1d_float(upper, name="upper")
    if not (y_true.shape == lower.shape == upper.shape):
        raise ValueError("y_true, lower, and upper must have the same shape")
    if np.any(lower > upper):
        raise ValueError("lower must be less than or equal to upper")

    below = y_true < lower
    above = y_true > upper
    score = (upper - lower).copy()
    score[below] += (2.0 / alpha) * (lower[below] - y_true[below])
    score[above] += (2.0 / alpha) * (y_true[above] - upper[above])
    return float(np.mean(score))


def crps_from_samples(y_true: np.ndarray, predictive_samples: np.ndarray) -> float:
    """Sample-based continuous ranked probability score.

    ``predictive_samples`` is expected to have shape
    ``(n_posterior_samples, n_observations)``. Lower values are better.
    """

    y_true = _as_1d_float(y_true, name="y_true")
    samples = np.asarray(predictive_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("predictive_samples must be a 2D array")
    if samples.shape[1] != y_true.shape[0]:
        raise ValueError("predictive_samples columns must match y_true length")
    if not np.isfinite(samples).all():
        raise ValueError("predictive_samples must contain only finite values")

    first_term = np.mean(np.abs(samples - y_true[None, :]), axis=0)
    sorted_samples = np.sort(samples, axis=0)
    n_samples = sorted_samples.shape[0]
    weights = 2 * np.arange(1, n_samples + 1) - n_samples - 1
    mean_pairwise_abs = (
        2.0
        * np.sum(weights[:, None] * sorted_samples, axis=0)
        / (n_samples * n_samples)
    )
    crps = first_term - 0.5 * mean_pairwise_abs
    return float(np.mean(crps))


def negative_log_predictive_density_normal(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    """Mean negative log predictive density for normal predictions."""

    y_true = _as_1d_float(y_true, name="y_true")
    mean = _as_1d_float(mean, name="mean")
    std = _as_1d_float(std, name="std")
    if not (y_true.shape == mean.shape == std.shape):
        raise ValueError("y_true, mean, and std must have the same shape")
    if np.any(std <= 0):
        raise ValueError("std must be positive")

    z = (y_true - mean) / std
    log_density = -0.5 * np.log(2.0 * np.pi) - np.log(std) - 0.5 * z**2
    return float(-np.mean(log_density))


def crps_normal(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    """Closed-form mean CRPS for normal predictive distributions."""

    y_true = _as_1d_float(y_true, name="y_true")
    mean = _as_1d_float(mean, name="mean")
    std = _as_1d_float(std, name="std")
    if not (y_true.shape == mean.shape == std.shape):
        raise ValueError("y_true, mean, and std must have the same shape")
    if np.any(std <= 0):
        raise ValueError("std must be positive")

    z = (y_true - mean) / std
    normal_pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    normal_cdf = ndtr(z)
    score = std * (
        z * (2.0 * normal_cdf - 1.0)
        + 2.0 * normal_pdf
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(score))


def normal_predictive_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    level: float = 0.95,
) -> dict[str, float]:
    """Coverage, sharpness, and proper scores for normal predictions."""

    if not np.isclose(level, 0.95):
        raise ValueError("normal_predictive_metrics currently reports 95% metrics only")

    lower, upper = normal_prediction_interval(mean, std, level=level)
    return {
        **interval_metrics(y_true, lower, upper),
        "nlpd": negative_log_predictive_density_normal(y_true, mean, std),
        "crps": crps_normal(y_true, mean, std),
        "interval_score_95": interval_score(
            y_true,
            lower,
            upper,
            alpha=1.0 - level,
        ),
    }


def negative_log_predictive_density_mixture(
    y_true: np.ndarray,
    location_samples: np.ndarray,
    sigma2_samples: np.ndarray,
) -> float:
    """Negative log predictive density for a normal posterior mixture.

    The predictive density for each observation is approximated as an equally
    weighted mixture of normal densities from posterior draws.
    """

    y_true = _as_1d_float(y_true, name="y_true")
    locations = np.asarray(location_samples, dtype=float)
    sigma2 = _as_1d_float(sigma2_samples, name="sigma2_samples")

    if locations.ndim != 2:
        raise ValueError("location_samples must be a 2D array")
    if locations.shape[1] != y_true.shape[0]:
        raise ValueError("location_samples columns must match y_true length")
    if locations.shape[0] != sigma2.shape[0]:
        raise ValueError("location_samples rows must match sigma2_samples length")
    if not np.isfinite(locations).all():
        raise ValueError("location_samples must contain only finite values")
    if np.any(sigma2 <= 0):
        raise ValueError("sigma2_samples must be positive")

    residual2 = (y_true[None, :] - locations) ** 2
    log_density = -0.5 * (
        np.log(2.0 * np.pi * sigma2[:, None]) + residual2 / sigma2[:, None]
    )
    log_predictive_density = _logsumexp(log_density, axis=0) - np.log(sigma2.shape[0])
    return float(-np.mean(log_predictive_density))
