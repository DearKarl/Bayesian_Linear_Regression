"""Lightweight single-chain MCMC diagnostics."""

from __future__ import annotations

import numpy as np


def _as_finite_1d(samples: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(samples, dtype=float).ravel()
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one draw")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def autocorrelation_1d(samples: np.ndarray, lag: int) -> float:
    """Estimate autocorrelation for a one-dimensional chain at a given lag."""

    values = _as_finite_1d(samples, name="samples")
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if lag >= values.size:
        return float("nan")
    if lag == 0:
        return 1.0

    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        return float("nan")

    numerator = float(np.dot(centered[:-lag], centered[lag:]))
    return numerator / denominator


def effective_sample_size_1d(
    samples: np.ndarray,
    max_lag: int | None = None,
) -> float:
    """Approximate effective sample size from positive autocorrelation lags.

    This is a lightweight single-chain approximation for inspection. It is not
    a replacement for multi-chain diagnostics such as R-hat.
    """

    values = _as_finite_1d(samples, name="samples")
    n_draws = values.size
    if n_draws == 1:
        return 1.0

    if max_lag is None:
        max_lag = n_draws - 1
    if max_lag < 1:
        return float(n_draws)
    max_lag = min(int(max_lag), n_draws - 1)

    autocorr_sum = 0.0
    for lag in range(1, max_lag + 1):
        rho = autocorrelation_1d(values, lag)
        if not np.isfinite(rho) or rho <= 0:
            break
        autocorr_sum += rho

    ess = n_draws / (1.0 + 2.0 * autocorr_sum)
    return float(np.clip(ess, 1.0, n_draws))


def summarize_mcmc_samples(
    parameter_names: list[str],
    samples: np.ndarray,
) -> list[dict[str, float | str]]:
    """Summarize posterior draws with lightweight single-chain diagnostics."""

    draws = np.asarray(samples, dtype=float)
    if draws.ndim != 2:
        raise ValueError("samples must be a 2D array")
    if draws.shape[1] != len(parameter_names):
        raise ValueError("parameter_names length must match sample columns")
    if not np.isfinite(draws).all():
        raise ValueError("samples must contain only finite values")

    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(parameter_names):
        values = draws[:, index]
        rows.append(
            {
                "parameter": name,
                "posterior_mean": float(np.mean(values)),
                "posterior_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "ess": effective_sample_size_1d(values),
                "autocorr_lag1": autocorrelation_1d(values, 1),
                "autocorr_lag5": autocorrelation_1d(values, 5),
                "autocorr_lag10": autocorrelation_1d(values, 10),
                "autocorr_lag20": autocorrelation_1d(values, 20),
            }
        )
    return rows
