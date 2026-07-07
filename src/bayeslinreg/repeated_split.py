"""Utilities for repeated train/test split comparisons."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t

REPEATED_SPLIT_METRICS: tuple[str, ...] = (
    "rmse",
    "mae",
    "r2",
    "coverage_95",
    "mean_interval_width",
    "nlpd",
    "crps",
    "interval_score_95",
)

LOWER_IS_BETTER_METRICS: frozenset[str] = frozenset(
    {
        "rmse",
        "mae",
        "mean_interval_width",
        "nlpd",
        "crps",
        "interval_score_95",
    }
)


def _confidence_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty 1D array")
    if not np.isfinite(values).all():
        raise ValueError("values must contain only finite values")

    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, 0.0, mean, mean

    std = float(np.std(values, ddof=1))
    margin = float(t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
    return mean, std, mean - margin, mean + margin


def _ordered_unique(values: Iterable[str], preferred_order: Sequence[str] | None) -> list[str]:
    seen = list(dict.fromkeys(values))
    if preferred_order is None:
        return seen
    preferred = [value for value in preferred_order if value in seen]
    remainder = [value for value in seen if value not in preferred]
    return preferred + remainder


def summarize_repeated_splits(
    raw_results: pd.DataFrame,
    *,
    metrics: Sequence[str] = REPEATED_SPLIT_METRICS,
    model_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize repeated-split metrics by model."""

    required = {"model", "split", *metrics}
    missing = required.difference(raw_results.columns)
    if missing:
        raise ValueError(f"raw_results is missing columns: {sorted(missing)}")

    rows = []
    models = _ordered_unique(raw_results["model"].astype(str), model_order)
    for model in models:
        group = raw_results[raw_results["model"] == model]
        for metric in metrics:
            mean, std, lower, upper = _confidence_interval(group[metric].to_numpy())
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_repeats": int(group["split"].nunique()),
                    "mean": mean,
                    "std": std,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def pairwise_against_reference(
    raw_results: pd.DataFrame,
    *,
    reference_model: str = "Bayesian Gibbs",
    metrics: Sequence[str] = REPEATED_SPLIT_METRICS,
    lower_is_better_metrics: set[str] | frozenset[str] = LOWER_IS_BETTER_METRICS,
    model_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute paired baseline-minus-reference differences across splits."""

    required = {"model", "split", *metrics}
    missing = required.difference(raw_results.columns)
    if missing:
        raise ValueError(f"raw_results is missing columns: {sorted(missing)}")
    if reference_model not in set(raw_results["model"]):
        raise ValueError(f"reference_model {reference_model!r} is not present")

    reference = raw_results[raw_results["model"] == reference_model].set_index("split")
    models = [
        model
        for model in _ordered_unique(raw_results["model"].astype(str), model_order)
        if model != reference_model
    ]

    rows = []
    for model in models:
        baseline = raw_results[raw_results["model"] == model].set_index("split")
        common_splits = baseline.index.intersection(reference.index)
        if len(common_splits) == 0:
            raise ValueError(f"model {model!r} has no splits in common with {reference_model!r}")

        baseline = baseline.loc[common_splits].sort_index()
        reference_aligned = reference.loc[common_splits].sort_index()
        for metric in metrics:
            differences = baseline[metric].to_numpy() - reference_aligned[metric].to_numpy()
            mean, std, lower, upper = _confidence_interval(differences)
            ties = np.isclose(differences, 0.0)
            if metric in lower_is_better_metrics:
                wins = differences > 0.0
            else:
                wins = differences < 0.0
            win_rate = float((np.sum(wins) + 0.5 * np.sum(ties)) / len(differences))
            rows.append(
                {
                    "reference_model": reference_model,
                    "comparison_model": model,
                    "metric": metric,
                    "n_repeats": int(len(common_splits)),
                    "mean_difference": mean,
                    "std_difference": std,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "win_rate_for_gibbs": win_rate,
                }
            )

    return pd.DataFrame(rows)
