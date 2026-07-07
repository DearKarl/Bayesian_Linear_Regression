"""Research utilities for Bayesian linear regression experiments."""

from .data import FEATURE_DESCRIPTIONS, TARGET, load_boston_csv, make_feature_target
from .metrics import (
    crps_from_samples,
    interval_metrics,
    interval_score,
    negative_log_predictive_density_mixture,
    regression_metrics,
    rmse,
)
from .models import BayesianLinearRegressionGibbs, PosteriorSummary

__all__ = [
    "BayesianLinearRegressionGibbs",
    "FEATURE_DESCRIPTIONS",
    "PosteriorSummary",
    "TARGET",
    "interval_metrics",
    "interval_score",
    "crps_from_samples",
    "load_boston_csv",
    "make_feature_target",
    "negative_log_predictive_density_mixture",
    "regression_metrics",
    "rmse",
]
