"""Research utilities for Bayesian linear regression experiments."""

from .data import FEATURE_DESCRIPTIONS, TARGET, load_boston_csv, make_feature_target
from .diagnostics import (
    autocorrelation_1d,
    effective_sample_size_1d,
    summarize_mcmc_samples,
)
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
    "autocorrelation_1d",
    "interval_metrics",
    "interval_score",
    "crps_from_samples",
    "effective_sample_size_1d",
    "load_boston_csv",
    "make_feature_target",
    "negative_log_predictive_density_mixture",
    "regression_metrics",
    "rmse",
    "summarize_mcmc_samples",
]
