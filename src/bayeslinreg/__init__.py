"""Research utilities for Bayesian linear regression experiments."""

from .data import FEATURE_DESCRIPTIONS, TARGET, load_boston_csv, make_feature_target
from .metrics import interval_metrics, regression_metrics, rmse
from .models import BayesianLinearRegressionGibbs, PosteriorSummary

__all__ = [
    "BayesianLinearRegressionGibbs",
    "FEATURE_DESCRIPTIONS",
    "PosteriorSummary",
    "TARGET",
    "interval_metrics",
    "load_boston_csv",
    "make_feature_target",
    "regression_metrics",
    "rmse",
]
