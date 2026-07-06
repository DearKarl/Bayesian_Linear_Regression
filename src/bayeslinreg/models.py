"""Bayesian linear regression with a conjugate Gibbs sampler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PosteriorSummary:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    samples: np.ndarray


class BayesianLinearRegressionGibbs:
    """Bayesian linear regression using Gibbs sampling.

    The sampler follows the coursework/report model but exposes it as a reusable
    Python estimator: a Gaussian prior over beta, an inverse-gamma prior over
    sigma^2, and closed-form conditional draws for both blocks.
    """

    def __init__(
        self,
        *,
        tau2: float = 10.0,
        a0: float = 0.01,
        b0: float = 0.01,
        n_iter: int = 2500,
        burn_in: int = 500,
        random_state: int | None = 42,
        intercept_prior_var: float = 1e6,
    ) -> None:
        if tau2 <= 0:
            raise ValueError("tau2 must be positive")
        if n_iter <= burn_in:
            raise ValueError("n_iter must be greater than burn_in")

        self.tau2 = float(tau2)
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.n_iter = int(n_iter)
        self.burn_in = int(burn_in)
        self.random_state = random_state
        self.intercept_prior_var = float(intercept_prior_var)

    @staticmethod
    def _add_intercept(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("X must be a 2D array")
        return np.column_stack([np.ones(x.shape[0]), x])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BayesianLinearRegressionGibbs":
        x_design = self._add_intercept(x)
        y = np.asarray(y, dtype=float).ravel()
        if x_design.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible lengths")
        if not np.isfinite(x_design).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values")

        rng = np.random.default_rng(self.random_state)
        n_obs, n_params = x_design.shape

        beta0 = np.zeros(n_params)
        prior_var = np.full(n_params, self.tau2)
        prior_var[0] = self.intercept_prior_var
        v0_inv = np.diag(1.0 / prior_var)

        xtx = np.dot(x_design.T, x_design)
        xty = np.dot(x_design.T, y)
        beta_current = beta0.copy()
        sigma2_current = float(np.var(y, ddof=1))

        beta_samples = np.empty((self.n_iter, n_params), dtype=float)
        sigma2_samples = np.empty(self.n_iter, dtype=float)

        for iteration in range(self.n_iter):
            precision = v0_inv + xtx / sigma2_current
            covariance = np.linalg.inv(precision)
            covariance = 0.5 * (covariance + covariance.T)
            mean = np.dot(covariance, np.dot(v0_inv, beta0) + xty / sigma2_current)
            beta_current = rng.multivariate_normal(mean, covariance)

            residual = y - np.dot(x_design, beta_current)
            prior_delta = beta_current - beta0
            prior_quadratic = float(np.dot(prior_delta, np.dot(v0_inv, prior_delta)))
            shape = self.a0 + 0.5 * n_obs
            rate = self.b0 + 0.5 * residual @ residual + 0.5 * prior_quadratic
            sigma2_current = 1.0 / rng.gamma(shape=shape, scale=1.0 / rate)

            beta_samples[iteration] = beta_current
            sigma2_samples[iteration] = sigma2_current

        self.beta_samples_ = beta_samples[self.burn_in :]
        self.sigma2_samples_ = sigma2_samples[self.burn_in :]
        self.beta_mean_ = self.beta_samples_.mean(axis=0)
        self.sigma2_mean_ = float(self.sigma2_samples_.mean())
        self.beta_cov_ = np.cov(self.beta_samples_, rowvar=False)
        self.n_features_in_ = x_design.shape[1] - 1
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        x_design = self._add_intercept(x)
        return np.dot(x_design, self.beta_mean_)

    def posterior_predictive(
        self,
        x: np.ndarray,
        *,
        level: float = 0.95,
        include_noise: bool = True,
        random_state: int | None = None,
    ) -> PosteriorSummary:
        self._check_is_fitted()
        x_design = self._add_intercept(x)
        samples = np.dot(self.beta_samples_, x_design.T)

        if include_noise:
            rng = np.random.default_rng(self.random_state if random_state is None else random_state)
            noise = rng.normal(
                loc=0.0,
                scale=np.sqrt(self.sigma2_samples_)[:, None],
                size=samples.shape,
            )
            samples = samples + noise

        alpha = 1.0 - level
        return PosteriorSummary(
            mean=samples.mean(axis=0),
            lower=np.quantile(samples, alpha / 2.0, axis=0),
            upper=np.quantile(samples, 1.0 - alpha / 2.0, axis=0),
            samples=samples,
        )

    def coefficient_summary(
        self,
        feature_names: list[str],
        *,
        level: float = 0.95,
    ) -> dict[str, np.ndarray | list[str]]:
        self._check_is_fitted()
        if len(feature_names) != self.n_features_in_:
            raise ValueError("feature_names length does not match fitted features")

        names = ["intercept", *feature_names]
        alpha = 1.0 - level
        return {
            "feature": names,
            "mean": self.beta_samples_.mean(axis=0),
            "lower": np.quantile(self.beta_samples_, alpha / 2.0, axis=0),
            "upper": np.quantile(self.beta_samples_, 1.0 - alpha / 2.0, axis=0),
            "posterior_sd": self.beta_samples_.std(axis=0, ddof=1),
        }

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "beta_samples_"):
            raise RuntimeError("The model must be fitted before prediction")
