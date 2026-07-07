import numpy as np

from bayeslinreg import (
    autocorrelation_1d,
    effective_sample_size_1d,
    summarize_mcmc_samples,
)


def test_autocorrelation_1d_known_values():
    samples = np.array([1.0, 2.0, 3.0, 4.0])

    assert autocorrelation_1d(samples, 0) == 1.0
    assert np.isclose(autocorrelation_1d(samples, 1), 0.25)
    assert np.isclose(autocorrelation_1d(samples, 3), -0.45)
    assert np.isnan(autocorrelation_1d(samples, 4))


def test_autocorrelation_1d_constant_chain_is_undefined_after_lag_zero():
    samples = np.ones(5)

    assert autocorrelation_1d(samples, 0) == 1.0
    assert np.isnan(autocorrelation_1d(samples, 1))


def test_effective_sample_size_1d_uses_positive_autocorrelation_lags():
    samples = np.array([1.0, 2.0, 3.0, 4.0])

    assert np.isclose(effective_sample_size_1d(samples, max_lag=1), 4.0 / 1.5)

    alternating = np.array([1.0, -1.0] * 20)
    assert effective_sample_size_1d(alternating, max_lag=5) == alternating.size


def test_summarize_mcmc_samples_returns_expected_columns():
    samples = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0],
            [6.0, 12.0],
        ]
    )

    rows = summarize_mcmc_samples(["alpha", "beta"], samples)

    assert [row["parameter"] for row in rows] == ["alpha", "beta"]
    assert np.isclose(rows[0]["posterior_mean"], 3.5)
    assert np.isclose(rows[1]["posterior_mean"], 7.0)
    assert rows[0]["posterior_sd"] > 0
    assert rows[0]["ess"] >= 1
    assert "autocorr_lag20" in rows[0]
