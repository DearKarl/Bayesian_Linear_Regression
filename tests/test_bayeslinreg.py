import math

import numpy as np

from bayeslinreg import (
    BayesianLinearRegressionGibbs,
    crps_from_samples,
    interval_score,
    load_boston_csv,
    make_feature_target,
    negative_log_predictive_density_mixture,
)


def test_load_boston_csv_shape():
    data = load_boston_csv()
    assert data.shape[0] == 506
    assert "medv" in data.columns
    assert data.isna().sum().sum() == 0


def test_make_feature_target_can_drop_legacy_race_feature():
    data = load_boston_csv()
    x_all, y = make_feature_target(data)
    x_screened, _ = make_feature_target(data, drop_legacy_race_feature=True)
    assert len(y) == len(data)
    assert "b" in x_all.columns
    assert "b" not in x_screened.columns
    assert x_screened.shape[1] == x_all.shape[1] - 1


def test_gibbs_sampler_shapes_and_predictions():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 3))
    beta = np.array([1.5, -2.0, 0.7])
    y = 3.0 + x @ beta + rng.normal(scale=0.3, size=80)

    model = BayesianLinearRegressionGibbs(
        tau2=10,
        n_iter=160,
        burn_in=40,
        random_state=7,
    ).fit(x, y)

    point = model.predict(x[:5])
    predictive = model.posterior_predictive(x[:5], random_state=8)

    assert model.beta_samples_.shape == (120, 4)
    assert point.shape == (5,)
    assert predictive.mean.shape == (5,)
    assert np.all(predictive.lower < predictive.upper)


def test_interval_score_rewards_narrow_covered_intervals():
    y_true = np.array([0.0, 1.0])
    lower = np.array([-1.0, 0.0])
    upper = np.array([1.0, 2.0])

    assert interval_score(y_true, lower, upper, alpha=0.05) == 2.0

    wider = interval_score(y_true, lower - 1.0, upper + 1.0, alpha=0.05)
    assert wider > 2.0

    missed = interval_score(np.array([3.0]), np.array([0.0]), np.array([1.0]), alpha=0.05)
    assert missed == 81.0


def test_crps_from_samples_matches_point_mass_absolute_error():
    y_true = np.array([1.0, 3.0])
    samples = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])

    assert crps_from_samples(y_true, samples) == 1.0


def test_negative_log_predictive_density_mixture_matches_standard_normal():
    y_true = np.array([0.0])
    location_samples = np.zeros((4, 1))
    sigma2_samples = np.ones(4)

    expected = 0.5 * math.log(2.0 * math.pi)
    actual = negative_log_predictive_density_mixture(
        y_true,
        location_samples,
        sigma2_samples,
    )
    assert np.isclose(actual, expected)


def test_negative_log_predictive_density_mixture_is_stable_for_tiny_density():
    y_true = np.array([100.0])
    location_samples = np.zeros((5, 1))
    sigma2_samples = np.ones(5)

    actual = negative_log_predictive_density_mixture(
        y_true,
        location_samples,
        sigma2_samples,
    )
    assert np.isfinite(actual)
    assert actual > 1000.0
