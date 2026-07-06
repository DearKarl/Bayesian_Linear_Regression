import numpy as np

from bayeslinreg import BayesianLinearRegressionGibbs, load_boston_csv, make_feature_target


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
