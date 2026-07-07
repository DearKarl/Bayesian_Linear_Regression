import numpy as np
import pandas as pd

from bayeslinreg import pairwise_against_reference, summarize_repeated_splits


def make_raw_results() -> pd.DataFrame:
    rows = []
    values = {
        "Bayesian Gibbs": {
            "rmse": [1.0, 2.0, 3.0],
            "r2": [0.90, 0.70, 0.50],
            "nlpd": [1.0, 1.0, 1.0],
        },
        "Ordinary least squares": {
            "rmse": [2.0, 3.0, 4.0],
            "r2": [0.80, 0.70, 0.60],
            "nlpd": [1.5, 0.5, 1.0],
        },
    }
    for model, metrics in values.items():
        for split in range(3):
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "rmse": metrics["rmse"][split],
                    "r2": metrics["r2"][split],
                    "nlpd": metrics["nlpd"][split],
                }
            )
    return pd.DataFrame(rows)


def test_summarize_repeated_splits_returns_model_metric_rows():
    summary = summarize_repeated_splits(
        make_raw_results(),
        metrics=("rmse", "r2", "nlpd"),
        model_order=("Bayesian Gibbs", "Ordinary least squares"),
    )

    gibbs_rmse = summary[
        (summary["model"] == "Bayesian Gibbs") & (summary["metric"] == "rmse")
    ].iloc[0]
    assert gibbs_rmse["n_repeats"] == 3
    assert gibbs_rmse["mean"] == 2.0
    assert np.isclose(gibbs_rmse["std"], 1.0)
    assert gibbs_rmse["ci95_lower"] < gibbs_rmse["mean"] < gibbs_rmse["ci95_upper"]


def test_pairwise_against_reference_uses_baseline_minus_reference_difference():
    pairwise = pairwise_against_reference(
        make_raw_results(),
        reference_model="Bayesian Gibbs",
        metrics=("rmse", "r2", "nlpd"),
        model_order=("Bayesian Gibbs", "Ordinary least squares"),
    )

    ols_rmse = pairwise[
        (pairwise["comparison_model"] == "Ordinary least squares")
        & (pairwise["metric"] == "rmse")
    ].iloc[0]
    assert ols_rmse["mean_difference"] == 1.0
    assert ols_rmse["std_difference"] == 0.0
    assert ols_rmse["ci95_lower"] == 1.0
    assert ols_rmse["ci95_upper"] == 1.0
    assert ols_rmse["win_rate_for_gibbs"] == 1.0

    ols_r2 = pairwise[
        (pairwise["comparison_model"] == "Ordinary least squares")
        & (pairwise["metric"] == "r2")
    ].iloc[0]
    assert np.isclose(ols_r2["mean_difference"], 0.0)
    assert np.isclose(ols_r2["win_rate_for_gibbs"], 0.5)

    ols_nlpd = pairwise[
        (pairwise["comparison_model"] == "Ordinary least squares")
        & (pairwise["metric"] == "nlpd")
    ].iloc[0]
    assert np.isclose(ols_nlpd["mean_difference"], 0.0)
    assert np.isclose(ols_nlpd["win_rate_for_gibbs"], 0.5)
