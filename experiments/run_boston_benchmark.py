"""Run the Boston Housing ordinary vs Bayesian linear regression benchmark."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import ARDRegression, BayesianRidge, LinearRegression, RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*encountered in matmul.*",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bayeslinreg import (  # noqa: E402
    BayesianLinearRegressionGibbs,
    FEATURE_DESCRIPTIONS,
    crps_from_samples,
    interval_metrics,
    interval_score,
    load_boston_csv,
    make_feature_target,
    negative_log_predictive_density_mixture,
    regression_metrics,
    rmse,
    summarize_mcmc_samples,
)

SEED = 42
TAU_GRID = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
TABLES_DIR = REPO_ROOT / "reports" / "tables"

PALETTE = {
    "Ordinary least squares": "#4C78A8",
    "RidgeCV": "#54A24B",
    "BayesianRidge": "#B279A2",
    "ARDRegression": "#E45756",
    "Bayesian Gibbs": "#F58518",
}


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def fit_gibbs_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    tau2: float,
    n_iter: int,
    burn_in: int,
    random_state: int,
) -> tuple[BayesianLinearRegressionGibbs, np.ndarray, object, StandardScaler]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = BayesianLinearRegressionGibbs(
        tau2=tau2,
        n_iter=n_iter,
        burn_in=burn_in,
        random_state=random_state,
    ).fit(x_train_scaled, y_train)
    predictive = model.posterior_predictive(x_test_scaled, random_state=random_state + 10_000)
    return model, model.predict(x_test_scaled), predictive, scaler


def evaluate_tau_cv(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tau_grid: np.ndarray,
    n_splits: int = 5,
) -> pd.DataFrame:
    rows = []
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tau2 in tau_grid:
        for fold, (train_idx, val_idx) in enumerate(folds.split(x), start=1):
            _, y_pred, _, _ = fit_gibbs_model(
                x[train_idx],
                y[train_idx],
                x[val_idx],
                tau2=float(tau2),
                n_iter=1200,
                burn_in=300,
                random_state=SEED + fold + int(np.log10(tau2) * 17 + 100),
            )
            rows.append(
                {
                    "tau2": float(tau2),
                    "fold": fold,
                    "rmse": rmse(y[val_idx], y_pred),
                }
            )
    cv = pd.DataFrame(rows)
    summary = (
        cv.groupby("tau2", as_index=False)
        .agg(cv_rmse_mean=("rmse", "mean"), cv_rmse_std=("rmse", "std"))
        .sort_values("tau2")
    )
    summary.to_csv(TABLES_DIR / "tau_cv_summary.csv", index=False)
    cv.to_csv(TABLES_DIR / "tau_cv_folds.csv", index=False)
    return summary


def benchmark_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    best_tau2: float,
) -> tuple[pd.DataFrame, pd.DataFrame, BayesianLinearRegressionGibbs, StandardScaler]:
    rows = []

    classical_models = {
        "Ordinary least squares": Pipeline(
            [("scale", StandardScaler()), ("model", LinearRegression())]
        ),
        "RidgeCV": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 80))),
            ]
        ),
        "BayesianRidge": Pipeline(
            [("scale", StandardScaler()), ("model", BayesianRidge(compute_score=True))]
        ),
        "ARDRegression": Pipeline(
            [("scale", StandardScaler()), ("model", ARDRegression())]
        ),
    }

    for name, model in classical_models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rows.append({"model": name, **regression_metrics(y_test, y_pred)})

    gibbs_model, gibbs_pred, predictive, scaler = fit_gibbs_model(
        x_train,
        y_train,
        x_test,
        tau2=best_tau2,
        n_iter=3200,
        burn_in=800,
        random_state=SEED,
    )
    x_test_scaled = scaler.transform(x_test)
    location_samples = gibbs_model.posterior_predictive(
        x_test_scaled,
        include_noise=False,
    ).samples
    rows.append(
        {
            "model": "Bayesian Gibbs",
            **regression_metrics(y_test, gibbs_pred),
            **interval_metrics(y_test, predictive.lower, predictive.upper),
            "nlpd": negative_log_predictive_density_mixture(
                y_test,
                location_samples,
                gibbs_model.sigma2_samples_,
            ),
            "crps": crps_from_samples(y_test, predictive.samples),
            "interval_score_95": interval_score(
                y_test,
                predictive.lower,
                predictive.upper,
                alpha=0.05,
            ),
        }
    )

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    predictions = pd.DataFrame(
        {
            "actual": y_test,
            "bayesian_mean": predictive.mean,
            "bayesian_lower_95": predictive.lower,
            "bayesian_upper_95": predictive.upper,
            "bayesian_point": gibbs_pred,
            "residual": y_test - gibbs_pred,
        }
    )
    predictions.to_csv(TABLES_DIR / "test_predictions.csv", index=False)
    return summary, predictions, gibbs_model, scaler


def coefficient_table(
    model: BayesianLinearRegressionGibbs,
    feature_names: list[str],
) -> pd.DataFrame:
    summary = pd.DataFrame(model.coefficient_summary(feature_names))
    summary["abs_mean"] = summary["mean"].abs()
    summary["description"] = summary["feature"].map(FEATURE_DESCRIPTIONS).fillna("")
    summary = summary.sort_values("abs_mean", ascending=False)
    summary.to_csv(TABLES_DIR / "posterior_coefficients.csv", index=False)
    return summary


def mcmc_diagnostics_table(
    model: BayesianLinearRegressionGibbs,
    feature_names: list[str],
) -> pd.DataFrame:
    parameter_names = ["intercept", *feature_names, "sigma2"]
    samples = np.column_stack([model.beta_samples_, model.sigma2_samples_])
    diagnostics = pd.DataFrame(summarize_mcmc_samples(parameter_names, samples))
    diagnostics.to_csv(TABLES_DIR / "mcmc_diagnostics.csv", index=False)
    return diagnostics


def training_size_experiment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tau2: float,
    repeats: int = 20,
) -> pd.DataFrame:
    rows = []
    train_fracs = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    rng = np.random.default_rng(SEED)
    n_samples = len(y)

    for frac in train_fracs:
        for repeat in range(repeats):
            shuffled = rng.permutation(n_samples)
            n_train = int(frac * n_samples)
            train_idx = shuffled[:n_train]
            test_idx = shuffled[n_train:]

            ols = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
            ols.fit(x[train_idx], y[train_idx])
            ols_pred = ols.predict(x[test_idx])
            rows.append(
                {
                    "model": "Ordinary least squares",
                    "train_fraction": frac,
                    "repeat": repeat,
                    "rmse": rmse(y[test_idx], ols_pred),
                }
            )

            _, bayes_pred, _, _ = fit_gibbs_model(
                x[train_idx],
                y[train_idx],
                x[test_idx],
                tau2=tau2,
                n_iter=900,
                burn_in=250,
                random_state=SEED + repeat + int(frac * 1000),
            )
            rows.append(
                {
                    "model": "Bayesian Gibbs",
                    "train_fraction": frac,
                    "repeat": repeat,
                    "rmse": rmse(y[test_idx], bayes_pred),
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(TABLES_DIR / "training_size_raw.csv", index=False)
    summary = (
        results.groupby(["model", "train_fraction"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"))
        .sort_values(["model", "train_fraction"])
    )
    summary.to_csv(TABLES_DIR / "training_size_summary.csv", index=False)
    return summary


def bias_variance_experiment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tau_grid: np.ndarray,
    simulations: int = 25,
) -> pd.DataFrame:
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=SEED, shuffle=True
    )
    rng = np.random.default_rng(SEED + 900)
    rows = []

    for tau2 in tau_grid:
        prediction_matrix = np.empty((len(y_val), simulations), dtype=float)
        for simulation in range(simulations):
            boot_idx = rng.choice(len(y_train), size=len(y_train), replace=True)
            _, y_pred, _, _ = fit_gibbs_model(
                x_train[boot_idx],
                y_train[boot_idx],
                x_val,
                tau2=float(tau2),
                n_iter=700,
                burn_in=200,
                random_state=SEED + simulation + int(tau2 * 3),
            )
            prediction_matrix[:, simulation] = y_pred

        mean_prediction = prediction_matrix.mean(axis=1)
        bias_squared = float(np.mean((mean_prediction - y_val) ** 2))
        variance = float(np.mean(np.var(prediction_matrix, axis=1, ddof=1)))
        rows.append(
            {
                "tau2": float(tau2),
                "bias_squared": bias_squared,
                "variance": variance,
                "mse": bias_squared + variance,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(TABLES_DIR / "bias_variance.csv", index=False)
    return results


def legacy_feature_sensitivity(x: pd.DataFrame, y: pd.Series, *, tau2: float) -> pd.DataFrame:
    rows = []
    for label, drop_b in [
        ("all_features_legacy", False),
        ("drop_b_feature", True),
    ]:
        x_variant, y_variant = make_feature_target(
            pd.concat([x, y], axis=1),
            drop_legacy_race_feature=drop_b,
        )
        x_train, x_test, y_train, y_test = train_test_split(
            x_variant.to_numpy(),
            y_variant.to_numpy(),
            test_size=0.2,
            random_state=SEED,
            shuffle=True,
        )
        _, y_pred, predictive, _ = fit_gibbs_model(
            x_train,
            y_train,
            x_test,
            tau2=tau2,
            n_iter=2200,
            burn_in=600,
            random_state=SEED + (10 if drop_b else 0),
        )
        rows.append(
            {
                "feature_set": label,
                **regression_metrics(y_test, y_pred),
                **interval_metrics(y_test, predictive.lower, predictive.upper),
                "n_features": x_variant.shape[1],
            }
        )
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(TABLES_DIR / "legacy_feature_sensitivity.csv", index=False)
    return sensitivity


def plot_model_comparison(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = summary.sort_values("rmse")["model"].tolist()
    sns.barplot(
        data=summary,
        x="rmse",
        y="model",
        hue="model",
        order=order,
        palette=PALETTE,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Held-out RMSE")
    axes[0].set_xlabel("RMSE, lower is better")
    axes[0].set_ylabel("")
    for patch in axes[0].patches:
        width = patch.get_width()
        axes[0].text(width + 0.05, patch.get_y() + patch.get_height() / 2, f"{width:.2f}")

    sns.barplot(
        data=summary,
        x="r2",
        y="model",
        hue="model",
        order=order,
        palette=PALETTE,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Held-out R2")
    axes[1].set_xlabel("R2, higher is better")
    axes[1].set_ylabel("")
    for patch in axes[1].patches:
        width = patch.get_width()
        axes[1].text(width + 0.01, patch.get_y() + patch.get_height() / 2, f"{width:.2f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tau_sensitivity(cv: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x_values = np.log10(cv["tau2"].to_numpy())
    means = cv["cv_rmse_mean"].to_numpy()
    stds = cv["cv_rmse_std"].to_numpy()
    ax.plot(x_values, means, marker="o", color="#F58518", linewidth=2.5)
    ax.fill_between(x_values, means - stds, means + stds, color="#F58518", alpha=0.18)
    ax.set_title("Prior variance sensitivity")
    ax.set_xlabel("log10(tau^2)")
    ax.set_ylabel("5-fold CV RMSE")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tau_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.scatter(
        predictions["actual"],
        predictions["bayesian_point"],
        s=50,
        alpha=0.82,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.5,
    )
    lims = [
        min(predictions["actual"].min(), predictions["bayesian_point"].min()) - 2,
        max(predictions["actual"].max(), predictions["bayesian_point"].max()) + 2,
    ]
    ax.plot(lims, lims, color="#333333", linestyle="--", linewidth=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Bayesian predictions vs actual values")
    ax.set_xlabel("Actual MEDV")
    ax.set_ylabel("Predicted MEDV")

    ordered = predictions.sort_values("actual").reset_index(drop=True)
    subset = ordered.iloc[np.linspace(0, len(ordered) - 1, min(35, len(ordered))).astype(int)]
    x_axis = np.arange(len(subset))
    axes[1].errorbar(
        x_axis,
        subset["bayesian_mean"],
        yerr=[
            subset["bayesian_mean"] - subset["bayesian_lower_95"],
            subset["bayesian_upper_95"] - subset["bayesian_mean"],
        ],
        fmt="o",
        color="#F58518",
        ecolor="#B279A2",
        elinewidth=1.5,
        capsize=2,
        label="Posterior predictive 95% interval",
    )
    axes[1].scatter(x_axis, subset["actual"], color="#333333", s=25, label="Actual")
    axes[1].set_title("Posterior predictive intervals")
    axes[1].set_xlabel("Held-out samples sorted by target")
    axes[1].set_ylabel("MEDV")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "predictions_and_intervals.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_coefficients(coefficients: pd.DataFrame) -> None:
    plot_df = coefficients[coefficients["feature"] != "intercept"].copy()
    plot_df = plot_df.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(plot_df))
    ax.errorbar(
        plot_df["mean"],
        y_pos,
        xerr=[plot_df["mean"] - plot_df["lower"], plot_df["upper"] - plot_df["mean"]],
        fmt="o",
        color="#4C78A8",
        ecolor="#F58518",
        elinewidth=1.8,
        capsize=3,
    )
    ax.axvline(0, color="#333333", linewidth=1, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"])
    ax.set_title("Posterior coefficient intervals")
    ax.set_xlabel("Standardized coefficient")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "posterior_coefficients.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mcmc_trace_diagnostics(
    model: BayesianLinearRegressionGibbs,
    feature_names: list[str],
) -> None:
    coefficient_means = model.beta_samples_.mean(axis=0)
    top_feature_indices = np.argsort(np.abs(coefficient_means[1:]))[::-1][:5] + 1

    traces: list[tuple[str, np.ndarray]] = [("intercept", model.beta_samples_[:, 0])]
    traces.append(("sigma2", model.sigma2_samples_))
    for index in top_feature_indices:
        traces.append((feature_names[index - 1], model.beta_samples_[:, index]))

    fig, axes = plt.subplots(
        len(traces),
        1,
        figsize=(12, 2.0 * len(traces)),
        sharex=True,
    )
    if len(traces) == 1:
        axes = [axes]

    draw_index = np.arange(model.beta_samples_.shape[0])
    for ax, (name, values) in zip(axes, traces):
        ax.plot(draw_index, values, color="#4C78A8", linewidth=0.7, alpha=0.9)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)

    axes[0].set_title("Single-chain Gibbs trace diagnostics")
    axes[-1].set_xlabel("Post-burn-in draw")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mcmc_trace_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_size(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in summary.groupby("model"):
        color = PALETTE.get(model, "#333333")
        x_values = group["train_fraction"].to_numpy() * 100
        means = group["rmse_mean"].to_numpy()
        stds = group["rmse_std"].to_numpy()
        ax.plot(x_values, means, marker="o", linewidth=2.5, color=color, label=model)
        ax.fill_between(x_values, means - stds, means + stds, color=color, alpha=0.14)
    ax.set_title("Small-data robustness")
    ax.set_xlabel("Training data used (%)")
    ax.set_ylabel("RMSE on held-out remainder")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_size_robustness.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_bias_variance(results: pd.DataFrame) -> None:
    long = results.melt(
        id_vars="tau2",
        value_vars=["bias_squared", "variance", "mse"],
        var_name="component",
        value_name="value",
    )
    labels = {
        "bias_squared": "Bias^2",
        "variance": "Variance",
        "mse": "MSE",
    }
    long["log10_tau2"] = np.log10(long["tau2"])
    long["component_label"] = long["component"].map(labels)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=long,
        x="log10_tau2",
        y="value",
        hue="component_label",
        style="component_label",
        markers=True,
        dashes=False,
        linewidth=2.3,
        ax=ax,
    )
    ax.set_title("Bias-variance trade-off under Bayesian shrinkage")
    ax.set_xlabel("log10(tau^2)")
    ax.set_ylabel("Component value")
    ax.legend(title="", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bias_variance_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(
        data=predictions,
        x="bayesian_point",
        y="residual",
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.5,
        ax=axes[0],
    )
    axes[0].axhline(0, color="#333333", linestyle="--", linewidth=1.3)
    axes[0].set_title("Residuals vs fitted values")
    axes[0].set_xlabel("Bayesian fitted value")
    axes[0].set_ylabel("Residual")

    sns.histplot(predictions["residual"], kde=True, color="#F58518", ax=axes[1])
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "residual_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_metrics_json(
    *,
    best_tau2: float,
    model_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    payload = {
        "dataset": "Boston Housing legacy benchmark",
        "n_rows": 506,
        "target": "medv",
        "random_seed": SEED,
        "best_tau2": best_tau2,
        "model_comparison": model_summary.to_dict(orient="records"),
        "legacy_feature_sensitivity": sensitivity.to_dict(orient="records"),
    }
    (TABLES_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")

    data = load_boston_csv(REPO_ROOT / "BostonHousing_data.csv")
    x_df, y_series = make_feature_target(data)
    x = x_df.to_numpy()
    y = y_series.to_numpy()
    feature_names = x_df.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, shuffle=True
    )

    tau_cv = evaluate_tau_cv(x_train, y_train, tau_grid=TAU_GRID)
    best_tau2 = float(tau_cv.loc[tau_cv["cv_rmse_mean"].idxmin(), "tau2"])

    model_summary, predictions, gibbs_model, _ = benchmark_models(
        x_train,
        y_train,
        x_test,
        y_test,
        best_tau2=best_tau2,
    )
    coefficients = coefficient_table(gibbs_model, feature_names)
    mcmc_diagnostics_table(gibbs_model, feature_names)
    training_summary = training_size_experiment(x, y, tau2=best_tau2)
    bias_variance = bias_variance_experiment(x, y, tau_grid=TAU_GRID)
    sensitivity = legacy_feature_sensitivity(x_df, y_series, tau2=best_tau2)

    plot_model_comparison(model_summary)
    plot_tau_sensitivity(tau_cv)
    plot_predictions(predictions)
    plot_coefficients(coefficients)
    plot_mcmc_trace_diagnostics(gibbs_model, feature_names)
    plot_training_size(training_summary)
    plot_bias_variance(bias_variance)
    plot_residuals(predictions)
    write_metrics_json(
        best_tau2=best_tau2,
        model_summary=model_summary,
        sensitivity=sensitivity,
    )

    print("Benchmark complete")
    print(f"Best tau^2: {best_tau2:g}")
    print(model_summary.to_string(index=False))


if __name__ == "__main__":
    main()
