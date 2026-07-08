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
    normal_predictive_metrics,
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
MODEL_ORDER = [
    "Ordinary least squares",
    "RidgeCV",
    "BayesianRidge",
    "ARDRegression",
    "Bayesian Gibbs",
]
METRIC_LABELS = {
    "rmse": "RMSE",
    "r2": "R2",
    "nlpd": "NLPD",
    "crps": "CRPS",
    "interval_score_95": "95% interval score",
    "coverage_95": "95% coverage",
}
VALUE_LABEL_BBOX = {
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.82,
    "pad": 1.6,
}


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def residual_normal_std(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> float:
    residuals = y_true - y_pred
    dof = max(1, len(residuals) - n_features - 1)
    return float(np.sqrt(np.sum(residuals**2) / dof))


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
        if name in {"BayesianRidge", "ARDRegression"}:
            x_test_scaled = model.named_steps["scale"].transform(x_test)
            y_pred, predictive_std = model.named_steps["model"].predict(
                x_test_scaled,
                return_std=True,
            )
        else:
            y_pred = model.predict(x_test)
            y_train_pred = model.predict(x_train)
            predictive_std = np.full(
                shape=y_test.shape,
                fill_value=residual_normal_std(
                    y_train,
                    y_train_pred,
                    n_features=x_train.shape[1],
                ),
                dtype=float,
            )

        rows.append(
            {
                "model": name,
                **regression_metrics(y_test, y_pred),
                **normal_predictive_metrics(y_test, y_pred, predictive_std),
            }
        )

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


def _ordered_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.copy()
    ordered["model"] = pd.Categorical(ordered["model"], categories=MODEL_ORDER, ordered=True)
    return ordered.sort_values("model").reset_index(drop=True)


def _zoomed_axis_limits(
    values: pd.Series | np.ndarray,
    *,
    pad_fraction: float = 0.22,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span == 0:
        span = max(abs(high), 1.0) * 0.02
    padding = span * pad_fraction
    return low - padding, high + padding


def plot_model_comparison(summary: pd.DataFrame) -> None:
    plot_df = summary.sort_values("rmse").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.9, len(plot_df) + 1.0)

    x_model = 0.07
    x_rmse_value = 0.43
    x_rmse_track = (0.51, 0.66)
    x_r2_value = 0.76
    x_r2_track = (0.84, 0.97)
    header_y = len(plot_df) + 0.45

    ax.text(0.03, header_y, "Model", weight="bold", fontsize=13, va="center")
    ax.text(x_rmse_value, header_y, "RMSE", weight="bold", fontsize=13, ha="right", va="center")
    ax.text(
        np.mean(x_rmse_track),
        header_y,
        "within-split range",
        weight="bold",
        fontsize=10,
        ha="center",
        va="center",
        color="#6B7280",
    )
    ax.text(x_r2_value, header_y, "R2", weight="bold", fontsize=13, ha="right", va="center")
    ax.text(
        np.mean(x_r2_track),
        header_y,
        "within-split range",
        weight="bold",
        fontsize=10,
        ha="center",
        va="center",
        color="#6B7280",
    )

    rmse_values = plot_df["rmse"].to_numpy()
    r2_values = plot_df["r2"].to_numpy()
    rmse_low, rmse_high = float(rmse_values.min()), float(rmse_values.max())
    r2_low, r2_high = float(r2_values.min()), float(r2_values.max())
    rmse_span = rmse_high - rmse_low or 1.0
    r2_span = r2_high - r2_low or 1.0

    for row_index, row in plot_df.iterrows():
        y_position = len(plot_df) - row_index - 1
        model = str(row["model"])
        color = PALETTE.get(model, "#333333")

        ax.hlines(y_position - 0.45, 0.03, 0.97, color="#ECEFF3", linewidth=1.0)
        ax.scatter(0.04, y_position, color=color, s=90, edgecolor="white", linewidth=0.7)
        ax.text(x_model, y_position, model, fontsize=12.5, va="center")

        rmse_weight = "bold" if np.isclose(row["rmse"], rmse_low) else "normal"
        r2_weight = "bold" if np.isclose(row["r2"], r2_high) else "normal"
        ax.text(
            x_rmse_value,
            y_position,
            f"{row['rmse']:.3f}",
            fontsize=12.5,
            ha="right",
            va="center",
            weight=rmse_weight,
        )
        ax.text(
            x_r2_value,
            y_position,
            f"{row['r2']:.3f}",
            fontsize=12.5,
            ha="right",
            va="center",
            weight=r2_weight,
        )

        rmse_position = x_rmse_track[0] + (row["rmse"] - rmse_low) / rmse_span * (
            x_rmse_track[1] - x_rmse_track[0]
        )
        r2_position = x_r2_track[0] + (row["r2"] - r2_low) / r2_span * (
            x_r2_track[1] - x_r2_track[0]
        )
        ax.plot(x_rmse_track, [y_position, y_position], color="#D5DAE1", linewidth=5)
        ax.plot(x_r2_track, [y_position, y_position], color="#D5DAE1", linewidth=5)
        ax.scatter(rmse_position, y_position, color=color, s=105, edgecolor="white", linewidth=0.8)
        ax.scatter(r2_position, y_position, color=color, s=105, edgecolor="white", linewidth=0.8)

    ax.text(
        x_rmse_track[0],
        -0.35,
        "lower",
        fontsize=9,
        color="#6B7280",
        ha="left",
        va="center",
    )
    ax.text(
        x_rmse_track[1],
        -0.35,
        "higher",
        fontsize=9,
        color="#6B7280",
        ha="right",
        va="center",
    )
    ax.text(
        x_r2_track[0],
        -0.35,
        "lower",
        fontsize=9,
        color="#6B7280",
        ha="left",
        va="center",
    )
    ax.text(
        x_r2_track[1],
        -0.35,
        "higher",
        fontsize=9,
        color="#6B7280",
        ha="right",
        va="center",
    )
    ax.text(
        0.03,
        -0.72,
        "Values are shown in fixed columns; mini-bars use the observed fixed-split range and are not zero-based.",
        fontsize=9.5,
        color="#6B7280",
        ha="left",
        va="center",
    )
    fig.suptitle("Fixed-split point metrics", y=0.98, fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fixed_split_point_metrics.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fixed_split_probabilistic_metrics(summary: pd.DataFrame) -> None:
    plot_df = _ordered_summary(summary)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    metrics = ("nlpd", "crps", "interval_score_95", "coverage_95")
    subtitles = {
        "nlpd": "Negative log predictive density, lower is better",
        "crps": "CRPS, lower is better",
        "interval_score_95": "95% interval score, lower is better",
        "coverage_95": "95% coverage, target = 0.95",
    }
    y_pos = np.arange(len(plot_df))

    for ax, metric in zip(axes.ravel(), metrics):
        values = plot_df[metric].to_numpy()
        colors = [PALETTE[model] for model in plot_df["model"].astype(str)]
        ax.scatter(
            values,
            y_pos,
            color=colors,
            s=90,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        if metric == "coverage_95":
            ax.axvline(0.95, color="#222222", linestyle="--", linewidth=1.2)
        x_min, x_max = _zoomed_axis_limits(values)
        if metric == "coverage_95":
            x_min = min(x_min, 0.95 - 0.01)
            x_max = max(x_max, 0.95 + 0.01)
        ax.set_xlim(x_min, x_max)
        offset = (x_max - x_min) * 0.018
        for y_index, value in enumerate(values):
            ha = "left"
            x_text = value + offset
            if x_text > x_max - offset:
                ha = "right"
                x_text = value - offset
            ax.text(
                x_text,
                y_index,
                f"{value:.3f}",
                va="center",
                ha=ha,
                fontsize=10,
                bbox=VALUE_LABEL_BBOX,
            )

        ax.set_title(subtitles[metric])
        ax.set_xlabel(METRIC_LABELS[metric])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["model"])
        ax.grid(True, axis="x", alpha=0.25)
        ax.grid(False, axis="y")

    fig.suptitle("Fixed-split probabilistic metrics", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "fixed_split_probabilistic_metrics.png",
        dpi=220,
        bbox_inches="tight",
    )
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
    plot_fixed_split_probabilistic_metrics(model_summary)
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
