"""Run repeated train/test split comparisons for Part I linear models."""

from __future__ import annotations

import argparse
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
from sklearn.model_selection import train_test_split
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
    REPEATED_SPLIT_METRICS,
    crps_from_samples,
    interval_metrics,
    interval_score,
    load_boston_csv,
    make_feature_target,
    negative_log_predictive_density_mixture,
    normal_predictive_metrics,
    pairwise_against_reference,
    regression_metrics,
    summarize_repeated_splits,
)

SEED = 42
TEST_SIZE = 0.2
N_REPEATS = 30
GIBBS_TAU2 = 10.0  # Selected by the main benchmark; fixed here to test split stability.
GIBBS_N_ITER = 3200
GIBBS_BURN_IN = 800
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
TABLES_DIR = REPO_ROOT / "reports" / "tables"

MODEL_ORDER = [
    "Ordinary least squares",
    "RidgeCV",
    "BayesianRidge",
    "ARDRegression",
    "Bayesian Gibbs",
]

PALETTE = {
    "Ordinary least squares": "#4C78A8",
    "RidgeCV": "#54A24B",
    "BayesianRidge": "#B279A2",
    "ARDRegression": "#E45756",
    "Bayesian Gibbs": "#F58518",
}

PLOT_METRICS = ("rmse", "nlpd", "crps", "interval_score_95")
METRIC_LABELS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R2",
    "coverage_95": "95% coverage",
    "mean_interval_width": "95% interval width",
    "nlpd": "NLPD",
    "crps": "CRPS",
    "interval_score_95": "95% interval score",
}
COMPARISON_ORDER = [model for model in MODEL_ORDER if model != "Bayesian Gibbs"]


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


def make_classical_models() -> dict[str, Pipeline]:
    return {
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


def evaluate_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    split: int,
    seed: int,
    test_size: float,
) -> list[dict[str, float | int | str]]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    rows: list[dict[str, float | int | str]] = []

    for name, model in make_classical_models().items():
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
                "split": split,
                "seed": seed,
                "model": name,
                **regression_metrics(y_test, y_pred),
                **normal_predictive_metrics(y_test, y_pred, predictive_std),
            }
        )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    gibbs = BayesianLinearRegressionGibbs(
        tau2=GIBBS_TAU2,
        n_iter=GIBBS_N_ITER,
        burn_in=GIBBS_BURN_IN,
        random_state=seed + 10_000,
    ).fit(x_train_scaled, y_train)
    gibbs_pred = gibbs.predict(x_test_scaled)
    predictive = gibbs.posterior_predictive(
        x_test_scaled,
        random_state=seed + 20_000,
    )
    location_samples = gibbs.posterior_predictive(
        x_test_scaled,
        include_noise=False,
    ).samples
    rows.append(
        {
            "split": split,
            "seed": seed,
            "model": "Bayesian Gibbs",
            **regression_metrics(y_test, gibbs_pred),
            **interval_metrics(y_test, predictive.lower, predictive.upper),
            "nlpd": negative_log_predictive_density_mixture(
                y_test,
                location_samples,
                gibbs.sigma2_samples_,
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
    return rows


def run_repeated_split_comparison(
    *,
    n_repeats: int = N_REPEATS,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> pd.DataFrame:
    data = load_boston_csv()
    x_frame, y_series = make_feature_target(data)
    x = x_frame.to_numpy()
    y = y_series.to_numpy()

    rows: list[dict[str, float | int | str]] = []
    for split in range(n_repeats):
        split_seed = seed + split
        rows.extend(
            evaluate_split(
                x,
                y,
                split=split,
                seed=split_seed,
                test_size=test_size,
            )
        )
        print(f"completed split {split + 1}/{n_repeats}")
    return pd.DataFrame(rows)


def plot_metric_distributions(raw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    for ax, metric in zip(axes.ravel(), PLOT_METRICS):
        sns.boxplot(
            data=raw,
            x=metric,
            y="model",
            hue="model",
            order=MODEL_ORDER,
            palette=PALETTE,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=raw,
            x=metric,
            y="model",
            order=MODEL_ORDER,
            color="#222222",
            alpha=0.35,
            size=3,
            ax=ax,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Lower is better")
        ax.set_ylabel("")
        ax.grid(True, axis="x", alpha=0.25)
    fig.suptitle("Repeated split metric distributions", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "repeated_split_metric_distributions.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


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


def plot_repeated_split_mean_ci(summary: pd.DataFrame) -> None:
    plot_df = summary[summary["metric"].isin(PLOT_METRICS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    y_pos = np.arange(len(MODEL_ORDER))

    for ax, metric in zip(axes.ravel(), PLOT_METRICS):
        metric_df = plot_df[plot_df["metric"] == metric].copy()
        metric_df["model"] = pd.Categorical(
            metric_df["model"],
            categories=MODEL_ORDER,
            ordered=True,
        )
        metric_df = metric_df.sort_values("model")
        means = metric_df["mean"].to_numpy()
        xerr = np.vstack(
            [
                means - metric_df["ci95_lower"].to_numpy(),
                metric_df["ci95_upper"].to_numpy() - means,
            ]
        )
        colors = [PALETTE[model] for model in metric_df["model"].astype(str)]
        ax.errorbar(
            means,
            y_pos,
            xerr=xerr,
            fmt="none",
            ecolor="#333333",
            elinewidth=1.2,
            capsize=3,
            zorder=1,
        )
        ax.scatter(
            means,
            y_pos,
            color=colors,
            s=75,
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )
        x_min = float(np.min(metric_df["ci95_lower"]))
        x_max = float(np.max(metric_df["ci95_upper"]))
        axis_min, axis_max = _zoomed_axis_limits(np.array([x_min, x_max]), pad_fraction=0.12)
        ax.set_xlim(axis_min, axis_max)
        offset = (axis_max - axis_min) * 0.014
        for y_index, value in enumerate(means):
            ha = "left"
            x_text = value + offset
            if x_text > axis_max - offset:
                ha = "right"
                x_text = value - offset
            ax.text(
                x_text,
                y_index,
                f"{value:.3f}",
                va="center",
                ha=ha,
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1.5,
                },
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(MODEL_ORDER)
        ax.set_title(f"{METRIC_LABELS[metric]} mean with 95% CI")
        ax.set_xlabel("Lower is better")
        ax.grid(True, axis="x", alpha=0.25)
        ax.grid(False, axis="y")

    fig.suptitle("Repeated split summary: model means and uncertainty", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "repeated_split_mean_ci.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_differences(pairwise: pd.DataFrame) -> None:
    plot_df = pairwise[pairwise["metric"].isin(PLOT_METRICS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)

    for ax, metric in zip(axes.ravel(), PLOT_METRICS):
        metric_df = plot_df[plot_df["metric"] == metric].copy()
        metric_df["comparison_model"] = pd.Categorical(
            metric_df["comparison_model"],
            categories=COMPARISON_ORDER,
            ordered=True,
        )
        metric_df = metric_df.sort_values("comparison_model")
        y_pos = np.arange(len(metric_df))
        x_values = metric_df["mean_difference"].to_numpy()
        xerr = np.vstack(
            [
                x_values - metric_df["ci95_lower"].to_numpy(),
                metric_df["ci95_upper"].to_numpy() - x_values,
            ]
        )
        colors = [PALETTE.get(model, "#333333") for model in metric_df["comparison_model"]]
        ax.errorbar(
            x_values,
            y_pos,
            xerr=xerr,
            fmt="none",
            ecolor="#333333",
            elinewidth=1.3,
            capsize=3,
            zorder=1,
        )
        ax.scatter(x_values, y_pos, color=colors, s=60, zorder=2)
        ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_df["comparison_model"])
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Baseline - Bayesian Gibbs; positive favors Gibbs")
        ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle("Paired differences relative to Bayesian Gibbs", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "repeated_split_pairwise_differences.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_pairwise_forest(pairwise: pd.DataFrame) -> None:
    plot_df = pairwise[pairwise["metric"].isin(PLOT_METRICS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)

    for ax, metric in zip(axes.ravel(), PLOT_METRICS):
        metric_df = plot_df[plot_df["metric"] == metric].copy()
        metric_df["comparison_model"] = pd.Categorical(
            metric_df["comparison_model"],
            categories=COMPARISON_ORDER,
            ordered=True,
        )
        metric_df = metric_df.sort_values("comparison_model")
        y_pos = np.arange(len(metric_df))
        means = metric_df["mean_difference"].to_numpy()
        lower = metric_df["ci95_lower"].to_numpy()
        upper = metric_df["ci95_upper"].to_numpy()
        crosses_zero = (lower <= 0.0) & (upper >= 0.0)
        colors = np.where(
            crosses_zero,
            "#8E99A8",
            np.where(means > 0.0, "#F58518", "#4C78A8"),
        )
        max_abs = float(np.max(np.abs(np.concatenate([lower, upper, means, np.array([0.0])]))))
        axis_limit = max_abs * 1.2 if max_abs > 0 else 1.0

        ax.axvspan(-axis_limit, 0.0, color="#4C78A8", alpha=0.055, zorder=0)
        ax.axvspan(0.0, axis_limit, color="#F58518", alpha=0.065, zorder=0)
        for y_index, mean_value, lower_value, upper_value, color in zip(
            y_pos,
            means,
            lower,
            upper,
            colors,
        ):
            ax.errorbar(
                mean_value,
                y_index,
                xerr=[[mean_value - lower_value], [upper_value - mean_value]],
                fmt="none",
                ecolor=color,
                elinewidth=2.0,
                capsize=4,
                zorder=1,
            )
            ax.scatter(
                mean_value,
                y_index,
                color=color,
                s=70,
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
        ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.2, zorder=3)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_df["comparison_model"])
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Baseline - Bayesian Gibbs")
        ax.text(0.02, 1.04, "favors baseline", transform=ax.transAxes, ha="left", fontsize=9)
        ax.text(0.98, 1.04, "favors Gibbs", transform=ax.transAxes, ha="right", fontsize=9)
        for y_index, is_crossing in enumerate(crosses_zero):
            if is_crossing:
                ax.text(
                    axis_limit * 0.98,
                    y_index,
                    "crosses 0",
                    va="center",
                    ha="right",
                    fontsize=8,
                    color="#6B7280",
                )
        ax.grid(True, axis="x", alpha=0.22)
        ax.grid(False, axis="y")

    fig.suptitle("Repeated split paired differences with 95% CIs", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "repeated_split_pairwise_forest.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_gibbs_win_rate_heatmap(pairwise: pd.DataFrame) -> None:
    heatmap_df = (
        pairwise[pairwise["metric"].isin(PLOT_METRICS)]
        .pivot(index="comparison_model", columns="metric", values="win_rate_for_gibbs")
        .loc[COMPARISON_ORDER, list(PLOT_METRICS)]
    )
    heatmap_df = heatmap_df.rename(columns=METRIC_LABELS)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.heatmap(
        heatmap_df,
        vmin=0.0,
        vmax=1.0,
        center=0.5,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Gibbs win rate"},
        ax=ax,
    )
    ax.set_title("Gibbs win rate across repeated splits")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "repeated_split_gibbs_win_rate_heatmap.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated train/test split comparison for Boston Housing models."
    )
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_repeats <= 0:
        raise ValueError("--n-repeats must be positive")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")

    ensure_dirs()
    raw = run_repeated_split_comparison(
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        seed=args.seed,
    )
    summary = summarize_repeated_splits(
        raw,
        metrics=REPEATED_SPLIT_METRICS,
        model_order=MODEL_ORDER,
    )
    pairwise = pairwise_against_reference(
        raw,
        reference_model="Bayesian Gibbs",
        metrics=REPEATED_SPLIT_METRICS,
        model_order=MODEL_ORDER,
    )

    raw.to_csv(TABLES_DIR / "repeated_split_raw.csv", index=False)
    summary.to_csv(TABLES_DIR / "repeated_split_summary.csv", index=False)
    pairwise.to_csv(TABLES_DIR / "repeated_split_pairwise.csv", index=False)
    plot_metric_distributions(raw)
    plot_repeated_split_mean_ci(summary)
    plot_pairwise_differences(pairwise)
    plot_pairwise_forest(pairwise)
    plot_gibbs_win_rate_heatmap(pairwise)

    print("Repeated split comparison complete")
    print(
        summary[summary["metric"].isin(PLOT_METRICS)]
        .pivot(index="model", columns="metric", values="mean")
        .loc[MODEL_ORDER]
        .round(4)
    )
    print("\nPairwise differences relative to Bayesian Gibbs")
    print(
        pairwise[pairwise["metric"].isin(PLOT_METRICS)]
        .pivot(index="comparison_model", columns="metric", values="mean_difference")
        .loc[[model for model in MODEL_ORDER if model != "Bayesian Gibbs"]]
        .round(4)
    )


if __name__ == "__main__":
    main()
