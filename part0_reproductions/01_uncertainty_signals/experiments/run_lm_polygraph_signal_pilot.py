"""Run a small official LM-Polygraph uncertainty-signal reproduction.

This pilot is deliberately separate from the Part I Boston Housing benchmark.
It executes three estimators from lm-polygraph 0.5.0 on a transparent prompt
stress test and writes raw scores, summary metrics, and two figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph import WhiteboxModel
from lm_polygraph.defaults.register_default_stat_calculators import (
    register_default_stat_calculators,
)
from lm_polygraph.estimators import (
    MaximumSequenceProbability,
    MeanTokenEntropy,
    Perplexity,
)
from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentStatCalculator,
)
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.manager import UEManager

ROUTE_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROUTE_ROOT / "data" / "prompt_stress_test.csv"
ANNOTATIONS_PATH = ROUTE_ROOT / "data" / "manual_generation_annotations.csv"
TABLES_DIR = ROUTE_ROOT / "reports" / "tables"
FIGURES_DIR = ROUTE_ROOT / "reports" / "figures"

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
ESTIMATOR_NAMES = {
    "MeanTokenEntropy": "mean_token_entropy",
    "Perplexity": "mean_negative_log_likelihood",
    "MaximumSequenceProbability": "negative_sequence_log_probability",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def select_device() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_model(
    model_name: str,
    *,
    max_new_tokens: int,
    local_files_only: bool,
) -> WhiteboxModel:
    device, dtype = select_device()
    common = {
        "local_files_only": local_files_only,
        "torch_dtype": dtype,
        "attn_implementation": "eager",
    }

    if "Qwen2.5-VL" in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **common,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **common)

    base_model = base_model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return WhiteboxModel(
        base_model,
        tokenizer,
        model_path=model_name,
        model_type="CausalLM",
        generation_parameters=GenerationParameters(
            do_sample=False,
            max_new_tokens=max_new_tokens,
        ),
    )


def run_estimators(frame: pd.DataFrame, model: WhiteboxModel, max_new_tokens: int) -> pd.DataFrame:
    estimators = [
        MeanTokenEntropy(),
        Perplexity(),
        MaximumSequenceProbability(),
    ]
    dataset = Dataset(
        frame["prompt"].tolist(),
        [""] * len(frame),
        batch_size=1,
    )
    manager = UEManager(
        dataset,
        model,
        estimators,
        available_stat_calculators=register_default_stat_calculators("Whitebox"),
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(model),
        generation_metrics=[],
        ue_metrics=[],
        processors=[],
        ignore_exceptions=False,
        verbose=False,
        max_new_tokens=max_new_tokens,
    )
    manager()

    result = frame.copy()
    result["generation"] = manager.stats["greedy_texts"]
    result["generation"] = result["generation"].str.replace(
        r"<\|endoftext\|>$",
        "",
        regex=True,
    ).str.strip()
    for estimator in estimators:
        result[ESTIMATOR_NAMES[str(estimator)]] = manager.estimations[
            estimator.level,
            str(estimator),
        ]
    return result


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = result["stress_label"].to_numpy()
    for estimator in ESTIMATOR_NAMES.values():
        scores = result[estimator].to_numpy(dtype=float)
        row = {
                "estimator": estimator,
                "n_prompts": len(result),
                "roc_auc_for_stress_label": roc_auc_score(target, scores),
                "average_precision_for_stress_label": average_precision_score(
                    target,
                    scores,
                ),
                "answerable_mean": result.loc[
                    result["prompt_type"] == "answerable",
                    estimator,
                ].mean(),
                "ambiguous_mean": result.loc[
                    result["prompt_type"] == "ambiguous",
                    estimator,
                ].mean(),
                "unsupported_mean": result.loc[
                    result["prompt_type"] == "unsupported",
                    estimator,
                ].mean(),
            }
        if "unsafe_generation_label" in result:
            generation_target = result["unsafe_generation_label"].to_numpy()
            row["roc_auc_for_unsafe_generation"] = roc_auc_score(
                generation_target,
                scores,
            )
            row["average_precision_for_unsafe_generation"] = average_precision_score(
                generation_target,
                scores,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_scores(result: pd.DataFrame) -> None:
    metrics = list(ESTIMATOR_NAMES.values())
    normalized = MinMaxScaler().fit_transform(result[metrics])
    plot_frame = result[["prompt_id", "prompt_type"]].copy()
    plot_frame[metrics] = normalized
    long = plot_frame.melt(
        id_vars=["prompt_id", "prompt_type"],
        value_vars=metrics,
        var_name="estimator",
        value_name="normalized_uncertainty",
    )

    colors = {
        "answerable": "#4C78A8",
        "ambiguous": "#F2CF5B",
        "unsupported": "#E45756",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, estimator in zip(axes, metrics):
        subset = long[long["estimator"] == estimator]
        for index, prompt_type in enumerate(colors):
            values = subset.loc[
                subset["prompt_type"] == prompt_type,
                "normalized_uncertainty",
            ].to_numpy()
            jitter = np.linspace(-0.08, 0.08, len(values))
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                color=colors[prompt_type],
                s=58,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.6,
            )
            ax.hlines(
                values.mean(),
                index - 0.24,
                index + 0.24,
                color="#222222",
                linewidth=2,
            )
        ax.set_xticks(range(3), ["Answerable", "Ambiguous", "Unsupported"], rotation=18)
        ax.set_title(estimator.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Min--max normalized uncertainty")
    fig.suptitle("LM-Polygraph signals on the prompt stress test", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "uncertainty_by_prompt_type.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    order = result.sort_values("mean_token_entropy")["prompt_id"].tolist()
    x = np.arange(len(order))
    width = 0.25
    for offset, metric in enumerate(metrics):
        values = plot_frame.set_index("prompt_id").loc[order, metric].to_numpy()
        ax.bar(
            x + (offset - 1) * width,
            values,
            width=width,
            label=metric.replace("_", " "),
        )
    ax.set_xticks(x, order, rotation=55, ha="right")
    ax.set_ylabel("Min--max normalized uncertainty")
    ax.set_title("Signal disagreement across individual prompts")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "signal_disagreement.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if "unsafe_generation_label" in result:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        n = len(result)
        for metric in metrics:
            ordered = result.sort_values(metric)
            coverage = np.arange(1, n + 1) / n
            selective_risk = (
                ordered["unsafe_generation_label"].expanding().mean().to_numpy()
            )
            ax.plot(
                coverage,
                selective_risk,
                marker="o",
                markersize=3.5,
                linewidth=2,
                label=metric.replace("_", " "),
            )
        overall_risk = result["unsafe_generation_label"].mean()
        ax.axhline(
            overall_risk,
            color="#222222",
            linestyle="--",
            linewidth=1.2,
            label=f"overall unsafe rate = {overall_risk:.2f}",
        )
        ax.set_xlabel("Coverage retained after rejecting high-uncertainty prompts")
        ax.set_ylabel("Observed unsafe-generation rate")
        ax.set_title("Pilot risk--coverage curves")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.02, max(0.55, overall_risk + 0.1))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8.5)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "risk_coverage_pilot.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    prompts = pd.read_csv(DATA_PATH)
    if args.limit is not None:
        prompts = prompts.head(args.limit).copy()

    model = load_model(
        args.model,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
    )
    result = run_estimators(prompts, model, args.max_new_tokens)
    if ANNOTATIONS_PATH.exists() and args.limit is None:
        annotations = pd.read_csv(ANNOTATIONS_PATH)
        result = result.merge(
            annotations,
            on="prompt_id",
            how="left",
            validate="one_to_one",
        )
    summary = summarize(result)
    result.to_csv(TABLES_DIR / "signal_scores.csv", index=False)
    summary.to_csv(TABLES_DIR / "signal_summary.csv", index=False)
    plot_scores(result)

    metadata = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "n_prompts": len(result),
        "device": str(model.device()),
        "torch_version": torch.__version__,
    }
    (TABLES_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
