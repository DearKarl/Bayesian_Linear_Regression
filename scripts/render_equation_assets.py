"""Render equation assets for GitHub-friendly documentation.

The script uses matplotlib mathtext rather than external LaTeX so it can run in
standard Python environments without a TeX installation. Each equation is saved
as both SVG and PNG; documentation should prefer PNG when SVG rendering is
unreliable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "docs" / "assets" / "equations"

EQUATIONS = {
    "bayesian_linear_model": [
        r"$y \mid X,\beta,\sigma^2 \sim \mathcal{N}(X\beta,\sigma^2 I)$",
        r"$\beta \sim \mathcal{N}(0,V_0)$",
        r"$\sigma^2 \sim \mathrm{InvGamma}(a_0,b_0)$",
    ],
    "posterior_predictive": [
        r"$p(\tilde{y}\mid \tilde{x},D)$",
        r"$= \int p(\tilde{y}\mid \tilde{x},\beta,\sigma^2)"
        r"\,p(\beta,\sigma^2\mid D)\,d\beta\,d\sigma^2$",
    ],
    "nlpd": [
        r"$\mathrm{NLPD} = -\frac{1}{n}\sum_{i=1}^{n}"
        r"\log p(y_i\mid x_i,D)$",
    ],
    "paired_difference": [
        r"$\Delta_m = \mathrm{metric}_m - \mathrm{metric}_{\mathrm{Gibbs}}$",
    ],
    "hallucination_risk_score": [
        r"$P(H=1\mid \mathrm{prompt},\mathrm{answer},"
        r"\mathrm{evidence},\mathrm{uncertainty\ features})$",
    ],
    "bayesian_logistic_risk_model": [
        r"$H_i\sim\mathrm{Bernoulli}(r_i)$",
        r"$\mathrm{logit}(r_i)=\alpha+z_i^\top w$",
        r"$\alpha\sim\mathcal{N}(0,\sigma_\alpha^2),\quad"
        r"w_j\sim\mathcal{N}(0,\tau^2)$",
    ],
    "binary_nll": [
        r"$\mathrm{NLL}=-\frac{1}{n}\sum_{i=1}^{n}"
        r"\left[y_i\log r_i+(1-y_i)\log(1-r_i)\right]$",
    ],
    "brier_score": [
        r"$\mathrm{Brier}=\frac{1}{n}\sum_{i=1}^{n}(r_i-y_i)^2$",
    ],
}


def _strip_trailing_whitespace(path: Path) -> None:
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def render_equation(name: str, lines: list[str]) -> None:
    height = 0.55 + 0.62 * len(lines)
    fig = plt.figure(figsize=(8.8, height), dpi=180)
    fig.patch.set_facecolor("white")

    for index, line in enumerate(lines):
        y_position = 1.0 - (index + 1) / (len(lines) + 1)
        fig.text(
            0.5,
            y_position,
            line,
            ha="center",
            va="center",
            fontsize=18,
            color="#111827",
        )

    for extension in ("svg", "png"):
        output_path = OUTPUT_DIR / f"{name}.{extension}"
        fig.savefig(
            output_path,
            format=extension,
            facecolor=fig.get_facecolor(),
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.08,
            metadata={"Date": None} if extension == "svg" else {"Software": "matplotlib"},
        )
        if extension == "svg":
            _strip_trailing_whitespace(output_path)

    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = "bayesian-methods-lab"
    for name, lines in EQUATIONS.items():
        render_equation(name, lines)
    print(f"Rendered {len(EQUATIONS)} equations as SVG and PNG to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
