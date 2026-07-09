"""Render equation SVG assets for GitHub-friendly documentation.

The script uses matplotlib mathtext rather than external LaTeX so it can run in
standard Python environments without a TeX installation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "docs" / "assets" / "equations"

EQUATIONS = {
    "bayesian_linear_model.svg": [
        r"$y \mid X,\beta,\sigma^2 \sim \mathcal{N}(X\beta,\sigma^2 I)$",
        r"$\beta \sim \mathcal{N}(0,V_0)$",
        r"$\sigma^2 \sim \mathrm{InvGamma}(a_0,b_0)$",
    ],
    "posterior_predictive.svg": [
        r"$p(\tilde{y}\mid \tilde{x},D)$",
        r"$= \int p(\tilde{y}\mid \tilde{x},\beta,\sigma^2)"
        r"\,p(\beta,\sigma^2\mid D)\,d\beta\,d\sigma^2$",
    ],
    "nlpd.svg": [
        r"$\mathrm{NLPD} = -\frac{1}{n}\sum_{i=1}^{n}"
        r"\log p(y_i\mid x_i,D)$",
    ],
    "paired_difference.svg": [
        r"$\Delta_m = \mathrm{metric}_m - \mathrm{metric}_{\mathrm{Gibbs}}$",
    ],
}


def render_equation(filename: str, lines: list[str]) -> None:
    height = 0.5 + 0.55 * len(lines)
    fig = plt.figure(figsize=(8.0, height), dpi=160)
    fig.patch.set_alpha(0.0)
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
    output_path = OUTPUT_DIR / filename
    fig.savefig(
        output_path,
        format="svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.05,
        metadata={"Date": None},
    )
    plt.close(fig)
    output_path.write_text(
        "\n".join(line.rstrip() for line in output_path.read_text().splitlines())
        + "\n"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = "bayesian-methods-lab"
    for filename, lines in EQUATIONS.items():
        render_equation(filename, lines)
    print(f"Rendered {len(EQUATIONS)} equation assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
