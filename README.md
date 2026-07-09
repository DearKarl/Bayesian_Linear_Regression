# Bayesian Methods Lab

Bayesian Methods Lab studies how Bayesian reasoning can make AI systems more
uncertainty-aware, calibrated, and decision-conscious. The lab begins with
Bayesian regression foundations and then moves toward Bayesian
hallucination-risk modeling for language and multimodal systems.

## Research Parts

| Part | Theme | Status | Entry Point |
| --- | --- | --- | --- |
| I | Bayesian Regression Foundations | Completed foundation study | [Part I report](docs/part1_bayesian_regression_foundations.md) |
| II | Bayesian Hallucination Risk Modeling | Next | [Part II scaffold](docs/part2_bayesian_hallucination_risk_modeling.md) |
| III | Multimodal Hallucination Uncertainty | Planned | Future text-image/audio-video uncertainty experiments |
| IV | Bayesian Abstention and Decision Rules | Planned | Future risk-aware abstention and escalation experiments |

## Current Foundation Study

Part I is a controlled foundation study before the lab moves to
hallucination-risk modeling. It keeps the model class simple so that posterior
prediction, probabilistic scoring, and uncertainty diagnostics can be inspected
cleanly.

| Component | Part I Summary |
| --- | --- |
| Dataset | Legacy Boston Housing, with an explicit ethical note |
| Models | Ordinary least squares, RidgeCV, BayesianRidge, ARDRegression, custom Bayesian Gibbs |
| Bayesian focus | Posterior coefficients, residual variance, and posterior predictive samples |
| Evaluation | RMSE, MAE, R2, coverage, NLPD, CRPS, and 95% interval score |
| Stability check | 30 repeated train/test splits with paired baseline-minus-Gibbs differences |
| Diagnostics | Lightweight single-chain ESS, autocorrelation, and trace plots |

Detailed model equations, experiment settings, and figures are kept in the
Part I report rather than displayed on the README.

## Current Findings

| Finding | Evidence So Far | Interpretation |
| --- | --- | --- |
| No stable RMSE advantage | Repeated-split RMSE confidence intervals cross zero against every baseline | The current benchmark does not support a point-prediction improvement claim |
| Repeated-split NLPD favors Gibbs | Gibbs has better paired NLPD differences against the current baselines | Posterior predictive density is the strongest Part I signal |
| CRPS favors RidgeCV and BayesianRidge | Repeated-split CRPS differences are negative for these baselines | Gibbs does not dominate all probabilistic scores |
| Interval score is mixed | Gibbs is favored against OLS and RidgeCV, but not clearly against BayesianRidge or ARDRegression | Interval quality depends on the baseline and scoring rule |
| Gibbs remains useful | It provides posterior uncertainty, predictive intervals, and inspectable samples | Bayesian Gibbs is a useful uncertainty-aware research baseline |

The safest Part I conclusion is that Bayesian Gibbs is valuable for posterior
uncertainty and predictive density, not that it broadly outperforms ordinary
least squares.

## How To Navigate The Lab

| Need | Go To |
| --- | --- |
| Full Part I regression study | [docs/part1_bayesian_regression_foundations.md](docs/part1_bayesian_regression_foundations.md) |
| Part II hallucination-risk scaffold | [docs/part2_bayesian_hallucination_risk_modeling.md](docs/part2_bayesian_hallucination_risk_modeling.md) |
| Research questions by part | [docs/research_questions.md](docs/research_questions.md) |
| PR-sized roadmap | [docs/roadmap.md](docs/roadmap.md) |
| Dataset ethics and benchmark caveats | [docs/dataset_note.md](docs/dataset_note.md) |

## Reproduce Part I

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/render_equation_assets.py
python experiments/run_boston_benchmark.py
python experiments/run_repeated_split_comparison.py
pytest -q
```

Use `--n-repeats` with `experiments/run_repeated_split_comparison.py` for a
faster repeated-split smoke run.

## Contributor Notes

- Keep experiments reproducible with deterministic seeds, saved outputs, and
  explicit validation commands.
- Avoid strong claims without repeated-split, external-benchmark, or
  task-specific calibration evidence.
- Keep generated outputs under `reports/tables/` and `reports/figures/`.
- Keep the package name `bayeslinreg` until a dedicated rename PR.

## References

Detailed Part I references are listed in the
[Part I report](docs/part1_bayesian_regression_foundations.md). Future
hallucination-risk references should be added to the relevant Part II document
rather than expanded into a long README bibliography.
