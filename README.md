# Bayesian Methods Lab

Bayesian Methods Lab is a small research-oriented Python repository for
exploring Bayesian modeling, posterior inference, uncertainty quantification,
and robust prediction. The lab starts with a transparent Bayesian linear
regression benchmark and is designed to grow into broader Bayesian methods
experiments.

## Research Map

| Part | Theme | Status | Core Question |
| --- | --- | --- | --- |
| I | Bayesian Regression Foundations | Current | What does posterior inference add beyond OLS on a compact tabular regression benchmark? |
| II | Probabilistic Inference Engines | Planned | How do custom Gibbs, PyMC/NUTS, and related samplers compare in accuracy, diagnostics, and workflow? |
| III | Robust And Sparse Bayesian Regression | Planned | When do robust likelihoods and sparse priors improve predictive reliability? |
| IV | Hierarchical And Nonparametric Models | Planned | How do hierarchical models, Gaussian processes, and BART represent structure and uncertainty? |
| V | Engineering-Mathematics Applications | Planned | How can Bayesian methods support inverse problems, calibration, and uncertainty-aware simulation? |

## Part I: Bayesian Regression Foundations

Part I compares ordinary least squares, RidgeCV, BayesianRidge,
ARDRegression, and a custom conjugate Gibbs sampler on the legacy Boston
Housing regression benchmark. The custom model is intentionally simple:

![Bayesian linear model equation](docs/assets/equations/bayesian_linear_model.svg)

The Bayesian value proposition is not just a fitted mean. It is the posterior
predictive distribution:

![Posterior predictive equation](docs/assets/equations/posterior_predictive.svg)

The current repeated-split evidence supports a careful interpretation:

- there is no stable RMSE advantage for Bayesian Gibbs;
- repeated-split NLPD favors Bayesian Gibbs;
- CRPS favors RidgeCV and BayesianRidge;
- interval score evidence is mixed;
- Bayesian Gibbs is most useful here for posterior uncertainty and predictive
  density, not broad dominance over every baseline.

Full Part I details are in
[`docs/part1_bayesian_regression_foundations.md`](docs/part1_bayesian_regression_foundations.md).

## Key Results

### Posterior Predictive Uncertainty

The posterior predictive interval plot shows what the Gibbs sampler adds beyond
a point estimate: held-out predictions are represented with uncertainty bands
that include posterior coefficient uncertainty and residual noise.

![Posterior predictions and intervals](reports/figures/predictions_and_intervals.png)

### Repeated-Split Stability

Fixed train/test splits can overstate tiny model differences. The forest plot
uses paired repeated-split differences, computed as baseline metric minus Gibbs
metric. For lower-is-better metrics, positive values favor Gibbs.

![Repeated split paired differences](reports/figures/repeated_split_pairwise_forest.png)

The main result is cautious: RMSE intervals cross zero for every baseline, NLPD
favors Gibbs, CRPS favors RidgeCV and BayesianRidge, and interval score is
mixed.

### Gibbs Win Rates

The win-rate heatmap summarizes how often Gibbs beats each baseline across
repeated splits. It reinforces the same message: Gibbs wins often on NLPD, but
not consistently on RMSE or CRPS.

![Gibbs win rate heatmap](reports/figures/repeated_split_gibbs_win_rate_heatmap.png)

## Reproduce

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
faster smoke run.

## Project Layout

```text
.
|-- experiments/
|   |-- run_boston_benchmark.py
|   `-- run_repeated_split_comparison.py
|-- src/bayeslinreg/
|   |-- data.py
|   |-- diagnostics.py
|   |-- metrics.py
|   |-- models.py
|   `-- repeated_split.py
|-- docs/
|   |-- assets/equations/
|   |-- artifacts.md
|   |-- dataset_note.md
|   |-- part1_bayesian_regression_foundations.md
|   |-- research_questions.md
|   `-- roadmap.md
|-- reports/
|   |-- figures/
|   `-- tables/
`-- tests/
```

## Documentation

- [Part I: Bayesian Regression Foundations](docs/part1_bayesian_regression_foundations.md)
- [Research Questions](docs/research_questions.md)
- [Roadmap](docs/roadmap.md)
- [Dataset Note](docs/dataset_note.md)
- [Artifact Inventory](docs/artifacts.md)

## Near-Term Roadmap

- compare the custom Gibbs sampler with PyMC/NUTS or NumPyro/HMC;
- add robust Student-t regression and sparse priors;
- add hierarchical regression and modern tabular benchmarks;
- evaluate Bayesian inverse-problem and calibration toy examples.

## References

- scikit-learn,
  [BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
  and
  [ARDRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html).
- scikit-learn example,
  [Comparing Linear Bayesian Regressors](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html).
- Harrison, D. and Rubinfeld, D. L. (1978).
  [Hedonic housing prices and the demand for clean air](https://doi.org/10.1016/0095-0696(78)90006-2).
