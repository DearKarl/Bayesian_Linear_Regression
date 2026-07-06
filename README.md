# Bayesian Linear Regression on Boston Housing

Pure-Python research benchmark for comparing ordinary linear regression with
Bayesian linear regression on the Boston Housing dataset.

This repository started as an MSc Bayesian linear regression coursework project.
It is now structured as a reproducible research codebase: reusable Python
modules, a single experiment runner, generated benchmark tables, publication
style figures, and explicit notes about the legacy Boston Housing dataset.

## Research Question

How much do Bayesian linear regression techniques help beyond ordinary least
squares when the dataset is small, correlated, and uncertainty matters?

The current experiments compare point prediction, posterior uncertainty,
prior-variance sensitivity, small-data robustness, and the bias-variance
trade-off.

## Snapshot

The latest generated results use a fixed 80/20 train/test split, 5-fold
cross-validation on the training split to choose the Bayesian prior variance,
and standardized predictors. The selected Gibbs prior variance is `tau^2 = 10`.

| Model | RMSE | MAE | R2 | 95% interval coverage |
| --- | ---: | ---: | ---: | ---: |
| Ordinary least squares | 4.940 | 3.206 | 0.667 | - |
| Bayesian Gibbs | 4.948 | 3.206 | 0.666 | 94.1% |
| BayesianRidge | 4.953 | 3.195 | 0.665 | - |
| RidgeCV | 4.956 | 3.193 | 0.665 | - |
| ARDRegression | 4.982 | 3.210 | 0.662 | - |

On the full held-out split, ordinary least squares and Bayesian Gibbs are
effectively tied on point prediction. The Bayesian model adds calibrated
posterior predictive intervals and is slightly more stable in repeated
small-data experiments: across 20%-80% training-size repeats, Bayesian Gibbs
reduces average RMSE by about 0.018, or 0.35%, relative to OLS.

The `b` feature in Boston Housing is ethically problematic. A sensitivity run
that drops it improves the Bayesian Gibbs test RMSE from 4.951 to 4.791 and
raises 95% interval coverage from 94.1% to 96.1%. This is not a causal claim,
but it is a useful reminder that benchmark features need auditing.

## Figures

![Model comparison](reports/figures/model_comparison.png)

![Posterior predictions and intervals](reports/figures/predictions_and_intervals.png)

![Posterior coefficient intervals](reports/figures/posterior_coefficients.png)

![Tau sensitivity](reports/figures/tau_sensitivity.png)

![Small-data robustness](reports/figures/training_size_robustness.png)

![Bias variance tradeoff](reports/figures/bias_variance_tradeoff.png)

## Repository Layout

```text
.
|-- BostonHousing_data.csv
|-- experiments/
|   `-- run_boston_benchmark.py
|-- src/
|   `-- bayeslinreg/
|       |-- data.py
|       |-- metrics.py
|       `-- models.py
|-- reports/
|   |-- figures/
|   `-- tables/
|-- docs/
|   |-- dataset_note.md
|   `-- original_report_summary.md
`-- tests/
    `-- test_bayeslinreg.py
```

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_boston_benchmark.py
pytest -q
```

The experiment runner regenerates every table under `reports/tables/` and every
figure under `reports/figures/`.

## Methodology

The benchmark includes five linear models:

| Family | Implementation | Purpose |
| --- | --- | --- |
| Ordinary least squares | `sklearn.linear_model.LinearRegression` | Classical point-estimate baseline |
| Ridge regression | `sklearn.linear_model.RidgeCV` | Frequentist shrinkage baseline |
| Empirical Bayes | `sklearn.linear_model.BayesianRidge` | Mainstream Python Bayesian baseline |
| Sparse empirical Bayes | `sklearn.linear_model.ARDRegression` | Automatic relevance determination baseline |
| Conjugate Bayesian Gibbs | `src/bayeslinreg/models.py` | Transparent sampler matching the original report |

The custom Gibbs sampler uses:

```text
y | X, beta, sigma^2 ~ Normal(X beta, sigma^2 I)
beta ~ Normal(0, V0)
sigma^2 ~ Inverse-Gamma(a0, b0)
```

The sampler alternates between closed-form draws of `beta | sigma^2, X, y` and
`sigma^2 | beta, X, y`. Posterior predictive intervals include both coefficient
uncertainty and residual noise.

## Result Artifacts

| File | Description |
| --- | --- |
| `reports/tables/model_comparison.csv` | Held-out RMSE, MAE, R2, and interval coverage |
| `reports/tables/tau_cv_summary.csv` | 5-fold CV prior-variance sweep |
| `reports/tables/posterior_coefficients.csv` | Posterior coefficient means and 95% intervals |
| `reports/tables/training_size_summary.csv` | Repeated small-data robustness experiment |
| `reports/tables/bias_variance.csv` | Bootstrap bias-variance decomposition |
| `reports/tables/legacy_feature_sensitivity.csv` | Full legacy features vs dropping `b` |
| `reports/tables/test_predictions.csv` | Held-out Bayesian predictions and intervals |

## Dataset Note

Boston Housing is retained because it is the dataset used in the original
analysis and remains useful as a compact regression benchmark. It should not be
treated as a modern housing-policy dataset. The original `b` variable encodes a
racial-composition transform, and scikit-learn deprecated `load_boston` for
ethical reasons. See [`docs/dataset_note.md`](docs/dataset_note.md).

## Research Direction

Good next steps for a PhD-facing Bayesian regression project:

- add posterior convergence diagnostics such as effective sample size and
  trace plots;
- compare Gibbs sampling with PyMC/NUTS or NumPyro/HMC;
- add hierarchical priors, horseshoe shrinkage, and robust likelihoods;
- evaluate beyond Boston Housing on modern tabular regression datasets;
- use PSIS-LOO, WAIC, and calibrated predictive log scores, not only RMSE.

## References

- scikit-learn, [BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
  and [ARDRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html).
- scikit-learn example,
  [Comparing Linear Bayesian Regressors](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html).
- scikit-learn legacy documentation,
  [`load_boston` deprecation note](https://scikit-learn.org/1.1/modules/generated/sklearn.datasets.load_boston.html).
- Harrison, D. and Rubinfeld, D. L. (1978).
  [Hedonic housing prices and the demand for clean air](https://doi.org/10.1016/0095-0696(78)90006-2).
