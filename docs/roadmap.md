# Roadmap

This roadmap keeps future work PR-sized and research-question driven. The goal
is to grow Bayesian Methods Lab without losing reproducibility or overstating
results.

## 1. Documentation-First Project Reframing

- Reframe the repository as Bayesian Methods Lab.
- Keep the Boston Housing benchmark as Part I: Bayesian Regression Foundations.
- Add repository rules, research questions, and this roadmap.
- Do not change current numeric results.

## 2. Probabilistic Scoring Metrics

- Add log predictive density or negative log predictive density.
- Add interval coverage and interval width summaries across repeated splits.
- Consider CRPS if the predictive distribution interface supports it cleanly.
- Keep RMSE and MAE as point-prediction baselines, not the only metrics.

## 3. MCMC Diagnostics

- Add trace plots for selected coefficients and residual variance.
- Report effective sample size and autocorrelation diagnostics.
- Add lightweight convergence checks for the custom Gibbs sampler.
- Save diagnostic figures under `reports/figures/`.

## 4. Repeated-Split Statistical Comparison

- Run repeated train/test splits for OLS, RidgeCV, BayesianRidge,
  ARDRegression, and Bayesian Gibbs.
- Report means, standard deviations, confidence intervals, and paired
  differences.
- Only claim predictive improvement if repeated-split evidence supports it.

## 5. PyMC/NUTS Implementation

- Reimplement the Part I Bayesian linear regression model in PyMC.
- Compare posterior summaries from Gibbs sampling and NUTS.
- Use the same preprocessing and train/test splits for fair comparison.

## 6. Student-t Robust Regression

- Add a Student-t likelihood for robust Bayesian regression.
- Evaluate behavior under outliers and target capping.
- Compare point metrics, interval calibration, and residual diagnostics.

## 7. Horseshoe Sparse Regression

- Add a horseshoe-prior regression experiment.
- Compare sparse posterior shrinkage with ARDRegression and BayesianRidge.
- Report feature relevance with uncertainty intervals.

## 8. Hierarchical Regression

- Introduce a dataset or synthetic setting with meaningful groups.
- Compare pooled, unpooled, and partially pooled regression.
- Emphasize small-sample behavior and uncertainty propagation.

## 9. Gaussian Process or BART

- Add one nonlinear Bayesian regression method.
- Compare against linear Bayesian baselines on the same metrics.
- Focus on calibrated nonlinear prediction rather than raw RMSE alone.

## 10. Bayesian Inverse Problem Toy Example

- Add a small engineering-mathematics inverse problem.
- Demonstrate prior design, likelihood construction, posterior inference, and
  uncertainty propagation.
- Keep the example compact enough to reproduce from a single experiment script.
