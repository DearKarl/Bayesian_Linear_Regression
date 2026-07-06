# Original Report Summary

The uploaded MSc report framed Bayesian linear regression as a way to move
beyond point estimates toward posterior uncertainty, credible intervals, and
more robust behavior under limited data.

The original analysis included:

- a Gaussian likelihood for linear regression;
- a Gaussian prior for regression coefficients;
- an inverse-gamma prior for residual variance;
- Gibbs sampling for posterior inference;
- posterior predictive intervals;
- sensitivity analysis over `tau^2` values;
- 5-fold cross-validation;
- training-size experiments;
- a bootstrap bias-variance decomposition.

This Python version keeps those ideas but turns them into a research-style
benchmark:

- RMarkdown code was replaced by a reusable Python package under `src/`;
- experiments are regenerated from `experiments/run_boston_benchmark.py`;
- OLS, RidgeCV, BayesianRidge, ARDRegression, and a custom Gibbs sampler are
  compared in one table;
- all plots and CSV artifacts are saved under `reports/`;
- the legacy Boston Housing `b` variable is audited through a drop-feature
  sensitivity experiment;
- tests cover data loading and the sampler interface.
