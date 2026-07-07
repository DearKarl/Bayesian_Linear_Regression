# Research Questions

Bayesian Methods Lab is an exploratory research repository for Bayesian
modeling, posterior inference, uncertainty quantification, and robust
prediction. The current Boston Housing benchmark is Part I: Bayesian Regression
Foundations.

## Bayesian Regression Foundations

- How does ordinary least squares compare with Bayesian linear regression when
  the model class is intentionally simple?
- How does prior variance affect shrinkage, bias, variance, and held-out
  prediction?
- When do Bayesian posterior summaries add value even when point-prediction
  metrics are comparable to OLS?

## Posterior Inference and MCMC Diagnostics

- Are posterior samples mixing well enough to support the reported summaries?
- How should effective sample size, trace plots, autocorrelation, and
  convergence diagnostics be integrated into the benchmark?
- How do Gibbs sampling and gradient-based samplers such as NUTS compare on the
  same regression model?

## Uncertainty Quantification

- Are posterior predictive intervals calibrated under repeated train/test
  splits?
- How should RMSE be complemented by log predictive density, interval coverage,
  interval width, CRPS, or other probabilistic scoring rules?
- When does uncertainty quantification change the interpretation of model
  performance?

## Robust Bayesian Regression

- How sensitive is Gaussian linear regression to outliers, heavy tails, and
  capped target values?
- Does a Student-t likelihood produce more stable posterior predictions than a
  Gaussian likelihood?
- Can robust Bayesian regression improve calibration without overstating point
  prediction gains?

## Sparse Priors

- Which predictors remain influential under sparse Bayesian priors?
- How do ARD, Laplace, spike-and-slab, and horseshoe priors compare for feature
  relevance and predictive uncertainty?
- Can sparse priors improve interpretability while preserving predictive
  performance?

## Hierarchical Bayesian Models

- What structure can be learned when observations are grouped by domain,
  geography, time, or experimental condition?
- How do partial pooling and hierarchical priors change estimates in small-data
  settings?
- How should hierarchical models be evaluated against pooled and unpooled
  baselines?

## Gaussian Processes and BART

- When do nonparametric Bayesian methods outperform linear Bayesian baselines?
- How should Gaussian processes be used for calibrated nonlinear regression?
- Can Bayesian additive regression trees provide stronger tabular prediction
  while preserving uncertainty estimates?

## Engineering-Mathematics Applications

- How can Bayesian regression ideas transfer to inverse problems and Bayesian
  calibration?
- What toy inverse problems can demonstrate prior choice, likelihood design, and
  posterior uncertainty clearly?
- How can this repository become a foundation for engineering-mathematics
  research, including model calibration, parameter inference, and uncertainty
  propagation?
