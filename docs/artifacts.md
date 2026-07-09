# Artifact Inventory

This page records generated files for reproducibility. The main README only
highlights the figures needed to understand the current Part I conclusions.

## Tables

| File | Description |
| --- | --- |
| `reports/tables/model_comparison.csv` | Fixed-split RMSE, MAE, R2, coverage, NLPD, CRPS, and interval score |
| `reports/tables/metrics.json` | Machine-readable summary of fixed-split metrics and sensitivity results |
| `reports/tables/tau_cv_summary.csv` | 5-fold cross-validation summary for Gibbs prior variance |
| `reports/tables/tau_cv_folds.csv` | Fold-level prior-variance sweep results |
| `reports/tables/posterior_coefficients.csv` | Posterior coefficient means, standard deviations, and intervals |
| `reports/tables/mcmc_diagnostics.csv` | Lightweight single-chain ESS and autocorrelation diagnostics |
| `reports/tables/repeated_split_raw.csv` | Per-split metrics for repeated train/test comparisons |
| `reports/tables/repeated_split_summary.csv` | Repeated-split metric means, standard deviations, and confidence intervals |
| `reports/tables/repeated_split_pairwise.csv` | Paired baseline-minus-Gibbs differences and Gibbs win rates |
| `reports/tables/training_size_raw.csv` | Raw small-data robustness experiment results |
| `reports/tables/training_size_summary.csv` | Aggregated small-data robustness experiment results |
| `reports/tables/bias_variance.csv` | Bootstrap bias-variance decomposition |
| `reports/tables/legacy_feature_sensitivity.csv` | Full feature set versus dropping the legacy `b` feature |
| `reports/tables/test_predictions.csv` | Held-out Bayesian predictions, intervals, and residuals |

## Figures

| File | Description |
| --- | --- |
| `reports/figures/model_comparison.png` | Fixed-split RMSE and R2 bar comparison |
| `reports/figures/fixed_split_point_metrics.png` | Same fixed-split point metric comparison saved under a descriptive name |
| `reports/figures/fixed_split_probabilistic_metrics.png` | Fixed-split NLPD, CRPS, interval score, and coverage comparison |
| `reports/figures/predictions_and_intervals.png` | Held-out predictions with posterior predictive intervals |
| `reports/figures/posterior_coefficients.png` | Posterior coefficient intervals |
| `reports/figures/mcmc_trace_diagnostics.png` | Trace plots for intercept, sigma2, and top coefficients |
| `reports/figures/repeated_split_mean_ci.png` | Repeated-split metric means with 95% confidence intervals |
| `reports/figures/repeated_split_pairwise_forest.png` | Paired metric differences relative to Bayesian Gibbs |
| `reports/figures/repeated_split_gibbs_win_rate_heatmap.png` | Gibbs win rates across metrics and baselines |
| `reports/figures/repeated_split_metric_distributions.png` | Repeated-split metric distributions |
| `reports/figures/repeated_split_pairwise_differences.png` | Earlier paired-difference visualization retained for comparison |
| `reports/figures/tau_sensitivity.png` | Prior-variance sensitivity curve |
| `reports/figures/training_size_robustness.png` | Small-data robustness comparison |
| `reports/figures/bias_variance_tradeoff.png` | Bias-variance decomposition across prior variance |
| `reports/figures/residual_diagnostics.png` | Fixed-split residual diagnostics |

## Equation Assets

| File | Description |
| --- | --- |
| `docs/assets/equations/bayesian_linear_model.svg` | Bayesian linear regression model |
| `docs/assets/equations/posterior_predictive.svg` | Posterior predictive distribution |
| `docs/assets/equations/nlpd.svg` | Negative log predictive density |
| `docs/assets/equations/paired_difference.svg` | Paired repeated-split difference |
