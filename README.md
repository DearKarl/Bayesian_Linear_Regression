# Bayesian Linear Regression Analysis

Comprehensive Study of Principles, Hyperparameter Tuning, and Model Performance

## Project Overview

This repository provides a complete R implementation and empirical study of Bayesian linear regression on the Boston Housing dataset. It covers posterior inference via Gibbs sampling, uncertainty quantification through credible intervals and posterior predictive distributions, principled hyperparameter tuning of prior variance, systematic cross-validation, and a simulation-based exploration of the bias–variance trade-off.

## Background

Bayesian linear regression is suited to scenarios where uncertainty quantification is essential and prior knowledge can be meaningfully incorporated. Typical applications include housing price prediction, healthcare prognosis, and reliability analysis. By regularising coefficients through priors, the method is robust to multicollinearity and performs reliably on relatively small datasets while retaining interpretability through posterior summaries and credible intervals.

## Key Features

### Data Loading and Preprocessing

- **Dataset**: Boston Housing Dataset
- **Data Inspection and Cleaning**:
  - Missing values are handled using mean imputation.
  - Dataset is divided into training (80%), validation (10%), and test (10%) sets.

### Bayesian Linear Regression Implementation

- **Gibbs Sampling**:
  - Bayesian inference using Gibbs sampling (MCMC) for posterior distributions
  - Sampling regression coefficients (β) and residual variance (σ²)
  - Calculation of posterior means, covariance matrices, and credible intervals

### Prediction and Uncertainty Quantification

- **Posterior Predictive Distributions**:
  - Prediction function leveraging posterior samples of coefficients and variance
  - Calculation of predictive means and 95% credible intervals

- **Visualization**:
  - Regression coefficient credible intervals
  - Comparison between predicted and actual housing prices
  - Residual analysis (residuals vs predicted values, residual distribution histograms)

### Hyperparameter Exploration

- **Training Set Size**:
  - RMSE analysis for varying proportions of training data (20%–80%)
  - Visualization of optimal training proportion based on validation RMSE

- **Prior Variance (τ²) Impact**:
  - RMSE analysis across varying prior variances (τ²: 0.01, 0.1, 1, 10, 100, 1000)
  - Optimal hyperparameter determination based on validation and test RMSE

### Cross-Validation Analysis

- **K-Fold Cross-Validation**:
  - 5-fold cross-validation to systematically evaluate model robustness
  - RMSE analysis across varying prior variances
  - Visualization of cross-validation performance relative to hyperparameter settings

### Bias-Variance Trade-Off

- **Simulation-Based Analysis**:
  - Bootstrapped simulations to calculate bias, variance, and MSE for different τ² values
  - Comprehensive visualization of bias-variance trade-off

## Data Sources

- [Boston Housing Dataset (Kaggle)](https://www.kaggle.com/datasets/manimala/boston-house-prices)
- [Boston Housing (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/housing)
- Original compilation: Harrison, D., & Rubinfeld, D. L. (1978), based on U.S. Census data

Data are used for research and educational purposes; please refer to the respective dataset licences and terms of use.

## References

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Gelfand, A. E., & Smith, A. F. M. (1990). Sampling-Based Approaches to Calculating Marginal Densities. *Journal of the American Statistical Association*, 85(410), 398–409.
- Andrieu, C., de Freitas, N., Doucet, A., & Jordan, M. I. (2003). An Introduction to MCMC for Machine Learning. *Machine Learning*, 50(1–2), 5–43.
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593–1623.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian Model Evaluation Using Leave-One-Out Cross-Validation and WAIC. *Statistics and Computing*, 27(5), 1413–1432.
- Arlot, S., & Celisse, A. (2010). A Survey of Cross-Validation Procedures for Model Selection. *Statistics Surveys*, 4, 40–79.
- Domingos, P. (2000). A Unified Bias–Variance Decomposition for Zero-One and Squared Loss. In *AAAI*, 564–569.

---

> Repository maintained by [DearKarl](https://github.com/DearKarl). Contributions and feedback welcome. 
