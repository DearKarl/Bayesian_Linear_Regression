# Bayesian Linear Regression

This repository contains a comprehensive R implementation and analysis of Bayesian linear regression applied to the Boston Housing dataset. The analysis demonstrates the Bayesian approach, including posterior inference, hyperparameter tuning, model performance evaluation, and a detailed exploration of bias-variance trade-offs.

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


© 2024 Karl Meng. All rights reserved.
