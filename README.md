# Bayesian Linear Regression Analysis

A comprehensive analysis of Bayesian linear regression study on the Boston Housing Dataset, focusing on hyperparameters, bias–variance trade-offs, and model robustness.

## Project Overview

This repository provides a complete R implementation and empirical study of Bayesian linear regression on the Boston Housing dataset. It covers posterior inference via Gibbs sampling, uncertainty quantification through credible intervals and posterior predictive distributions, principled hyperparameter tuning of prior variance, systematic cross-validation, and a simulation-based exploration of the bias–variance trade-off.

## Background

Bayesian linear regression is suited to scenarios where uncertainty quantification is essential and prior knowledge can be meaningfully incorporated. Typical applications include housing price prediction, healthcare prognosis, and reliability analysis. By regularising coefficients through priors, the method is robust to multicollinearity and performs reliably on relatively small datasets while retaining interpretability through posterior summaries and credible intervals.

## Methodology  

- **Model**:  
  We consider the Bayesian linear regression model:  
  \[
  y \mid X, \beta, \sigma^2 \sim \mathcal{N}(X\beta, \sigma^2 I),
  \]
  with priors  
  \[
  \beta \sim \mathcal{N}(\beta_0, V_0), \quad \sigma^2 \sim \text{Inv-Gamma}(a_0, b_0).
  \]  
  Here, \( \beta \) denotes the regression coefficients, \( V_0 = \tau^2 I \) the prior covariance, and \( \sigma^2 \) the residual variance.  

- **Inference**:  
  Posterior distributions are estimated via Gibbs sampling (MCMC). The algorithm iteratively samples from conditional distributions of \( \beta \) and \( \sigma^2 \), generating posterior draws. Posterior means and 95% credible intervals are used for statistical inference and feature interpretation.  

- **Prediction and Uncertainty Quantification**:  
  Posterior predictive distributions are computed using sampled parameters. Predictive means and 95% credible intervals are reported, providing interval forecasts that account for both model and parameter uncertainty.  

- **Evaluation**:  
  - Root Mean Squared Error (RMSE) is used as the primary evaluation metric:  
    \[
    \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}.
    \]  
  - Residual diagnostics (residuals vs. fitted plots, residual histograms) are applied to validate assumptions of linearity, homoscedasticity, and normality of errors.  

- **Hyperparameters**:  
  The prior covariance parameter is defined as \( V_0 = \tau^2 I \).  
  Sensitivity analysis is conducted by exploring a range of prior variances:  
  \[
  \tau^2 \in \{0.01, 0.1, 1, 10, 100, 1000\}.
  \]  
  This allows investigation of how regularisation strength influences predictive performance.  

- **Validation Schemes**:  
  - **Hold-out validation**: Data are split into training (80%), validation (10%), and test (10%) sets with fixed random seeds for reproducibility.  
  - **Cross-validation**: 5-fold cross-validation is applied across different prior variances (\( \tau^2 \)) to reduce sensitivity to specific data partitions and ensure robust model selection.  

- **Bias–Variance Analysis**:  
  Bootstrap simulations (50 replications per prior setting) are conducted to decompose mean squared error (MSE) into bias², variance, and irreducible noise:  
  \[
  \text{MSE} = \text{Bias}^2 + \text{Variance} + \sigma^2_{\text{noise}}.
  \]  
  This analysis reveals how different levels of prior variance (\( \tau^2 \)) balance model flexibility (variance) against underfitting (bias).  

- **Robustness Considerations**:  
  Further experiments assess performance under small training sets and imbalanced data distributions. Priors provide regularisation and mitigate instability, while credible intervals reflect increased uncertainty under limited or skewed data.  

## Key Features  

- **Bayesian Framework**: A pipeline from data preprocessing to posterior inference, prediction, and model evaluation, ensuring reproducibility and interpretability throughout.  
- **Uncertainty-Aware Modelling**: Beyond point estimates, the model quantifies predictive uncertainty via posterior distributions and credible intervals, providing probabilistic insights into feature effects and forecasts.  
- **Hyperparameter Study**: Thorough evaluation of prior variance (τ²) and training set proportions, showing their impact on predictive performance and generalisation.  
- **Robust Model Validation**: Both hold-out validation and k-fold cross-validation are employed, reducing variance from a single split and ensuring reliable hyperparameter selection.  
- **Bias–Variance Decomposition**: Simulation-based analysis explicitly disentangles bias, variance, and irreducible error, highlighting the trade-off that governs predictive performance in Bayesian regression.  

## Data Sources

- [Boston House Prices (Kaggle Dataset by vikrishnan)](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices)  

Data are used for research and educational purposes; please refer to Kaggle’s license and terms of use.

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
