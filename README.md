# Bayesian Linear Regression Analysis

A comprehensive analysis of Bayesian linear regression study on the Boston Housing Dataset, focusing on hyperparameters, bias–variance trade-offs, and model robustness.

## Project Overview

This repository provides a complete R implementation and empirical study of Bayesian linear regression on the Boston Housing dataset. It covers posterior inference via Gibbs sampling, uncertainty quantification through credible intervals and posterior predictive distributions, principled hyperparameter tuning of prior variance, systematic cross-validation, and a simulation-based exploration of the bias–variance trade-off.

## Background

Bayesian linear regression is suited to scenarios where uncertainty quantification is essential and prior knowledge can be meaningfully incorporated. Typical applications include housing price prediction, healthcare prognosis, and reliability analysis. By regularising coefficients through priors, the method is robust to multicollinearity and performs reliably on relatively small datasets while retaining interpretability through posterior summaries and credible intervals.

## Methodology

- **Model**  
  The Bayesian linear regression model is specified as:  
  - Likelihood: *y | X, β, σ² ~ Normal(Xβ, σ²I)*  
  - Priors: *β ~ Normal(β₀, V₀)*, *σ² ~ Inv-Gamma(a₀, b₀)*  
  - Notation: β denotes regression coefficients, V₀ = τ²I is the prior covariance matrix, and σ² is the residual variance.  

- **Inference**  
  Posterior distributions are estimated via Gibbs sampling (MCMC), alternating between:  
  - sampling β from the multivariate normal conditional posterior, and  
  - sampling σ² from the inverse-gamma conditional posterior.  
  After discarding burn-in iterations, posterior means and 95% credible intervals are used for parameter inference.  

- **Prediction and Uncertainty Quantification**  
  - Posterior predictive distributions are constructed by combining posterior draws of β and σ².  
  - For each test instance, predictive means and 95% credible intervals are reported, incorporating both parameter and residual uncertainty.  

- **Evaluation**  
  - Primary metric: RMSE = √( (1/n) · Σ(yᵢ − ŷᵢ)² )  
  - Residual diagnostics include residuals vs predicted values and residual histograms, assessing linearity, homoscedasticity, and distributional assumptions of errors.  

- **Hyperparameter Analysis**  
  - The prior covariance is defined as V₀ = τ²I. Sensitivity analysis is conducted across τ² ∈ {0.01, 0.1, 1, 10, 100, 1000}.  
  - RMSE trends for both training and validation sets are analysed to determine the optimal level of prior regularisation.  

- **Validation Schemes**  
  - **Training-set size experiments**: RMSE is computed for training proportions ranging from 20% to 80%.  
  - **Hold-out validation**: Data are partitioned into 80%/10%/10% train/validation/test splits with fixed random seeds.  
  - **Cross-validation**: 5-fold CV is applied for different τ² values, and mean CV-RMSE is reported for robust hyperparameter selection.  

- **Bias–Variance Analysis**  
  - For each τ², 50 bootstrap simulations are executed on the training data. In each resample, the model is fitted and posterior mean estimates of β are used to predict the        validation set.  
  - The mean squared error (MSE) is then decomposed as: MSE = Bias² + Variance
    where Bias² measures systematic error relative to the true value and Variance reflects instability across simulations.  

- **Robustness Considerations**  
  Experiments varying training-set size are included to simulate limited data scenarios. The results indicate that Bayesian priors introduce regularisation and improve the       stability of parameter estimates under small-sample conditions.

## Key Features  

- **Bayesian Framework**: A pipeline from data preprocessing to posterior inference, prediction, and model evaluation, ensuring reproducibility and interpretability throughout.  
- **Uncertainty-Aware Modelling**: Beyond point estimates, the model quantifies predictive uncertainty via posterior distributions and credible intervals, providing probabilistic insights into feature effects and forecasts.  
- **Hyperparameter Study**: Thorough evaluation of prior variance (τ²) and training set proportions, showing their impact on predictive performance and generalisation.  
- **Robust Model Validation**: Both hold-out validation and k-fold cross-validation are employed, reducing variance from a single split and ensuring reliable hyperparameter selection.  
- **Bias–Variance Decomposition**: Bootstrapped simulations quantify how prior variance (τ²) influences the balance between bias and variance, identifying the regime that minimises prediction error.

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
