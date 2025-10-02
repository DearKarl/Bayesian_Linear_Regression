# Bayesian Linear Regression Analysis

A comprehensive analysis of Bayesian linear regression study on the Boston Housing Dataset, focusing on hyperparameters, bias–variance trade-offs, and model robustness.

## Project Overview

This repository provides a complete R implementation and empirical study of Bayesian linear regression on the Boston Housing dataset. It covers posterior inference via Gibbs sampling, uncertainty quantification through credible intervals and posterior predictive distributions, principled hyperparameter tuning of prior variance, systematic cross-validation, and a simulation-based exploration of the bias–variance trade-off.

## Background

Bayesian linear regression is suited to scenarios where uncertainty quantification is essential and prior knowledge can be meaningfully incorporated. Typical applications include housing price prediction, healthcare prognosis, and reliability analysis. By regularising coefficients through priors, the method is robust to multicollinearity and performs reliably on relatively small datasets while retaining interpretability through posterior summaries and credible intervals.

## Methodology

- **Model**  
  The Bayesian linear regression framework assumes a normal likelihood with regression coefficients β and residual variance σ².  
  Coefficients β follow a normal prior with covariance V₀ = τ²I, while σ² is assigned an inverse-gamma prior.  

- **Inference**  
  Posterior distributions are obtained via Gibbs sampling (MCMC), alternating between sampling β from its conditional multivariate normal distribution and sampling σ² from its conditional inverse-gamma distribution.  
  Posterior means and 95% credible intervals are reported after discarding burn-in iterations.  

- **Prediction and Uncertainty Quantification**  
  Posterior predictive distributions are constructed from draws of β and σ².  
  Predictive means and 95% credible intervals provide both point estimates and measures of predictive uncertainty.  

- **Evaluation**  
  Root Mean Squared Error (RMSE) is used as the primary metric.  
  Residual diagnostics, including residual-versus-predicted plots and histograms, are examined to validate model assumptions.  

- **Hyperparameter Analysis**  
  Sensitivity analysis is performed on the prior variance parameter τ², tested across values {0.01, 0.1, 1, 10, 100, 1000}.  
  RMSE trends are compared across training and validation sets to identify the optimal degree of prior regularisation.  

- **Validation Schemes**  
  - Training-set size experiments evaluate RMSE for proportions ranging from 20% to 80%.  
  - Hold-out validation splits the dataset into 80% training, 10% validation, and 10% testing with fixed random seeds.  
  - 5-fold cross-validation is applied for different τ² values, and mean RMSE across folds is reported for robust model selection.  

- **Bias–Variance Analysis**  
  For each τ², 50 bootstrap simulations are conducted on the training data.  
  Prediction error is decomposed into bias and variance, showing how small τ² leads to underfitting (high bias) and large τ² leads to instability (high variance).  

- **Robustness Considerations**  
  Experiments with reduced training data simulate limited-data conditions. Results show that Bayesian priors provide regularisation and improve the stability of parameter estimates in small-sample settings.

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
