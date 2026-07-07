# Agent Guidelines

This repository is evolving into **Bayesian Methods Lab**, an exploratory
research codebase for Bayesian modeling, posterior inference, uncertainty
quantification, and robust prediction.

## Scope Rules

- Treat the current Boston Housing benchmark as **Part I: Bayesian Regression
  Foundations**.
- Preserve the existing benchmark, generated result tables, figures, and numeric
  results unless a task explicitly asks to rerun or revise the experiments.
- Do not claim that Bayesian methods improve RMSE over OLS unless
  repeated-split or otherwise statistically meaningful evidence supports it.
- Keep the current custom Gibbs sampler unless a task explicitly asks for a new
  inference implementation.
- Do not rename the Python package `bayeslinreg` without explicit instruction.
- Keep generated tables in `reports/tables/` and generated figures in
  `reports/figures/`.
- Prefer documentation-first, PR-sized changes over broad refactors.

## Development Rules

- Use the existing package structure under `src/bayeslinreg/`.
- Keep reproduce instructions in `README.md` working.
- Run `pytest -q` before publishing changes.
- Do not regenerate benchmark outputs unless necessary for the requested change.
- If benchmark outputs are regenerated, report the command used and summarize
  any numeric changes.
- Keep README claims aligned with the saved tables under `reports/tables/`.

## Research Framing

- Make a clear distinction between point-prediction accuracy and probabilistic
  value, such as posterior uncertainty, interval coverage, calibration, and
  robustness.
- Prefer careful empirical language: "comparable RMSE with additional
  uncertainty quantification" is better than unsupported claims of improvement.
- When adding future methods, document the research question, expected
  comparison, and validation plan before adding substantial code.
