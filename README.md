# Bayesian Methods Lab

Bayesian Methods Lab is a research project dedicated to Bayesian methods for reliable multimodal large language models. We study how Bayesian posterior inference can be used to quantify hallucination, calibrate model confidence, and support reliable decision making under uncertainty.

## Research Roadmap

| Part | Research Theme | Current Stage | Documentation |
| :--- | :--- | :--- | :--- |
| Part I | Bayesian Regression Foundations | Completed | [Report](docs/part1_bayesian_regression_foundations.md) |
| Part II | Bayesian Hallucination Risk Modeling | In Progress | [Research Plan](docs/part2_bayesian_hallucination_risk_modeling.md) |
| Part III | Multimodal Hallucination Uncertainty | Planned | Coming Soon |
| Part IV | Bayesian Abstention and Decision Rules | Planned | Coming Soon |

## Current Status

Part I is complete as a controlled foundation study. It shows that Bayesian
Gibbs regression is competitive on point prediction and useful for posterior
uncertainty and predictive density, but it does not provide stable evidence of
general RMSE improvement over ordinary least squares.

The next stage is Part II: Bayesian Hallucination Risk Modeling, where the lab
will study calibrated posterior risk estimates for language-model
hallucination.

## Navigate

| Looking for... | Go to |
| --- | --- |
| Completed Bayesian regression foundation study | [Part I report](docs/part1_bayesian_regression_foundations.md) |
| Next hallucination-risk research plan | [Part II scaffold](docs/part2_bayesian_hallucination_risk_modeling.md) |
| Planned multimodal hallucination direction | [Part III scaffold](docs/part3_multimodal_hallucination_uncertainty.md) |
| Planned Bayesian decision-rule direction | [Part IV scaffold](docs/part4_bayesian_abstention_decision_rules.md) |
| Lab-wide research questions | [Research questions](docs/research_questions.md) |
| Development roadmap | [Roadmap](docs/roadmap.md) |
| Boston Housing ethics and benchmark caveats | [Dataset note](docs/dataset_note.md) |

Reproduction commands are listed inside each Part report. Use `--n-repeats`
with `experiments/run_repeated_split_comparison.py` for a faster repeated-split
smoke run.

## References

References are organized by research part. See the
[Part I report](docs/part1_bayesian_regression_foundations.md) for the current
regression-foundation references; hallucination-risk references are collected
in the relevant later Part documents as those studies mature.
