# Bayesian Methods Lab

Bayesian Methods Lab explores Bayesian approaches to uncertainty
quantification, calibration, and risk-aware decision making in reliable AI. The
lab uses Bayesian regression as a controlled foundation and develops toward
hallucination-risk estimation, multimodal uncertainty modeling, and Bayesian
decision rules for language and vision-language systems.

## Research Parts

| Part | Focus | Status |
| --- | --- | --- |
| [Part I](docs/part1_bayesian_regression_foundations.md) | Bayesian Regression Foundations | Completed foundation study |
| [Part II](docs/part2_bayesian_hallucination_risk_modeling.md) | Bayesian Hallucination Risk Modeling | Next |
| [Part III](docs/part3_multimodal_hallucination_uncertainty.md) | Multimodal Hallucination Uncertainty | Planned |
| [Part IV](docs/part4_bayesian_abstention_decision_rules.md) | Bayesian Abstention and Decision Rules | Planned |

Part I establishes the workflow for posterior prediction, probabilistic scoring,
and uncertainty diagnostics. Part II transfers these ideas to
hallucination-risk estimation for language-model outputs. Part III extends risk
modeling to multimodal grounding failures, and Part IV studies how posterior
risk can drive abstention, verification, and other decision rules.

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
