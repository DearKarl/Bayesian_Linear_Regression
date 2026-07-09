# Bayesian Methods Lab

*From Bayesian uncertainty to reliable multimodal AI.*

Bayesian Methods Lab is a long-term research project on Bayesian approaches to
uncertainty quantification, calibration, and risk-aware decision making in
reliable AI. The lab studies how Bayesian posterior inference can support
hallucination quantification, calibrated confidence, and trustworthy decisions
in language and vision-language systems.

## Research Philosophy

The laboratory starts from research questions rather than model preferences.
Bayesian methods are studied because they provide a principled language for
posterior uncertainty, predictive calibration, and decision making under
uncertainty. Each research part is designed to be independently reproducible,
and every conclusion must be supported by empirical evidence.

The goal is not to prove that Bayesian methods universally outperform classical
or deterministic approaches. The goal is to identify when Bayesian posterior
inference provides measurable value, how that value should be evaluated, and how
uncertainty can inform safer AI-system behavior.

## Why Bayesian?

Modern large language models often produce fluent answers without clearly
communicating how reliable those answers are. This creates a gap between
generation quality and decision reliability. Bayesian methods are worth studying
because they treat uncertainty as a central object of inference rather than an
afterthought.

For reliable AI systems, calibrated uncertainty should influence downstream
actions. A model should not only produce an answer; it should also estimate when
that answer may be unsupported, contradicted, or unsafe to use without further
verification. Bayesian posterior inference provides one principled framework
for connecting prediction, uncertainty, and decision making.

## Research Roadmap

| Part | Research Theme | Current Stage | Documentation |
| :--- | :--- | :--- | :--- |
| Part I | Bayesian Regression Foundations | Completed | [Report](docs/part1_bayesian_regression_foundations.md) |
| Part II | Bayesian Hallucination Risk Modeling | In Progress | [Research Plan](docs/part2_bayesian_hallucination_risk_modeling.md) |
| Part III | Multimodal Hallucination Uncertainty | Planned | [Scaffold](docs/part3_multimodal_hallucination_uncertainty.md) |
| Part IV | Bayesian Abstention and Decision Rules | Planned | [Scaffold](docs/part4_bayesian_abstention_decision_rules.md) |

## Research Overview

- **Part I: Bayesian Regression Foundations** establishes the methodological
  foundation of the laboratory. This study investigates how Bayesian posterior
  inference differs from classical point estimation in a controlled regression
  setting. It focuses on posterior prediction, probabilistic scoring,
  uncertainty quantification, and Bayesian diagnostics to build a reproducible
  baseline before moving to more complex AI applications.

- **Part II: Bayesian Hallucination Risk Modeling** transfers Bayesian
  uncertainty modeling from regression to large language models. The objective
  is to develop Bayesian methods for estimating calibrated hallucination risk by
  integrating posterior inference with uncertainty signals, consistency
  measurements, and external evidence. Rather than treating hallucination
  detection as a binary classification problem, this part aims to model
  hallucination as a probabilistic posterior risk.

- **Part III: Multimodal Hallucination Uncertainty** extends the Bayesian risk
  modeling framework to multimodal large language models. This study
  investigates how Bayesian methods can quantify uncertainty when language
  generation must be grounded in images, documents, charts, or other modalities.
  The long-term goal is to understand different categories of hallucination and
  to build calibrated uncertainty models that remain reliable across multimodal
  tasks.

- **Part IV: Bayesian Abstention and Decision Rules** studies how Bayesian
  posterior risk can guide reliable decision making in generative AI systems.
  Instead of using uncertainty only for prediction, this part investigates
  Bayesian decision rules for answering, abstaining, verification, regeneration,
  and human review. The objective is to transform calibrated posterior
  uncertainty into practical decision strategies for trustworthy language and
  multimodal AI systems.

## Research Workflow

```text
Posterior Inference
        |
        v
Uncertainty Quantification
        |
        v
Hallucination Risk Modeling
        |
        v
Decision Making
```

The four parts are not independent projects. They form a single research
pipeline: Part I establishes reproducible posterior inference and probabilistic
evaluation; Part II applies those ideas to hallucination risk in language-model
outputs; Part III extends the risk framework to multimodal grounding failures;
and Part IV studies how posterior risk should guide actions.

## Repository Navigation

| Looking for... | Go to |
| :--- | :--- |
| Completed Bayesian regression foundation study | [Part I report](docs/part1_bayesian_regression_foundations.md) |
| Hallucination-risk research plan | [Part II scaffold](docs/part2_bayesian_hallucination_risk_modeling.md) |
| Multimodal hallucination direction | [Part III scaffold](docs/part3_multimodal_hallucination_uncertainty.md) |
| Bayesian decision-rule direction | [Part IV scaffold](docs/part4_bayesian_abstention_decision_rules.md) |
| Lab-wide research questions | [Research questions](docs/research_questions.md) |
| Development roadmap | [Roadmap](docs/roadmap.md) |
| Boston Housing ethics and benchmark caveats | [Dataset note](docs/dataset_note.md) |

Reproduction commands are listed inside each Part report. Use `--n-repeats`
with `experiments/run_repeated_split_comparison.py` for a faster repeated-split
smoke run.

## Future Research Directions

The long-term vision is to develop Bayesian tools for trustworthy generative AI.
Future work will study calibrated hallucination-risk estimation, multimodal
uncertainty, selective prediction, and Bayesian decision rules for systems that
must decide when to answer, abstain, verify, regenerate, or request human
review.

This direction treats uncertainty as part of the AI system itself. Reliable
language and multimodal models should not only generate plausible outputs; they
should also expose risk in a form that supports evidence-aware decisions.

## References

- scikit-learn,
  [BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
  and
  [ARDRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html).
- scikit-learn example,
  [Comparing Linear Bayesian Regressors](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html).
- Harrison, D. and Rubinfeld, D. L. (1978).
  [Hedonic housing prices and the demand for clean air](https://doi.org/10.1016/0095-0696(78)90006-2).

Detailed references for later research parts will be collected in the relevant
Part documents as those studies mature.
