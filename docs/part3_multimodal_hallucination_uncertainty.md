# Part III: Multimodal Hallucination Uncertainty

## Research Question

How can Bayesian risk modeling quantify hallucination uncertainty when
language-model answers must be grounded in images, documents, charts, or other
modalities?

## Motivation

Part II models hallucination risk for text-only language-model outputs. Part III
extends the same posterior-risk idea to multimodal systems, where hallucinations
may arise from missing visual evidence, incorrect object grounding, wrong
attributes, counting errors, OCR failures, chart misreading, or unsupported
visual reasoning.

## Target Quantity

The main target is a multimodal hallucination risk:

```text
P(H = 1 | prompt, image_or_context, answer, evidence, multimodal uncertainty features)
```

Typed risks can also be modeled:

```text
P(H_k = 1 | prompt, image_or_context, answer, evidence, features)
```

Here `k` may represent object, attribute, relation, counting, OCR, document,
chart, or reasoning hallucination.

## Candidate Evidence Features

- object grounding score;
- attribute consistency score;
- image-text similarity;
- OCR consistency;
- retrieval or document support;
- VQA cross-check score;
- contradiction or entailment score;
- self-consistency disagreement;
- semantic disagreement;
- model or dataset identity.

## First Bayesian Direction

A first conceptual direction is a hierarchical Bayesian risk model:

```text
H_ik ~ Bernoulli(r_ik)
logit(r_ik) = alpha_k + model_effect + dataset_effect + feature_effects
```

The hierarchy may help compare hallucination types, models, and datasets while
sharing statistical strength. This should remain conceptual until a dedicated
Part III implementation PR chooses a dataset and feature interface.

## Evaluation

- binary NLL / NLPD by hallucination type;
- Brier score;
- AUROC / AUPRC;
- calibration by hallucination type;
- risk-coverage curves;
- subgroup calibration across models and datasets.

## Minimal Prototype Plan

1. Start from an existing multimodal hallucination benchmark.
2. Define a small set of hallucination types.
3. Extract transparent grounding and evidence features.
4. Fit a Bayesian or hierarchical risk model.
5. Compare against heuristic detectors and non-Bayesian baselines.
6. Evaluate calibration and risk-coverage by hallucination type.

## Limitations

- Benchmark labels may be noisy.
- Visual grounding signals may be imperfect.
- LLM/VLM judge signals are not ground truth.
- Results may be model- and dataset-specific.
