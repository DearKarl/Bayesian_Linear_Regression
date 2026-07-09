# Roadmap

This roadmap keeps Bayesian Methods Lab PR-sized and research-question driven.
The near-term direction is Bayesian hallucination-risk modeling for language
and multimodal AI systems, built on the uncertainty-reporting discipline from
Part I.

## 1. Part I Cleanup

- Keep Bayesian Regression Foundations as the completed foundation study.
- Preserve existing Boston Housing numeric results.
- Keep formulas and essential figures rendering correctly in documentation.
- Avoid claiming broad Bayesian superiority from the Part I benchmark.

## 2. Part II Hallucination-Risk Documentation Scaffold

- Define the text-only hallucination-risk target.
- Specify a first Bayesian logistic risk model.
- List candidate evidence and uncertainty features.
- Define evaluation metrics for probabilistic binary risk prediction.
- Connect the text-only scaffold to future multimodal uncertainty work.

## 3. Text-Only Hallucination-Risk Prototype

- Add a small reproducible text-only hallucination-risk dataset or synthetic
  benchmark.
- Implement transparent feature extraction for prompt, answer, and evidence.
- Fit baseline logistic regression and a Bayesian logistic risk model.
- Report NLL / binary NLPD, Brier score, AUROC, AUPRC, calibration, ECE, and
  risk-coverage.
- Avoid strong claims until repeated splits or external benchmarks support
  them.

## 4. Multimodal Hallucination Uncertainty Prototype

- Add a compact image-text or multimodal verification benchmark.
- Define grounding, contradiction, and missing-evidence features.
- Compare text-only and multimodal risk features.
- Report calibration and risk-coverage, not only classification accuracy.

## 5. Bayesian Abstention And Decision-Rule Experiments

- Convert hallucination-risk posteriors into abstention or escalation policies.
- Compare fixed thresholds, posterior-interval rules, and expected-utility
  decision rules.
- Evaluate residual hallucination risk versus answer coverage.
- Document when Bayesian uncertainty changes the downstream decision.
