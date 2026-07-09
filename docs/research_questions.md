# Research Questions

Bayesian Methods Lab now uses Bayesian regression as a foundation for studying
hallucination-risk modeling in language and multimodal AI systems. The central
theme is uncertainty-aware prediction: not only what a model outputs, but how
risky, calibrated, and decision-relevant that output is.

## Part I: Bayesian Regression Foundations

- What does posterior inference add beyond ordinary least squares when the
  model class is intentionally simple?
- When do posterior predictive distributions improve probabilistic scores even
  when RMSE is similar across baselines?
- How should uncertainty diagnostics, repeated splits, and cautious
  interpretation be reported in a small benchmark?

## Part II: Bayesian Hallucination Risk Modeling

- Can Bayesian models estimate `P(hallucination = 1 | prompt, answer, evidence,
  uncertainty features)` in a calibrated way?
- Which evidence and uncertainty features are most useful for text-only
  hallucination-risk prediction?
- How should binary NLPD, Brier score, calibration curves, ECE, AUROC, AUPRC,
  and risk-coverage be used together?
- When should posterior uncertainty in the risk score trigger abstention,
  retrieval escalation, or human review?

## Part III: Multimodal Hallucination Uncertainty

- How can hallucination-risk modeling extend from text-only evidence to image,
  audio, video, or cross-modal evidence?
- Which uncertainty features capture missing grounding, cross-modal
  contradiction, or weak evidence alignment?
- How should multimodal hallucination risk be evaluated when labels are noisy,
  subjective, or task-dependent?

## Part IV: Bayesian Abstention and Decision Rules

- How should a calibrated hallucination-risk posterior be converted into a
  decision rule?
- What abstention, escalation, or verification policies reduce residual risk
  without discarding too many useful answers?
- How should expected utility, cost-sensitive decisions, and risk-coverage
  trade-offs be reported?
