# Roadmap

## Completed: Part I Foundation

- Bayesian regression benchmark.
- Probabilistic scoring.
- MCMC diagnostics.
- Repeated-split comparison.
- README / Part I documentation cleanup.
- Part III and Part IV documentation scaffolds.

Follow-up:

- Audit Part I sampler and written prior consistency in a dedicated PR before
  any paper-style release.

## Next: Part II Bayesian Hallucination Risk Modeling

Milestones:

1. Part II documentation scaffold.
2. Text-only hallucination-risk dataset prototype.
3. Feature extraction for uncertainty, consistency, and evidence.
4. Bayesian logistic risk model.
5. Calibration metrics: NLL, Brier, ECE, AUROC, AUPRC.
6. Risk-coverage analysis.

## Planned: Part III Multimodal Hallucination Uncertainty

Milestones:

1. Choose multimodal hallucination benchmarks.
2. Define hallucination types.
3. Extract visual grounding / evidence features.
4. Fit hierarchical Bayesian risk models.
5. Evaluate by hallucination type and dataset.

## Planned: Part IV Bayesian Abstention and Decision Rules

Milestones:

1. Define actions: answer, abstain, verify, regenerate.
2. Define decision costs.
3. Compare posterior-mean thresholds with credible-upper-bound thresholds.
4. Evaluate selective risk and coverage.
5. Study practical reliability trade-offs.
