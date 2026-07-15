# Roadmap

## Active: Part 0 Reproduction and Evidence Map

- Six-route literature and reproduction structure established.
- Route 1 official LM-Polygraph estimator pilot completed.
- Prompt-level stress labels separated from manually audited generation labels.

Next Route 1 milestones:

1. Replace the diagnostic prompt set with claim-level factuality benchmarks.
2. Add sampling-based semantic entropy and semantic-volume comparisons.
3. Evaluate several text-only open-weight model families and decoding settings.
4. Report calibration, compute cost, and risk--coverage with uncertainty intervals.

## Completed: Part I Predictive Foundation

- Bayesian regression benchmark.
- Probabilistic scoring.
- MCMC diagnostics.
- Repeated-split comparison.
- Part I reframed as the Bayesian predictive foundation for the doctoral programme.
- README expanded into a linked multi-level research map.
- Part III and Part IV documentation scaffolds.

Follow-up:

- Audit Part I sampler and written prior consistency in a dedicated PR before
  any paper-style release.

## Next Original-Method Study: Part II Bayesian Hallucination Risk Modeling

Milestones:

1. Part II documentation scaffold.
2. Text-only hallucination-risk dataset prototype.
3. Feature extraction for uncertainty, consistency, and evidence.
4. Bayesian logistic risk model.
5. Calibration metrics: NLL, Brier, ECE, AUROC, AUPRC.
6. Risk-coverage analysis.
7. Explicit observation models for noisy labels and correlated evidence signals.

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
