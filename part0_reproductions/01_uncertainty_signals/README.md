# Route 1: Uncertainty Signals

This route studies the observable quantities used to infer whether a language
model answer is unreliable. The central distinction is between an uncertainty
*signal* and a calibrated hallucination *risk*: entropy, likelihood, semantic
dispersion, verbal confidence, and latent activations are measurements, not
probabilities of hallucination by themselves.

## Representative Literature

- Vashurin et al. (2025), [Benchmarking Uncertainty Quantification Methods for
  Large Language Models with LM-Polygraph](https://aclanthology.org/2025.tacl-1.11/).
- Nguyen, Gupta, and Le (2026), [Probabilities Are All You Need](https://doi.org/10.1609/aaai.v40i38.40531).
- Li et al. (2026), [Semantic Volume](https://doi.org/10.1609/aaai.v40i37.40443).
- Sun et al. (2026), [Adaptive Bayesian Estimation of Semantic Entropy](https://doi.org/10.1609/aaai.v40i39.40595).

## Current Reproduction

The first pilot executes the official `lm-polygraph==0.5.0` estimators on a
small, deterministic prompt stress test using a locally available
`Qwen/Qwen2.5-VL-3B-Instruct` checkpoint in text-only mode. It compares mean
token entropy, mean negative token log-likelihood (named `Perplexity` in the
library version used), and negative sequence log-probability.

This is a **mechanism-level reproduction**, not a reproduction of the full
TACL benchmark. The prompt labels indicate whether a question is deliberately
answerable, ambiguous, or unsupported; they are not gold hallucination labels.

- [Pilot report](REPORT.md)
- [Experiment script](experiments/run_lm_polygraph_signal_pilot.py)
- [Prompt set](data/prompt_stress_test.csv)
- [Environment notes](ENVIRONMENT.md)

## Next Reproduction Steps

The next full experiment should use an established factuality dataset with
claim-level labels, include sampling-based semantic methods, repeat results over
multiple open-weight model families, and compare discrimination, calibration,
cost, and risk--coverage rather than a single AUROC value.
