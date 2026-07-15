# Part 0: Reproduction and Evidence Map

Part 0 is the empirical literature layer of Bayesian Methods Lab. It precedes
the original-method studies in Parts I--IV and asks a deliberately conservative
question: **which published uncertainty and hallucination results can be
reproduced, under what computational assumptions, and which limitations remain
after reproduction?**

The six routes mirror the main strands of the 2025--2026 literature. Each route
has its own reading list, reproduction status, experimental scope, and planned
outputs. A paper is marked as *reproduced* only when author-released code and a
documented environment have been executed. Implementations reconstructed from
equations are labelled *paper-derived* and are not presented as author-code
reproductions.

## Routes

1. [Uncertainty signals](01_uncertainty_signals/README.md): token probability,
   entropy, semantic consistency, latent representations, and efficient signal
   estimation.
2. [Bayesian risk and decision theory](02_bayesian_risk_and_decision_theory/README.md):
   posterior risk, task-dependent utility, and epistemic excess risk.
3. [Calibration and interaction](03_calibration_and_interaction/README.md):
   verbal confidence, calibration-aware training, clarification, and multi-turn
   calibration.
4. [RAG and heterogeneous evidence](04_rag_and_heterogeneous_evidence/README.md):
   retrieval uncertainty, factuality, faithfulness, and evidence fusion.
5. [Multimodal risk decomposition](05_multimodal_risk_decomposition/README.md):
   visual grounding, perception--reasoning decomposition, and typed multimodal
   hallucinations.
6. [Agentic and multi-agent uncertainty](06_agentic_and_multiagent_uq/README.md):
   trajectory-level uncertainty, correlated agents, verification, and adaptive
   control.

## Reproduction Standard

Every completed reproduction should record the source paper and code revision,
hardware and software environment, dataset and model versions, deviations from
the published protocol, raw outputs, evaluation code, visual summaries, and a
careful statement of what was and was not reproduced. Small diagnostic runs are
useful for testing mechanisms, but they are not substitutes for the full
published benchmark.

## Connection to the Research Programme

Part 0 identifies dependable measurements and unresolved failure modes. [Part
I](../docs/part1_bayesian_regression_foundations.md) establishes the laboratory's
Bayesian predictive and evaluation discipline. Parts II--IV will then develop
new posterior-risk models, extend them to multimodal and agentic settings, and
convert calibrated risk into sequential decisions.
