# Bayesian Methods Lab

### From uncertainty signals to posterior risk to reliable action.

Bayesian Methods Lab is a doctoral research programme on reliable language,
multimodal, and agentic AI. The laboratory studies how heterogeneous evidence
can be converted into calibrated posterior hallucination risk, how that risk
changes across models, domains, modalities, and interaction trajectories, and
how an AI system should act when the cost of being wrong is not zero.

```text
measure uncertainty  ->  infer posterior risk  ->  choose an action
```

The objective is not to prove that Bayesian methods always outperform
deterministic alternatives. It is to identify when posterior inference adds
measurable value and when that value supports safer decisions.

## Research Map

- **[Part 0: Reproduction and Evidence Map](part0_reproductions/README.md)**
  - [Route 1: Uncertainty signals](part0_reproductions/01_uncertainty_signals/README.md)
    - [LM-Polygraph pilot report](part0_reproductions/01_uncertainty_signals/REPORT.md)
    - [Experiment](part0_reproductions/01_uncertainty_signals/experiments/run_lm_polygraph_signal_pilot.py)
    - [Results](part0_reproductions/01_uncertainty_signals/reports/tables/signal_summary.csv)
  - [Route 2: Bayesian risk and decision theory](part0_reproductions/02_bayesian_risk_and_decision_theory/README.md)
  - [Route 3: Calibration and interaction](part0_reproductions/03_calibration_and_interaction/README.md)
  - [Route 4: RAG and heterogeneous evidence](part0_reproductions/04_rag_and_heterogeneous_evidence/README.md)
  - [Route 5: Multimodal risk decomposition](part0_reproductions/05_multimodal_risk_decomposition/README.md)
  - [Route 6: Agentic and multi-agent uncertainty](part0_reproductions/06_agentic_and_multiagent_uq/README.md)
- **[Part I: Bayesian Predictive Foundations for Uncertainty-Aware AI](docs/part1_bayesian_regression_foundations.md)**
  - [Original research-stage summary](docs/original_report_summary.md)
  - [Dataset and ethics note](docs/dataset_note.md)
  - [Artifact inventory](docs/artifacts.md)
- **[Part II: Bayesian Hallucination Risk Modeling](docs/part2_bayesian_hallucination_risk_modeling.md)**
- **[Part III: Multimodal Hallucination Uncertainty](docs/part3_multimodal_hallucination_uncertainty.md)**
- **[Part IV: Bayesian Abstention and Decision Rules](docs/part4_bayesian_abstention_decision_rules.md)**
- **Programme documents**
  - [Research questions](docs/research_questions.md)
  - [Roadmap](docs/roadmap.md)

## Current Evidence

Part I shows comparable point-prediction accuracy across the current linear
baselines while demonstrating the additional objects supplied by posterior
inference: predictive distributions, intervals, posterior samples, and proper
probabilistic scores. The repeated-split study does **not** support a general
RMSE superiority claim for the Gibbs model.

The first Part 0 pilot reproduces three official LM-Polygraph estimators on a
small prompt stress test. Likelihood and entropy signals recognize difficult
inputs, but their ranking is less reliable for the generations that were
actually fabricated or malformed. This gap between a raw signal and calibrated
outcome risk motivates Part II.

## Research Principles

- Separate point accuracy, uncertainty quality, calibration, and decision utility.
- Treat hallucination labels, verifier outputs, and agent agreement as noisy evidence.
- Evaluate with proper scores, calibration, risk--coverage, subgroup robustness,
  and repeated comparisons.
- Record reproduction scope and never label a paper-derived implementation as
  author-code reproduction.
- Prefer transparent baselines and falsifiable claims over broad performance narratives.

## Reproduce the Existing Studies

### Part I: Bayesian regression foundation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_boston_benchmark.py
python experiments/run_repeated_split_comparison.py
pytest -q
```

### Part 0, Route 1: LM-Polygraph signal pilot

Prepare the isolated environment described in [the route environment
record](part0_reproductions/01_uncertainty_signals/ENVIRONMENT.md), then run:

```bash
python part0_reproductions/01_uncertainty_signals/experiments/run_lm_polygraph_signal_pilot.py
```

The research programme is evolving. Results are retained when they are useful,
revised when evidence requires it, and never promoted beyond the experiment
that produced them.
