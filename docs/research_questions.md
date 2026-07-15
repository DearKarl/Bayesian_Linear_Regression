# Research Questions

## Part 0: Reproduction and Evidence Map

- Which published uncertainty and hallucination results remain reproducible
  across software versions, model families, datasets, and compute budgets?
- Which methods estimate prompt difficulty, which estimate output error, and
  which produce calibrated probabilities rather than uncalibrated scores?
- Where do author-code results, paper-derived implementations, and new
  laboratory baselines agree or diverge?

## Part I: Bayesian Regression Foundations

- What does posterior prediction add beyond ordinary least squares in a
  controlled regression setting?
- When do probabilistic scores reveal value that RMSE does not?
- How should posterior samples be diagnosed before interpreting uncertainty?

## Part II: Bayesian Hallucination Risk Modeling

- Can hallucination be modeled as a calibrated posterior risk?
- Which uncertainty and evidence signals are predictive of hallucination?
- Does Bayesian calibration improve NLL, Brier score, ECE, and risk-coverage
  over heuristic scores?
- Can label noise and dependence among uncertainty, retrieval, and verifier
  signals be represented explicitly rather than ignored?

## Part III: Multimodal Hallucination Uncertainty

- How does hallucination risk change when answers must be grounded in images,
  documents, charts, or other modalities?
- Can Bayesian models separate object, attribute, relation, OCR, counting, and
  reasoning hallucinations?
- Which multimodal evidence signals remain reliable across models and datasets?

## Part IV: Bayesian Abstention and Decision Rules

- How should posterior hallucination risk drive answer, abstain, verify,
  regenerate, or human-review decisions?
- Can credible intervals over risk produce safer decision rules than point
  thresholds?
- What is the trade-off between answer coverage, hallucination rate, and
  utility?
- When is the expected value of clarification, retrieval, tool use, another
  agent, or human review greater than its cost?
