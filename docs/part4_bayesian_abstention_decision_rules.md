# Part IV: Bayesian Abstention and Decision Rules

## Research Question

How should posterior hallucination risk guide decisions such as answering,
abstaining, verifying, regenerating, or escalating to human review?

## Motivation

Part II and Part III estimate hallucination risk. Part IV studies how to use
that risk in decision making. The goal is not only to detect hallucination, but
to decide what an AI system should do under uncertainty.

## Bayesian Decision Object

Candidate actions are:

```text
d in {answer, abstain, verify, regenerate, human_review}
```

The Bayesian decision rule is:

```text
d* = argmin_d E[L(d, H) | evidence]
```

Here `H` is the hallucination event and `L` is a task-specific loss. The loss
can encode the cost of hallucinating, abstaining, verifying, regenerating, or
routing an answer to human review.

## Conservative Risk Rule

A conservative rule can use an upper credible bound:

```text
answer if upper_95(P(H = 1 | evidence)) < threshold
```

This is more conservative than using the posterior mean alone, because uncertain
cases can be routed to verification or review even when their mean estimated
risk is moderate.

## Evaluation

- answer coverage;
- selective hallucination risk;
- risk-coverage curve;
- utility under different decision costs;
- abstention rate;
- verification rate;
- hallucination reduction among answered outputs.

## Minimal Prototype Plan

1. Reuse calibrated risks from Part II.
2. Compare posterior-mean thresholds with credible-upper-bound thresholds.
3. Evaluate risk-coverage curves.
4. Add a simple cost function.
5. Extend to multimodal risks from Part III.

## Limitations

- Decision costs are task-dependent.
- Overly conservative rules may reduce usefulness.
- Calibration failures can cause unsafe decisions.
- Human-review assumptions should be made explicit.
