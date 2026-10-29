# Prospective local plan: exploratory digits engineering replication

Author: Libor “Burny” Burian. Phase: preregister. Analysis kind: **exploratory**. This document is written before local training or scoring; the controller will commit and freeze it with [research/protocol.json](../research/protocol.json). The supplied protocol and [study.py](../study.py) remain unchanged. Configuration values below are prospective design constants, not measured results.

Prior Haiku and Luna outcomes are already known. Registration prospectively constrains this local execution; it cannot create untouched-data confirmation. No local effect has been estimated, supported or falsified. Prior executions are context, never substitutes for local measurements: [src-prior-haiku](https://github.com/BurnyCoder/eureka-digits-haiku), [src-prior-luna](https://github.com/BurnyCoder/eureka-digits-luna). Source IDs resolve in [sources.json](../research/sources.json); `audit-fixture-source` identifies the inspected fixture in [evidence.json](../research/evidence.json). The completed [research review](research.md) contains the literature and implementation review.

## Goal, immediate objective and selection rationale

The project goal is verified completion with reproducible artifacts, whether the decay effect is positive, negative or inconclusive. The scientific question is: does AdamW weight decay improve held-out accuracy of the supplied small MLP on bundled digits? The current subgoal is to freeze an executable local plan; this phase's immediate objective is to document the inherited hypothesis, competing explanations, measurements and interpretation rules without running training.

Expected learning from the later experiments is whether the registered intervention changes correctness on this fixed split, whether seed effects agree, whether real-label learning beats negative controls, and whether a simple linear model is competitive. Use the supplied fixture because it answers these bounded questions cheaply and keeps switching costs low. A new architecture, tuned sweep or dataset would answer a different question and consume the publication and review budget. Plan approximately ten active minutes for this documentation and draft verification, as an effort estimate rather than a measurement. Retain the overall protocol budget of **180.0 active minutes and 60 CLI invocations** across phases; controller accounting is authoritative and explicit extensions must remain separate historical records.

The process stage remains exploration. The inherited directional hypothesis is a comparison target, not a newly discovered mature explanation. Record anomalies and compare predictions with observations before distilling a few supported claims. Writing that exposes a substantive design gap should trigger a justified return to research. [src-nanda-process](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand).

## Hypothesis, alternatives and distinguishing predictions

**h1 (inherited):** the mean paired test-accuracy difference for weight decay `0.01` versus `0` is positive. Both arms use AdamW; this is a decay ablation, not an AdamW-versus-Adam comparison or a comparison against an L2 loss penalty. All parameters, including biases, receive the specified decay. [src-adamw-paper](https://arxiv.org/abs/1711.05101), [src-torch-adamw](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html); audit-fixture-source.

| Explanation | Prediction recorded before local experiments | Discriminating observation and limits |
| --- | --- | --- |
| Decay improves this fixed setup (h1) | The mean paired accuracy difference is positive. | A conditional interval wholly above zero supports a local exploratory benefit; one wholly below zero contradicts h1. Inspect every seed rather than selecting a favorable one. |
| Decay has no detectable accuracy benefit | Effects are near zero or correctness vectors tie. | An interval including zero is inconclusive. Identical examplewise correctness differences force a zero bootstrap distribution, without establishing algorithmic equivalence. |
| Decay harms this setup | The paired difference is negative. | A wholly negative conditional interval supports local harm. Loss trajectories may suggest fitting differences but do not establish a mechanism. |
| Training randomness explains an apparent gain | Paired seed effects differ in magnitude or sign. | Matching initialization and minibatch order isolates the decay intervention within each seed; show the variation across seeds separately. Agreement across the registered seeds cannot establish training-population stability. |
| Leakage, labels or arithmetic are wrong | Useful shuffled-label performance, overlapping split IDs, mismatched labels or disagreement with recomputed accuracy may appear. | Audit IDs and saved predictions, then inspect any anomaly. A poor shuffled control and passing checks narrow specific failure classes; they cannot certify universal leakage absence. |
| A simple linear rule suffices | Logistic regression is competitive with the MLP descriptively. | Report both scores without post-hoc baseline selection. A descriptive MLP advantage weakens this alternative only on this split. |

Real-label models are expected to outperform the shuffled-label control. This is a diagnostic prediction, not a registered numerical significance threshold. The class-prior arm provides a constant majority-class floor; it is not random class sampling. Logistic regression is itself regularized, so it is not the no-decay neural ablation. [src-logistic](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html); audit-fixture-source. No quantitative effect size is predicted from the original AdamW paper for this task.

## Fixed methods, seeds and hyperparameter rationale

All values here mirror the supplied protocol. Run the fixture without changing its scientific definitions.

| Component / arm | Registered configuration | Purpose |
| --- | --- | --- |
| Shared neural model | `64 → 128 → 10`, ReLU hidden activation, cross-entropy objective | Small supplied classifier; final fixed epoch, no selected checkpoint |
| Shared neural training | AdamW, learning rate `0.001`, batch size `64`, `50` epochs; seeds `[0, 1, 2]` | Matched initialization and minibatch order within each seed |
| `regularized` | Weight decay `0.01` | Primary intervention; seed `0` also supplies the validation pilot |
| `unregularized` | Weight decay `0.0` | Matched primary ablation |
| `sensitivity` | Weight decay `0.001` | Descriptive local coefficient sensitivity, no tuning |
| `shuffled_labels` | Weight decay `0.01`; training-label permutation from the arm's seed | Negative control preserving training label counts |
| `logistic` | `LogisticRegression(C=1.0, max_iter=1000, random_state=0)`; other library defaults | Simple linear baseline, fitted on training data only |
| `class_prior` | Training majority-class prediction; fixture seed field `0` | Constant baseline using training labels only |

The methods run in the protocol order: `regularized`, `unregularized`, `sensitivity`, `shuffled_labels`, `logistic`, `class_prior`. Neural arms use all registered seeds; non-neural arms run once. The inherited small network and fixed training budget provide inexpensive feedback, not evidence of optimal architecture or hyperparameters. The three decay coefficients are a descriptive sensitivity check; no validation or test score chooses the primary coefficient. The limited seed set exposes some initialization sensitivity without supporting broad population claims. These are the supplied protocol's engineering choices, preserved rather than optimized here.

`study.py` seeds Python, NumPy and torch, uses a separately seeded order generator, enables deterministic torch algorithms and selects CUDA when available, otherwise CPU. Record actual environment, device and checkpoint provenance later; do not promise identical results across platforms or library releases. Validation loss is logged after each epoch but does not select epochs or hyperparameters. [src-reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html); audit-fixture-source.

## Data boundaries and contamination limits

Use scikit-learn's bundled `load_digits` data, not MNIST. Cast pixels to float32 and divide by the documented constant `16`; no normalization statistic is fitted. Freeze the supplied sequential stratified split: hold out `0.4` with split seed `42`, then split that held set equally into validation and test using the same seed and stratification. This implements the nominal `60/20/20` train/validation/test allocation, with integer sizes to be verified by the fixture later. Evaluation ID: `digits-stratified-42-test-v1`. [src-digits-loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), [src-splits](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

Fit parameters and shuffled labels using training indices only. Use validation labels for pilot scoring and diagnostic validation loss. Do not select a model, epoch, preprocessing rule or hyperparameter from test outcomes. The fixture does forward all feature rows, including test inputs, during checkpoint reload verification; therefore “validation-only pilot” describes scoring, not complete exclusion of test features. No test labels enter fitting or selection in the inspected source. This distinction requires checking outputs later rather than claiming dynamic isolation now. audit-fixture-source; [src-haiku-disclosure](https://github.com/BurnyCoder/eureka-digits-haiku/blob/main/reports/post-run-publication.md).

Split disjointness and full coverage are asserted by the fixture. Later verify saved IDs, labels, hashes and predictions as well. A random example split does not establish separation of writers or absence of related examples; the original UCI writer-separated design is not preserved by this repartition. All generalization claims are confined to the supplied example split, not unseen writers, independent datasets or new inference models. [src-uci-digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), [src-lones-pitfalls](https://arxiv.org/html/2108.02497v5).

## Primary metric, analysis plan and falsification

For registered seeds \(S\) and common test indices \(T\), compute

\[
d_{si}=\mathbf{1}(\hat y^{0.01}_{si}=y_i)-\mathbf{1}(\hat y^{0}_{si}=y_i),
\qquad \hat\Delta=\frac{1}{|S||T|}\sum_{s\in S}\sum_{i\in T}d_{si}.
\]

The primary metric is the mean over registered seeds of test accuracy(`regularized`) minus test accuracy(`unregularized`), on the accuracy fraction scale. Any percentage-point display must explicitly multiply this difference by `100`. No control, sensitivity arm, best seed or best epoch can replace the primary contrast.

1. Validate record IDs, methods, seeds, test membership, aligned example IDs and true labels, protocol/code/data hashes, checkpoint hashes and reload flags before aggregation. Recompute each accuracy from saved `y_true` and `y_pred`. Investigate mismatches without deleting the original evidence or silently dropping failed runs.
2. Compute every paired seed difference, their mean and sample standard deviation (`ddof=1`). Report all registered seeds and failures. An incomplete pair blocks the complete registered primary summary; any available subset must be labeled incomplete and exploratory.
3. Use `10000` paired bootstrap resamples of test example indices with NumPy generator seed `2026`. Each resample draws `|T|` indices with replacement and uses those same indices across both methods and all fitted seeds. Average correctness differences over the sampled examples and the fixed seed set. Use the central empirical percentile interval (quantiles `0.025` and `0.975`, linear quantile interpolation), making explicit the percentile construction discussed in the research review. Do not substitute the library's default BCa interval or independently resample the methods. [src-bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
4. Interpret this as a **conditional example-bootstrap interval for the fitted models**. It does not resample training runs, estimate training-population uncertainty, validate new writers or correct prior exposure to this evaluation set. Report seed variation separately, without conflating it with the conditional interval.
5. Apply the inherited falsification rule: if the interval's upper endpoint is below zero, it contradicts the directional h1 under this conditional analysis. An interval including zero is inconclusive, not proof of equivalence; this includes a degenerate `[0, 0]` interval. A lower endpoint above zero supports only a local exploratory positive contrast. No confirmatory p-value or multiplicity-adjusted discovery claim is planned.
6. Describe the sensitivity, shuffled-label, logistic and class-prior results alongside the primary arms. Compare test accuracy, seed variation where available and neural train/validation loss trajectories. Keep null and negative results, anomalous controls and convergence or execution failures visible. No post-hoc primary selection or hyperparameter tuning is permitted.

Accuracy can tie despite different parameters or confidence. Exact equality of correctness vectors implies a zero interval by the statistic's definition; it does not prove equality of fitted functions outside these examples. Any additional probability, mechanism or subgroup analysis must be declared as a post-hoc exploratory audit before interpreting it, and must not replace this metric.

## Execution order, artifacts and stopping rules

This phase writes documentation only and runs the required draft artifact verifier; no training, pilot or comparison command is run now. The controller owns registration, commits, publication, inference accounting and uploads. After freezing the protocol, later phases should use the supplied environment and this order from `cycles/001`:

```text
study.py check
study.py pilot
[controller publishes the regularized seed-0 validation pilot and checkpoint]
study.py experiments
```

The supplied interpreter is `C:/Users/Happy/Documents/projects/auto-eureka-scientist-agent-harness/.venv/Scripts/python.exe`. The comparison command requires the controller's pilot-publication artifact. Reuse the published pilot checkpoint for the matching final seed-0 regularized comparison, as implemented by the fixture; do not refit it merely to populate a second result record.

Later preserve timestamped training output, per-experiment JSON predictions and split IDs, full neural loss histories, protocol/code/dataset hashes, device and software metadata, safetensors neural checkpoints and NumPy baseline checkpoints. Tables and figures must be generated from saved records rather than manually typed scores. Aggregate summaries are not additional experiment records. Directly cite source and result IDs; register only additional local audits in the evidence ledger. Do not change byte-hashed evidence during presentation repairs. [src-luna-disclosure](https://github.com/BurnyCoder/eureka-digits-luna/blob/main/reports/post-run-publication.md).

Preserve all failures. Reuse valid measurements when repairing paths or formatting. If a substantive data, method or protocol error is discovered after freezing, stop the affected inference, document it, and return to the appropriate earlier phase for a prospective amendment; never silently rewrite history or claim repaired exploratory work was confirmatory. Stop at the authorized budget rather than omit inconvenient results. Do not expand the study without a separately recorded authorization.

## Observations at registration and expected handoff

Only source and fixture inspection are available: both primary arms use AdamW; decay applies to all model parameters; the pilot scores validation labels while reload checking forwards test features; final epochs are fixed; and prior executions are known. No local training, predictive accuracy, loss trajectory, bootstrap estimate or control outcome exists at this point. `audit-fixture-source` supports these source-level observations, not empirical model behavior.

The lesson from planning is to keep three distinctions explicit: a decay intervention versus an optimizer comparison, conditional example uncertainty versus training variability, and validation-only scoring versus complete feature isolation. After the controller freezes this plan, the next phase can check implementation and artifact integrity before fitting. Later reports must compare actual observations against the predictions above, preserve anomalies and state what remains unresolved. Engineering completion is the objective; neither a positive scientific effect nor a provider ranking is required.
