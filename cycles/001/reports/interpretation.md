# Interpretation of the fixed digits replication

Author: Libor “Burny” Burian. Stage: distillation. Analysis kind: **exploratory**. The engineering objective is verified completion with reproducible artifacts regardless of scientific effect direction. Known prior outcomes prevent new confirmation on untouched data. This report interprets preserved measurements; no training, test-selected tuning or new predictive evaluation was performed in analysis.

## Motivation, context and claims

The scientific question is whether the supplied AdamW decay intervention improves held-out accuracy of the supplied small digit classifier. The strongest observation is narrower: the primary paired models have identical test correctness despite different fitted weights. The inherited rule calls the conditional result inconclusive. The positive directional prediction was not observed, but neither an equivalence claim nor a general rejection of weight decay follows. **claim-decay-tie**; evidence: regularized_s0, regularized_s1, regularized_s2, unregularized_s0, unregularized_s1, unregularized_s2, audit-experiments-artifacts.

The weaker-decay arm also ties the primary arms' per-seed accuracy. Real-label models exceed the negative controls, and penalized logistic regression supplies a useful simple baseline below the neural results descriptively. The saved artifacts pass the scoped local audits. These are the remaining claims in [claims.json](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/research/claims.json): **claim-local-sensitivity**, **claim-controls-and-baseline**, **claim-local-artifact-usability**. None establishes agent/model superiority or general scientific reliability.

All empirical scores, paired effects, interval endpoints, seed dispersion and parameter differences are generated from saved files in [experiments.md](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/reports/experiments.md#generated-measurements). This interpretation deliberately links that table rather than hand-enter empirical scores. Raw experiment IDs resolve in [results/](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/results); source IDs resolve in [sources.json](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/research/sources.json); local audit IDs resolve in [evidence.json](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/research/evidence.json). The derived [comparison-summary.json](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/reports/comparison-summary.json) is not an additional experiment.

## Methods and causal scope

The unchanged [fixture](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/study.py) and [registered protocol](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/research/protocol.json) define the full configuration. The [prospective method table](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/reports/preregistration.md#fixed-methods-seeds-and-hyperparameter-rationale) specifies architecture, all coefficients, learning rate, batch size, duration, seeds and logistic settings. These were inherited for inexpensive fixed feedback, not chosen from outcomes. The classifier is a fully connected hidden-ReLU MLP trained with cross-entropy and AdamW. Both primary arms use that optimizer; their matched initialization and minibatch-order construction isolate the decay coefficient within each seed. Decay applies to biases as well as weights. The weaker coefficient is a descriptive sensitivity arm, and training-label permutation is the negative-control intervention. Logistic regression is penalized and linear; the majority baseline predicts a constant class from training counts. Evidence: audit-fixture-source; [src-torch-adamw](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html), [src-logistic](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Bundled digits are scaled by a known constant and partitioned by frozen, class-stratified example indices. No learned preprocessing statistic is estimated from held-out data. Parameters use training labels; validation loss is diagnostic and does not select an epoch or configuration. The pilot scores validation labels. Its reload check forwards all feature rows, including test inputs, so complete test-feature isolation cannot be claimed. Source inspection finds no test-label fitting or selection path; the artifact audit checks the saved membership, labels and scores. This is evidence for these specific boundaries, not a blanket certification of no leakage. Evidence: audit-fixture-source, audit-pilot-artifacts, audit-experiments-artifacts; [src-digits-loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), [src-splits](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

The final fixed epoch supplies the neural checkpoints. The matching regularized test arm reuses the published pilot checkpoint rather than a new fit; the validation and test records are different evaluations of that fitted model and must not be counted as independent training runs. Evidence: pilot_regularized_s0, regularized_s0, audit-experiments-artifacts; controller record [pilot-publication.json](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/research/pilot-publication.json).

For seed set S and shared test examples T, define d_si as regularized correctness minus zero-decay correctness on example i. The primary statistic is the mean of d_si across seeds and examples. The [analysis implementation](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/reports/compare_artifacts.py) draws test-example indices with replacement, shares those indices across methods and fitted seeds, and takes the registered central percentile endpoints. The frozen [analysis plan](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/reports/preregistration.md#primary-metric-analysis-plan-and-falsification) fixes resample count, generator seed and quantile interpolation. Seed-effect sample dispersion is reported separately. This interval conditions on the fitted seed set; it does not resample training or estimate uncertainty across new writers or datasets. [src-bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) explains shared-index pairing and distinguishes percentile intervals from the library's default BCa method; this study uses its explicit registered NumPy calculation.



<!-- BEGIN FROZEN SETTINGS -->

### Complete frozen settings

Generated directly from the registered protocol, without outcome-based selection.

| Setting | Registered value |
| --- | --- |
| `architecture` | `[64, 128, 10]` |
| `activation` | `"ReLU"` |
| `optimizer` | `"AdamW"` |
| `learning_rate` | `0.001` |
| `batch_size` | `64` |
| `epochs` | `50` |
| `seeds` | `[0, 1, 2]` |
| `split_seed` | `42` |
| `logistic_C` | `1.0` |
| `logistic_max_iter` | `1000` |
| `bootstrap_seed` | `2026` |
| `bootstrap_resamples` | `10000` |
| `weight_decay.regularized` | `0.01` |
| `weight_decay.unregularized` | `0.0` |
| `weight_decay.sensitivity` | `0.001` |
| `weight_decay.shuffled_labels` | `0.01` |

<!-- END FROZEN SETTINGS -->

## Predictions, observations and negative findings

Scientific predictions were recorded before fitting in [preregistration.md](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/reports/preregistration.md) and [experiments-plan.md](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/research/experiments-plan.md). Their prospective local timing does not remove prior-result exposure. The analysis audit expectations were separately recorded after results were known in [analysis-plan.md](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/research/analysis-plan.md); they are consistency predictions, not confirmatory hypotheses.

**Prediction or alternative:** The inherited h1 predicts a positive mean decay effect; no effect, harm and seed-dependent changes compete.

**Observation:** Every primary paired correctness vector matches. The saved aggregate and all resampled effects reproduce the degenerate tie.

**Interpretation:** The positive prediction was not observed. The interval is not wholly negative, so the registered directional falsification criterion is not met; the rule labels it inconclusive. Finite-example accuracy insensitivity remains compatible with changes in confidence or behavior elsewhere. Evidence: primary result IDs above; audit-analysis-consistency.

**Prediction or alternative:** Accidental reuse of identical primary checkpoints could explain a tie.

**Observation:** Paired state dictionaries differ in every primary pair.

**Interpretation:** This specific reuse explanation is inconsistent with the audited bytes. Different states alone do not demonstrate a regularization mechanism or calibration benefit. Evidence: audit-experiments-artifacts, reproduced by audit-analysis-consistency.

**Prediction or alternative:** The weaker coefficient might change accuracy; training seeds might matter more than the coefficient.

**Observation:** The sensitivity arm ties primary accuracy within each seed, while neural accuracy varies between registered seeds.

**Interpretation:** This narrow coefficient check found no accuracy advantage. It does not identify an optimum or establish stability over the training population. Evidence: sensitivity_s0, sensitivity_s1, sensitivity_s2 and primary result IDs.

**Prediction or alternative:** Real labels should outperform shuffled labels and the majority floor.

**Observation:** Every real-label neural result and logistic result exceed every negative-control result. Shuffled-label online training loss decreases while true-label validation loss ends above its initial value.

**Interpretation:** Compatible with learning useful label associations in the real-label arms and limited fitting of arbitrary labels in the control. It narrows gross label/scoring-error alternatives without excluding contamination or related-example dependence. Evidence: shuffled_labels_s0, shuffled_labels_s1, shuffled_labels_s2, class_prior_s0, logistic_s0 and real-label neural IDs.

**Prediction or alternative:** A simple linear baseline may be competitive.

**Observation:** Logistic accuracy is descriptively below each real-label neural result; generated scores show the size of the gap.

**Interpretation:** A useful linear baseline prevents attributing all useful accuracy to the neural architecture. No registered neural-versus-linear superiority interval or chance test was performed. Evidence: logistic_s0 and real-label neural IDs.

**Prediction or alternative:** Hashes, split membership, scores, histories and reloads should remain consistent.

**Observation:** The existing full artifact audit and primary summary reproduce; protected scientific files, results and checkpoints remain unchanged.

**Interpretation:** No checked provenance, score or reload defect was found. This does not prove every implementation property or general harness reliability. Evidence: audit-analysis-consistency.

The tie is a genuine negative finding for the predicted improvement in the observed statistic. If every d_si vanishes, any shared-example resample also has a vanishing mean. The interval's collapse therefore follows algebraically from these observed differences; it is not evidence of broad experimental power. Correctness equality also leaves open different wrong-class predictions. No additional probability, calibration, subgroup or new-data analysis was performed. Evidence: primary result IDs, audit-experiments-artifacts; analysis definition above.

## Related work and implications

Loshchilov and Hutter distinguish decoupled decay from an L2 gradient penalty in adaptive optimization and report image-classification benefits. That motivates the component being replicated but does not predict an accuracy gain for this fixed digits setting. This experiment cannot reproduce the paper's optimizer comparison because it intervenes only on decay within AdamW. The original implementation and license were inspected during [source research](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/reports/research.md); no external implementation was copied here. [src-adamw-paper](https://arxiv.org/abs/1711.05101), [src-adamw-code](https://github.com/loshchil/AdamW-and-SGDW).

The loader uses a subset of UCI optical digits, not MNIST. Repartitioning that subset randomly by example does not preserve the original dataset's writer-held-out construction; the current checks do not establish writer isolation. [src-digits-loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), [src-uci-digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). Meaningful simple baselines and explicit evaluation boundaries follow the methodological concerns in [src-lones-pitfalls](https://arxiv.org/html/2108.02497v5); that guide is context, not empirical evidence for the local effect.

The closest supplied-fixture executions are [src-prior-haiku](https://github.com/BurnyCoder/eureka-digits-haiku) and [src-prior-luna](https://github.com/BurnyCoder/eureka-digits-luna). They are prior exposure, not local measurements or an agent ranking. Their [Haiku disclosure](https://github.com/BurnyCoder/eureka-digits-haiku/blob/main/reports/post-run-publication.md) (src-haiku-disclosure) clarifies pilot feature access; the [Luna disclosure](https://github.com/BurnyCoder/eureka-digits-luna/blob/main/reports/post-run-publication.md) (src-luna-disclosure) motivates retaining historical bytes through presentation repairs. No sibling project was inspected.

## Limitations, validity review and decision

The fixed small task, limited seeds, previously exposed outcomes and example-based partition restrict inference. Conditional example resampling assumes examples are appropriate resampling units; unmeasured writer or related-example dependence can weaken its sampling interpretation. No equivalence margin, untouched evaluation set, writer-held-out replication, split sensitivity or broader hyperparameter sweep was tested. A constant paired effect across these fitted seeds does not imply no training-population variability. Saved argmax predictions and loss histories do not establish confidence calibration, causal feature mechanisms or out-of-distribution behavior.

The validity review distinguishes source inspection, intervention evidence and post-hoc auditing. Matched decay interventions support the narrow local contrast; the label intervention supplies a negative control. The parameter comparison is a post-hoc artifact check, not a mechanistic demonstration. Recomputed labels, indices, hashes, metrics, log/history agreement and checkpoint outputs narrow specified ordinary-bug alternatives. Exact reloads show local artifact usability; optimizer state was not saved for exact interrupted-training resumption, and independent fresh-environment reproduction was not performed. Cross-platform identity is not guaranteed. Evidence: audit-fixture-source, audit-experiments-artifacts, audit-analysis-consistency; [src-reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

Timestamped fitting logs and full negative-control results remain preserved. The expected test-first import failures documented in the core and comparison reports remain part of the engineering history; no scientific failed fit or discarded seed was found in the scoped log audit. The current non-training fixture check and focused statistic tests passed; their actual outputs are recorded in [analysis-audit.json](https://github.com/BurnyCoder/eureka-digits-sol/blob/main/cycles/001/research/analysis-audit.json). No scientific definition, result record or checkpoint was edited, and no training or rendering ran in analysis. Historical controller-generated global summaries can still describe earlier accepted cutoffs; their refresh, final paper/site generation, publication and acceptance belong to the controller and later phases.

Writing exposed no substantive error requiring a return to training or protocol amendment. It sharpened the distinction between changed weights, unchanged correctness and conditional uncertainty. The useful next action is controller review of these bounded claims and the requirements map, preserving all negative findings. Further fitting after observing the test tie would answer a new question and has low information value for the authorized handoff. This moves from exploration through understanding to distillation, with a return justified only by a concrete evidence gap. [src-nanda-process](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand).
