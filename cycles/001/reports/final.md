# Final report: exploratory digits engineering replication

Author: Libor “Burny” Burian. Stage: distillation. Analysis kind: **exploratory**.

The local study has training, saved predictions and checkpoints, comparisons, interpretation and a substantive review. The predicted positive accuracy benefit of the supplied decay intervention was not observed. The registered conditional analysis remains **inconclusive**, and the engineering artifacts are ready for controller publication checks. Prior Haiku and Luna results were known; this execution cannot supply new confirmation on untouched data, establish agent superiority or demonstrate general scientific reliability.

## Objective and final evidence

The project goal was verified engineering completion with reproducible artifacts, whether the scientific effect was positive, negative or inconclusive. This phase reconciles the publication documents and requirements with the preserved evidence. Predictions and alternatives preceded the final non-fitting checks in [publish-plan.md](../research/publish-plan.md). Reusing the existing evidence has low switching cost and directly tests publication drift; another training run would not resolve a formatting or documentation defect.

<!-- BEGIN GENERATED FINAL MEASUREMENTS -->

| Test arm | Recorded seeds | Mean accuracy |
| --- | --- | --- |
| `regularized` | 3 | 96.759259% |
| `unregularized` | 3 | 96.759259% |
| `sensitivity` | 3 | 96.759259% |
| `shuffled_labels` | 3 | 9.537037% |
| `logistic` | 1 | 95.555556% |
| `class_prior` | 1 | 10.277778% |

| Primary quantity | Recorded value |
| --- | --- |
| Mean paired accuracy difference | 0.000000 |
| Conditional percentile interval | [0.0, 0.0] |
| Seed effect sample SD | 0.000000 |
| Registered decision | inconclusive |

<!-- END GENERATED FINAL MEASUREMENTS -->

The table is generated from [result records](../results/), the frozen protocol and [comparison-summary.json](comparison-summary.json) by [audit_publish.py](audit_publish.py). The [complete comparison report](experiments.md#generated-measurements) contains all individual scores, conditional interval settings, seed dispersion, losses and parameter differences. Derived summaries are not additional experiments.

| Claim and evidence IDs | Supported conclusion and boundary |
| --- | --- |
| claim-decay-tie: regularized_s0, regularized_s1, regularized_s2; unregularized_s0, unregularized_s1, unregularized_s2; audit-experiments-artifacts, audit-review-artifacts | Primary paired test correctness and predicted labels agree despite different fitted parameters. The directional benefit prediction was not observed; equal observed correctness does not establish equivalence or a universal null. |
| claim-local-sensitivity: sensitivity_s0, sensitivity_s1, sensitivity_s2 and the primary arms | The registered weaker coefficient also ties per-seed accuracy. This descriptive local check is neither an optimum search nor evidence about other settings or distributions. |
| claim-controls-and-baseline: shuffled_labels_s0, shuffled_labels_s1, shuffled_labels_s2; logistic_s0; class_prior_s0 and the real-label arms | Real-label neural models and penalized logistic regression exceed the negative controls. Logistic regression is descriptively below the neural models. No inferential superiority or theoretical chance test was registered for these contrasts. |
| claim-local-artifact-usability: pilot_regularized_s0; audit-pilot-artifacts, audit-experiments-artifacts, audit-analysis-consistency, audit-review-artifacts | Local metric, split, provenance, history/log and checkpoint checks passed; the published pilot checkpoint was reused for its matching test arm. This demonstrates local usability, not independent reproduction or absence of every bug. |

Claim definitions and limitations remain in [claims.json](../research/claims.json). Experiment IDs resolve in the result directory; source IDs in [sources.json](../research/sources.json); post-hoc audit IDs in [evidence.json](../research/evidence.json). No claim was upgraded from exploratory to confirmatory.

## Method and interpretation

The unchanged [fixture](../study.py), [protocol](../research/protocol.json) and [prospective exploratory plan](preregistration.md) define the fixed MLP, normalized bundled digits data, stratified example partition, matched seeds/order, training duration and coefficient settings. Both primary arms use AdamW: this is a decay intervention within that optimizer. It does not reproduce an optimizer-versus-optimizer or loss-penalty comparison. The connection to decoupled weight decay is grounded in [src-adamw-paper](https://arxiv.org/abs/1711.05101). All neural parameters, including biases, receive decay; logistic regression is penalized, and the class-prior baseline is a deterministic training-majority predictor.

The primary statistic averages paired accuracy differences across the fitted seeds. The registered percentile bootstrap shares example indices across both arms and all fitted seeds, following the paired-index principle described in [src-bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Identical observed correctness differences force its degenerate interval. That interval conditions on these fitted models and examples; it excludes uncertainty over future training runs and does not establish equivalence. No test outcome selected a checkpoint, epoch or coefficient. Complete hyperparameter rationale and method settings remain in the prospective plan and paper rather than being duplicated here.

The decay and label interventions provide local causal contrasts under the fixed implementation. Control ordering is descriptive evidence against some gross label/scoring failures. Different primary parameters undermine the duplicate-checkpoint explanation, while agreeing predictions remain compatible with different confidence or behavior on other inputs. Post-hoc artifact audits constrain specific bug alternatives rather than prove scientific correctness. These distinctions are retained in [interpretation.md](interpretation.md) and [review.md](review.md).

## Predictions, observations and learning

- **Prediction: preserved scientific artifacts should match review.** The publish audit compares fixture, protocol, registration, result and checkpoint hashes with the saved review baseline. Matching bytes retain the earlier scoped checks without refitting. Evidence: audit-publish-consistency and audit-review-artifacts.
- **Alternative: aggregation or scoring drift explains the tie.** The saved labels and predictions reproduce record accuracy and the registered paired summary in the publish audit. Earlier split/checkpoint/log checks remain available. This narrows the tested alternatives without excluding unknown bugs. Evidence: audit-publish-consistency, audit-experiments-artifacts.
- **Prediction: references and publication copies should agree.** Claim IDs, requirement paths, local document links and saved site records are checked; PDF copies retain the visually reviewed hashes. README now links this report and displays generated comparative scores and shared examples. The reproduction guide now distinguishes completed local work from controller publication. Evidence: audit-publish-consistency, audit-review-presentation.
- **Alternative: presentation repairs require new measurements.** Review repaired copied-paper links, omitted settings, pagination and example attribution using existing artifacts. This phase changes documentation and adds a non-fitting consistency check; no new scientific measurement, tuned search or training run was needed. Initial failures and earlier report versions remain preserved. Evidence: audit-review-initial, audit-review-site, audit-review-presentation.

The useful lesson is that changed fitted parameters and unchanged test accuracy are compatible. Here the engineering execution produced inspectable artifacts while the directional scientific prediction received no observed accuracy support. Writing and review sharpened that distinction without extending the claim. This follows the exploration-to-understanding-to-distillation process in [src-nanda-process](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand); it is a process heuristic, not a guarantee of correctness.

## Reconciliation, reproducibility and remaining limits

The [README](../../../README.md), [reproduction guide](../docs/reproduction.md), generated [all-experiment summary](summary.md), [paper](paper.pdf), [interactive explorer](../../../site/index.html), claims and [requirement checklist](../research/requirements.json) describe the same local evidence. Generated operational-status blocks retain the controller's recorded cutoff and must not be read as this phase's acceptance or final deployment status. Empirical tables and figures derive from saved files. The reproduction guide identifies the controller render command and local presentation postprocessor; repairs never require training.

The required canonical non-training check and the final consistency audit passed and are recorded in [publish-audit.json](../research/publish-audit.json). The substantive review already exercised the full unit suite and artifact audit. The required publish draft verifier also passed without errors; it checks artifact requirements without accepting the phase or validating scientific conclusions. Publication success is not inferred from a draft check. An initial checklist-edit command used a project-relative path from the cycle directory and failed before writing; correcting the path completed that documentation update without touching measurements.

Remaining scientific limits are prior-result exposure; a fixed example split without writer isolation; limited seed/coefficient sensitivity; conditional rather than training-population uncertainty; no equivalence test, calibrated-probability comparison or new-data evaluation; and no independent fresh-environment or cross-platform reproduction. Pilot scoring used validation labels, but checkpoint reload verification forwarded test features. The negative controls cannot certify absence of leakage or dependence. Timings mix checkpoint reuse, fitting and startup/device effects and are not controlled speed comparisons. Checkpoints do not contain optimizer state for exact interrupted-training resumption. [interpretation.md](interpretation.md); [src-reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

The paper's rendered pages were visually inspected during review, and their hashes are checked here. Website logic passed a minimal DOM/canvas-adapter probe, but browser connection and Windows automation failures prevented browser visual inspection. Responsive rendering, actual browser canvas output, keyboard/screen-reader behavior and deployed-page behavior remain unverified. The all-run website loss chart lacks a per-run legend; filtering and the labeled static comparison plots provide narrower views. Dependency versions were preserved for the fixed study; no upgrade or full supply-chain audit was performed. [review.md](review.md), audit-review-site, audit-review-presentation.

The controller still owns artifact regeneration, final status refresh, commits, Pages publication, model uploads and model cards. It should reapply `reports/repair_presentation.py` after rendering and preserve the reviewed example attribution and portable paper links, then perform its publication checks. This report makes no claim that those future actions succeeded. The local phase is ready to finish; no substantive evidence gap requires returning to an earlier research phase, and no new cycle is started.
