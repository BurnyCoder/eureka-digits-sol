# Initial validation pilot

Stage: exploration. Analysis kind: exploratory. The supplied regularized initial model trained successfully, saved its full loss trajectory and validation predictions, and produced a reloadable safetensors checkpoint. This is one execution on a known evaluation fixture, not new confirmation or a comparison of inference models. The inherited weight-decay hypothesis remains untested by this single arm.

## Objective, design and provenance

The project goal is reproducible engineering completion regardless of the eventual scientific effect. This phase's subgoal is the first trained model for controller publication. The immediate objective, expected learning, alternatives, effort estimate and switching-cost rationale were recorded before fitting in [pilot-plan.md](../research/pilot-plan.md). The [preregistration](preregistration.md) and frozen [protocol](../research/protocol.json) remain authoritative. Use this small fixed fit because it cheaply checks actual optimizer execution, full loss logging, predictive scoring and trained serialization together.

Experiment ID: **pilot_regularized_s0**, with the complete raw measurements in [its result record](../results/pilot_regularized_s0.json). The configuration, predictions, example IDs, partition IDs, code/protocol/data hashes, hardware, software versions and checkpoint hash are preserved there or in the hash-bound protocol. The additional post-hoc artifact review is **audit-pilot-artifacts**, in [pilot-audit.json](../research/pilot-audit.json), registered in [evidence.json](../research/evidence.json). It is an audit, not another experiment. The complete timestamped training log is identified by path and SHA in that audit.

Training used the fixture's cross-entropy objective, fixed final epoch and AdamW with decay applied to all parameters, including biases. There was no checkpoint selection, schedule change, tuned coefficient or early stopping. Exact settings and measured scores below are generated from the saved result and protocol. The audit also preserves a complete protocol snapshot and reconstructs the installed optimizer's effective parameter-group defaults without taking an optimizer step. These are source-based configuration reconstructions; the fixture saves model weights, not optimizer state, so exact interrupted-training resumption is not supported. Both primary arms later use AdamW; this pilot cannot establish an optimizer advantage. [src-torch-adamw](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html), [src-adamw-paper](https://arxiv.org/abs/1711.05101).

Data are the bundled scikit-learn UCI digits subset, normalized by the fixed documented pixel maximum. The stratified example partition is constructed before fitting, and no normalization parameter is estimated. Label scoring here uses validation indices. During the fixture's internal exact checkpoint comparison it forwards every feature row, including test features; therefore this is validation-only scoring, not complete test-feature isolation. The independent audit forwards validation features only and does not compute test accuracy. [src-digits-loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), [src-splits](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html); `audit-fixture-source`, `audit-pilot-artifacts`.

## Generated measurements

<!-- BEGIN GENERATED PILOT -->

| Quantity | Value |
| --- | --- |
| Experiment | pilot_regularized_s0 |
| Analysis / scored split | exploratory / validation |
| Architecture / activation | 64 → 128 → 10 / ReLU |
| Seed / split seed | 0 / 42 |
| Optimizer / learning rate / decay | AdamW / 0.001 / 0.01 |
| Epochs / batch size | 50 / 64 |
| Train / validation / test examples | 1078 / 359 / 360 |
| Validation correct / total (errors) | 347 / 359 (12) |
| Validation accuracy | 0.9665738162 (96.6574%) |
| Training loss: first → final epoch | 2.21190043 → 0.05678514 |
| Validation loss: first → final epoch | 2.08766460 → 0.12022846 |
| Local validation-loss increases | 7 |
| Experiment elapsed seconds | 6.083362 (fixture timer; not end-to-end runtime) |
| Training device / hardware | cuda / NVIDIA GeForce RTX 5070 Laptop GPU |
| Torch / NumPy | 2.14.1+cu130 / 2.5.4 |
| Checkpoint bytes / SHA-256 | 38728 / `a57cb6e24f5e7d2a3e83da8efd97233a4b7e0e5b8492f89b3cbe6064cb86cc3d` |

<!-- END GENERATED PILOT -->

![Complete unsmoothed pilot loss history](../figures/pilot-loss.png)

The plotted training loss is an example-weighted average of minibatch losses encountered while parameters change within an epoch. Validation loss is measured once after that epoch with fixed parameters. Their difference is not a directly matched final-model generalization gap. Axes use linear scales with a zero loss origin; no smoothing or omitted epochs. Source: `pilot_regularized_s0`; implementation: `audit-fixture-source`; rendering: [src-matplotlib-plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html).

## Predictions, observations and learning

| Prediction recorded before fitting | Actual observation | Interpretation and competing explanation |
| --- | --- | --- |
| Training and validation cross-entropy should fall overall, without requiring monotonicity. | Both final losses are below their initial logged values; validation has small local increases. All saved losses are finite. | Compatible with effective real-label fitting. Local increases do not establish late overfitting or a regularization mechanism; no schedule was selected from them. |
| Saved validation accuracy should equal prediction arithmetic; labels and IDs should match the frozen partition. | Recomputed accuracy is identical, labels and IDs match, and splits have unique IDs with disjoint full coverage. | Narrows arithmetic, wrong-split and stale-data explanations. It does not rule out writer dependence, related examples or every leakage path. |
| Trained weights should reload exactly and reproduce validation predictions independently. | Fixture exact-logit comparison passed; the audit reproduced all validation predictions and final validation loss on the recorded device. Checkpoint bytes match the saved SHA. | Supports local trained-artifact integrity. Cross-platform bitwise identity remains untested. |
| The phase should produce only the pilot experiment without changing scientific files. | At first audit only the pilot result existed; frozen code/protocol hashes matched registration and the saved record. Every epoch in the training log matched the saved history, and the process exited successfully. | Scope was preserved. No comparative arms or sensitivity runs were fitted. |

Evidence for training behavior is the saved experiment `pilot_regularized_s0`; evidence for independent consistency checks is `audit-pilot-artifacts`. The pilot supports a narrow existence claim: the supplied pipeline produced useful validation predictions and a verifiable trained artifact in this environment. The inherited directional decay claim, seed effects, negative-control behavior and linear-baseline competitiveness remain unresolved. No primary effect estimate, confidence interval, significance claim or equivalence claim can be inferred from this single arm. No failure or convergence warning appeared in the inspected training log; there was no fitting repair or repeated training.

## Validity review and next action

Correctness review covered prediction arithmetic, validation membership and labels, partition coverage, complete finite history, log/history agreement, protocol/code/data/checkpoint hashes and validation-only reload predictions. The audit code has no fitting path; it instantiates an optimizer solely to expose defaults. It loads safetensors without pickle and uses local files without credentials or network writes. This review supports the artifact checks above, not universal model or research reliability.

Loss reduction and accuracy alone cannot separate legitimate learning from all contamination explanations; later negative controls and paired ablations remain necessary. Prior Haiku and Luna outcomes were known before this run, so all conclusions remain exploratory. The random example split provides no verified separation by writer and does not establish generalization to new writers or datasets. One fitted seed provides no estimate of training variability; reported runtime is a single fixture timing, not a benchmark. Seed and deterministic-algorithm settings do not guarantee cross-release or CPU/GPU identity. [src-uci-digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), [src-reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html), [src-lones-pitfalls](https://arxiv.org/html/2108.02497v5); [prospective limitations](preregistration.md).

The important learning is that the previously untested actual fit, trained reload and saved validation score now work. Continue to **controller pilot publication**, preserving the checkpoint for later matched reuse. The present phase does not create a publication marker, publish files or execute comparisons. Only after the controller publishes this result/model may the experiments phase test decay, controls and seed sensitivity. This maintains inexpensive feedback and concise claims, consistent with [src-nanda-process](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand).

To rerun the independent audit and regenerate every numeric table entry and the loss figure, run from the cycle directory with the supplied research interpreter:

```powershell
& 'C:/Users/Happy/Documents/projects/auto-eureka-scientist-agent-harness/.venv/Scripts/python.exe' reports/build_pilot.py
```

This reads existing measurements, preserves the first audit and takes no optimizer steps. A repeat generation passed without changing the audit or retraining. Source IDs resolve in [sources.json](../research/sources.json). The required pilot draft verifier returned `complete: true`, `errors: []`, `draft_only: true`; it does not accept the phase or certify the scientific interpretation.
