# Comparison execution plan

Stage: exploration. Analysis kind: exploratory. Written before this phase's fitting and test scoring. The supplied protocol, registration and study.py remain unchanged; the existing pilot is preserved. Prior outcomes are known, so prospective local predictions do not make this confirmatory.

The project goal is verified reproducible completion. The subgoal is the frozen comparisons, controls, ablation and sensitivity runs. The immediate objective is to execute `study.py experiments` through the existing complete-log wrapper after the supplied check, then audit saved artifacts without refitting. The controller's pilot-publication.json records publication and model verification; the matching final arm must reuse that checkpoint.

Expected learning: whether the registered decay contrast changes test correctness, whether its sign agrees across seeds, whether shuffled-label learning fails to generalize, and whether the linear baseline is competitive. Estimate a few minutes for fitting and roughly twenty minutes for artifact review and interpretation. This fixed small comparison has high information per unit time and negligible switching cost; adding architectures, tuning or alternate datasets would change the question and consume later-phase time.

## Predictions and discriminating alternatives

- The inherited h1 predicts positive mean paired test accuracy for regularized minus unregularized. Competing explanations are no accuracy effect, harmful decay, seed-dependent changes, or a bookkeeping/implementation error. Preserve all paired seeds. A conditional percentile interval wholly below zero contradicts the directional hypothesis; an interval containing zero is inconclusive, including a degenerate zero interval.
- Real-label neural and logistic arms should outperform shuffled labels and the constant majority-class control. Unexpectedly useful shuffled-label test predictions would motivate a leakage, scoring or control-implementation investigation. Poor shuffled performance narrows these alternatives but does not certify absence of all leakage.
- The registered weaker decay may tie or differ; it is descriptive sensitivity, never a replacement primary contrast or a basis for selecting a coefficient. Differences between seeds may exceed differences between decay arms.
- Every saved accuracy should equal arithmetic on aligned true labels and predictions. Splits, hashes, checkpoint reloads and complete histories should agree. Failures favor ordinary engineering explanations before a scientific mechanism.
- Identical correctness vectors predict an exactly zero example-bootstrap interval by construction, even if weights differ. A cheap post-hoc artifact audit may compare paired state dictionaries to distinguish unchanged weights from coarse accuracy insensitivity. This is exploratory and cannot demonstrate a general mechanistic explanation.

## Analysis and preservation

Use the frozen primary metric and analysis plan in research/protocol.json and reports/preregistration.md: shared test-example resamples across all paired fitted seeds, registered generator seed and resample count, central percentile interval with linear quantiles, and separate sample standard deviation of paired seed effects. Show every arm and outcome. No test-based model, epoch or hyperparameter selection. Test labels are scored only now; the pilot had already forwarded test features for reload verification.

Audit the exact expected experiment matrix, recompute metrics and split/data/code/protocol/checkpoint hashes, reload every model on its recorded device, check all epoch histories against complete timestamped logs, and verify matching pilot reuse. Record only this additional local audit in research/evidence.json; source and result IDs are cited directly. Derived aggregates go in research/ or reports/ and do not declare experiment IDs or methods. Preserve all existing artifacts and failures. Do not rerun training for formatting or regenerate historical pilot evidence.

Sources: src-nanda-process, src-bootstrap, src-torch-adamw, src-lones-pitfalls; links and previous source research are in research/sources.json and reports/research.md. The source refresh before execution reconfirmed paired resampling, percentile-versus-BCa semantics and the decoupled AdamW update using primary documentation; no external code is copied and no dependency upgrade or protocol change is introduced.
