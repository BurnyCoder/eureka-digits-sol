# Research review: fixed digits engineering replication

Analysis kind: **exploratory**. Current phase: research. No training, pilot scoring or comparative evaluation was run in this phase. Source IDs resolve to [the source ledger](../research/sources.json); local inspection resolves to `audit-fixture-source` in [the evidence ledger](../research/evidence.json).

## Objective and decision

The project goal is verified execution with reproducible artifacts, regardless of the sign of the scientific effect. The scientific question is whether the registered AdamW decay intervention improves held-out accuracy for the supplied small MLP. The subgoal here is to ground that comparison in prior work and expose interpretation risks before fitting. The immediate objective is a primary-source review and a static inspection of [study.py](../study.py) against the unchanged [protocol](../research/protocol.json).

Choose the supplied study rather than another architecture or benchmark: it directly tests the engineering contract with low switching cost. Expected learning is whether the proposed comparison actually isolates decay, what its controls diagnose, and which claims its data can support. Estimated effort for this research pass was approximately a quarter hour; this is a planning estimate, not a measured performance result. Broader searches or new methods would add switching costs without resolving the immediate replication question. No claim of novelty, untouched-data confirmation, model superiority or general autonomous scientific reliability is justified.

## Prior work and reuse

**Optimizer method.** Loshchilov and Hutter distinguish decoupled shrinkage from adding a squared-weight penalty to Adam's loss; adaptive gradient scaling makes these different operations. Their paper and original implementation motivate the intervention, but their image-classification experiments do not establish its sign on this fixed MLP. This is a narrow engineering replication of a supplied study, not a reproduction of the paper's full benchmark results. [src-adamw-paper](https://arxiv.org/abs/1711.05101), [src-adamw-code](https://github.com/loshchil/AdamW-and-SGDW).

In PyTorch notation the update separates shrinkage and the adaptive step:

\[
\theta_t=(1-\gamma\lambda)\theta_{t-1}
-\gamma\widehat m_t/(\sqrt{\widehat v_t}+\epsilon).
\]

The moment estimates use the cross-entropy gradient, without adding the decay term. Both primary arms in `fit_mlp` instantiate `torch.optim.AdamW`; only the registered decay coefficient differs. All parameters passed by `model.parameters()`, including biases, receive decay. Consequently the contrast is decay versus no decay under this implementation, not AdamW versus a different optimizer or decoupled decay versus an L2 penalty. The original Lua code separately applies `weightDecaycurrent` to a saved old parameter vector; its configuration and coefficient convention should not be transplanted numerically. [src-torch-adamw](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html), src-adamw-code; audit-fixture-source.

**Dataset and boundary.** The loader provides a subset of UCI optical digits, not MNIST or the entire original UCI collection. Its documented pixel bound justifies division by a known constant, without fitting a normalizer on held-out examples. The UCI source describes writer-separated original partitions; the fixture instead randomly repartitions the loader's subset. Thus example-held-out accuracy cannot establish unseen-writer generalization. Stratifying by labels does not restore writer separation. [src-digits-loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), [src-uci-digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), [src-splits](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html); audit-fixture-source.

**Baselines and controls.** The no-decay MLP is the matched ablation. Logistic regression checks whether a simpler linear decision rule is competitive; its default solver fits regularized multinomial classification, so it is not a no-regularization control. The class-prior arm always predicts the training majority class; it does not sample from class frequencies. Shuffled training labels break the input-target association while preserving label counts. Poor control performance would be consistent with using real label information, but would not prove every leakage pathway absent. Lones motivates meaningful simple baselines and distinguishes comparisons of fitted instances from general algorithm comparisons. [src-logistic](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [src-lones-pitfalls](https://arxiv.org/html/2108.02497v5); audit-fixture-source.

**Closest existing executions.** The supplied [Haiku repository](https://github.com/BurnyCoder/eureka-digits-haiku) (`src-prior-haiku`) and [Luna repository](https://github.com/BurnyCoder/eureka-digits-luna) (`src-prior-luna`) already expose this study and prior outcomes. Their disclosures distinguish historical execution status from subsequent controller repairs. The Haiku disclosure corrects pilot access language; Luna documents publication changing previously hashed bytes. These are relevant failure modes, not local experimental observations. This review did not independently recompute their scientific scores. Prior exposure remains a reason to label all current scientific conclusions exploratory. [src-haiku-disclosure](https://github.com/BurnyCoder/eureka-digits-haiku/blob/main/reports/post-run-publication.md), [src-luna-disclosure](https://github.com/BurnyCoder/eureka-digits-luna/blob/main/reports/post-run-publication.md).

**Scientific process.** Nanda's sequence separates ideation, exploration, understanding and distillation. Here the problem is supplied; exploration should gain information without upgrading inherited expectations into confirmation. Keep highlights, test alternatives, then compress only supported claims. Writing that uncovers missing evidence can reopen research. [src-nanda-process](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand).

## Frozen design and statistical interpretation

The protocol remains the authority for architecture, normalization, partition, seeds, training duration and hyperparameters. These are inherited engineering choices, not values optimized by this review. Keeping them fixed makes the comparison interpretable and cheap; it does not establish they are optimal. `fit_mlp` resets random generators for each seed and uses a separately seeded generator for minibatch ordering. Matched arms should therefore share initialization and order within the same environment. Deterministic settings reduce variability but do not guarantee cross-device or cross-version identity. [src-reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html); audit-fixture-source.

For seed set \(S\) and test examples \(T\), the primary statistic remains

\[
\widehat\Delta=\frac{1}{|S||T|}\sum_{s\in S}\sum_{i\in T}
\left[\mathbf{1}(\widehat y^{\mathrm{decay}}_{si}=y_i)
-\mathbf{1}(\widehat y^{\mathrm{zero}}_{si}=y_i)\right].
\]

The registered bootstrap resamples test indices jointly across methods and all fitted seeds. The same example must remain paired, consistent with the shared-index definition in [src-bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Its interval conditions on fitted models; repeated test resampling is not repeated training. Show individual seed differences and their sample standard deviation separately. The fixture does not itself implement this later analysis; its existence or correctness has not been verified here. Do not silently substitute SciPy's default BCa interval or resample arms independently.

An interval wholly below zero contradicts the directional benefit prediction under this conditional analysis. One spanning zero is inconclusive. An interval above zero would support only a local exploratory contrast. If correctness differences vanish on every matched example, example resampling must yield a degenerate zero interval; that would not prove equivalence of parameters, probabilities or learning algorithms. These are mathematical implications of the statistic, not observed results.

## Predictions and alternatives recorded before local experiments

| Explanation | Distinguishing prediction | What would weaken or contradict it |
| --- | --- | --- |
| Decay improves this fixed setup | Positive mean matched accuracy difference; inspect every seed effect | A wholly negative registered interval contradicts the directional prediction; a zero-spanning interval leaves it unresolved |
| Decay harms performance or constrains useful fitting | Negative paired difference; possibly higher training loss | Consistently positive effects weaken this local alternative; loss alone does not establish the mechanism |
| Decay changes logits without changing accuracy | Matching correctness vectors can coexist with different weights or confidence | Any nonzero examplewise correctness difference contradicts an exact correctness tie; probabilities require a separately declared audit |
| Apparent gain is training randomness | Seed effects vary in sign or magnitude | Stable paired effects reduce this concern locally, without estimating a training-population effect |
| Leakage, wrong labels or metric error creates useful accuracy | Unexpectedly useful shuffled-label performance, overlapping IDs, or metric recomputation disagreement | Passing these probes narrows these bug classes; it cannot prove universal leakage absence |
| A simple linear rule is sufficient | Logistic accuracy is competitive with the MLP | A clear descriptive MLP advantage weakens this explanation on this split only |

No local accuracy, loss, confidence interval or control outcome is available yet. Future observations must be compared with these predictions rather than written back into them.

## Highlights, anomalies and next-phase handoff

The static inspection found that training indexes only training labels, validation loss is diagnostic, and the fixed final epoch is used without checkpoint selection. During reload checking, however, `fit_mlp` forwards the complete image array through both networks. The pilot saves and scores validation predictions, but test features are accessed. The present architecture has no batch normalization or dropout, and these forward passes are in evaluation mode without gradients; no test-label selection path is apparent in the inspected code. This is source-level evidence, not a dynamic proof of isolation. Preserve the distinction in later reports. audit-fixture-source; src-haiku-disclosure.

The ledger also records a documentation-version anomaly: PyTorch's moving stable URL redirected beyond the project's pinned release family, so the review used versioned optimizer and reproducibility docs. No dependencies or scientific definitions were changed.

Learning from this phase: the important distinctions are intervention versus optimizer comparison, conditional versus training-population uncertainty, and validation-only scoring versus zero test-feature access. These constrain the claims more than an optimistic prior about decay does. [Highlights](highlights.md) retain these issues for later checking.

The next harness phase should register and publish the unchanged prospective local plan while explicitly preserving its exploratory status. Later cheap checks should establish split integrity and artifact reloads before pilot fitting, then recompute metrics from saved paired predictions. Pilot publication remains a controller gate before comparisons. No later-phase command is executed here. If checks reveal a substantive protocol problem, return to research prospectively; preserve failed artifacts and do not retrain merely to repair presentation.

## Licensing and source access

The review reused concepts and installed APIs; it copied no external implementation or lengthy source prose. The original optimizer repository is BSD-3-Clause, the prior-run repositories identify MIT licenses, and UCI identifies CC BY 4.0 for the data. Cite Alpaydin and Kaynak and the dataset DOI when redistributing derived data. Unchecked documentation or prose rights remain `unknown` in the ledger, rather than being inferred from access availability. No license compatibility claim is made for unreviewed copying.

The AdamW HTML rendering timed out; the primary PDF supplied its algorithm. The GitHub directory view failed, but the raw implementation and license were successfully read. These retrieval failures did not require a different model or leave the central method unsupported.
