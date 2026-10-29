# logistic_s0

Method: logistic. Analysis: exploratory.

| Experiment | Method | Seed | Split | Accuracy | Seconds | Device |
|---|---|---:|---|---:|---:|---|
| logistic_s0 | logistic | 0 | test | 0.955556 | 0.109 | cpu |

## Method, provenance, and learning

Multinomial logistic regression: C=1.0, max_iter=1000. Neural architecture, AdamW, batches and epochs do not apply. See the [scikit-learn estimator](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Real-label models should outperform the label-shuffled control; the primary decay effect may be positive, negative, or inconclusive.

One random example split without writer IDs; conditional intervals exclude training-population uncertainty.

Protocol SHA256: `b5047a8549930ea1abedc111de1fca3f05034826805d4645473912bb5365456b`. Code SHA256: `a16454eb60949757d367bb81811a539d026eefa940575476e732e4c73cfe9358`.
