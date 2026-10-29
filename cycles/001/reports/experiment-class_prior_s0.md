# class_prior_s0

Method: class_prior. Analysis: exploratory.

| Experiment | Method | Seed | Split | Accuracy | Seconds | Device |
|---|---|---:|---|---:|---:|---|
| class_prior_s0 | class_prior | 0 | test | 0.102778 | 0.027 | cpu |

## Method, provenance, and learning

Class-prior baseline fitted on training labels. Neural architecture, optimizer, batches and epochs do not apply. The committed study source defines prediction behavior.

Real-label models should outperform the label-shuffled control; the primary decay effect may be positive, negative, or inconclusive.

One random example split without writer IDs; conditional intervals exclude training-population uncertainty.

Protocol SHA256: `b5047a8549930ea1abedc111de1fca3f05034826805d4645473912bb5365456b`. Code SHA256: `a16454eb60949757d367bb81811a539d026eefa940575476e732e4c73cfe9358`.
