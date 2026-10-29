# Pilot plan recorded before fitting

Stage: exploration. Analysis kind: exploratory. The project goal is verified, reproducible execution of the supplied study, regardless of the decay contrast's eventual sign. The subgoal is the first trained artifact for controller publication. The immediate objective is to run only `regularized` seed `0` on the fixed schedule, save its validation predictions, complete loss history and checkpoint, and inspect their integrity.

The inherited [preregistration](../reports/preregistration.md) remains authoritative. Prior outcomes are known; this is not untouched-data confirmation. Neither the registered [protocol](protocol.json) nor [fixture](../study.py) will change. Core publication is recorded in [core-publication.json](core-publication.json). No comparative or sensitivity fitting is authorized in this phase.

## Predictions and alternatives

| Prediction before fitting | Alternative and distinguishing observation |
| --- | --- |
| Real-label training should lower final training cross-entropy relative to its initial logged epoch; validation loss should also improve overall, without requiring monotonicity. | Flat, rising or non-finite losses suggest ineffective fitting, numerical failure or implementation error. Rising late validation loss alongside falling training loss suggests possible overfitting, not a proven mechanism. |
| Saved predictions and labels should reproduce the saved validation accuracy exactly, and labels/IDs should match the frozen partition. | Metric arithmetic errors, stale data or wrong split selection produce mismatches. Disjoint IDs do not establish writer isolation or eliminate all leakage. |
| The trained safetensors checkpoint should reload exactly in the fixture, and an independent validation-only reload should reproduce all saved validation predictions. | Corruption, incompatible configuration or stale provenance produces hash or prediction mismatches. A passing reload does not certify cross-platform identity. |
| Only the single pilot result should exist after fitting; scientific code/protocol hashes should remain unchanged. | Additional result files, changed hashes or a failure log would invalidate the intended scoped execution and require investigation. Preserve all such evidence. |

No numerical accuracy target or weight-decay benefit is predicted here. Above-chance accuracy by itself cannot eliminate leakage, and this pilot has no trained controls. The inherited primary contrast and seed sensitivity remain untested.

## Selection, effort and execution

Use `run_study.py pilot`, whose argv delegates to the supplied `study.py pilot` while preserving complete timestamped console output. Run the canonical `study.py check` first. Select the supplied CUDA device if available, otherwise preserve its CPU fallback. Do not tune the architecture, schedule, optimizer or data. These settings are inherited engineering choices, not locally optimized values.

Expected learning per unit time is high: one small fit jointly probes optimizer execution, loss logging, actual validation scoring and trained serialization. Estimate less than ten minutes for fitting and inspection, and roughly another ten for audit/report verification; these are planning estimates, not timings. Switching to another model, architecture or data would change the question and consume later-phase budget.

After fitting, inspect the full log, recompute accuracy, verify partition coverage/labels/hashes and reload validation predictions without optimizer steps. Preserve effective configuration and package/device provenance in an auxiliary audit, with a unique evidence ID. Generate report numbers and a loss figure directly from the saved result; the audit and summaries are not additional experiments. Do not score test predictions or run the comparison command. If valid results already exist, reuse them rather than train again for presentation repairs.

## Sources and interpretation

The source research is in [research.md](../reports/research.md). Rechecked the primary [digits API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) (`src-digits-loader`) and [AdamW API](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html) (`src-torch-adamw`) before this run. Fixed division by the documented pixel bound estimates no preprocessing statistics. Decay acts separately from Adam's gradient moments; no optimizer comparison is being made. No external implementation is copied.

Follow [src-nanda-process](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand): keep this inexpensive probe exploratory, compare observations to these predictions, retain anomalies, and distill only supported claims. The fixture forwards test features during reload verification, but scores validation labels only. Its random example partition cannot support unseen-writer generalization. Seed control does not guarantee identity across environments: [src-reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html). Source IDs resolve in [sources.json](sources.json).
