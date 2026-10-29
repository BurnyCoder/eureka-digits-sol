# Core engineering checks and review

Stage: exploration. Analysis kind: exploratory. This is an engineering report, not a trained-model experiment. Predictions and selection rationale were recorded in [core-plan.md](../research/core-plan.md) before the new tests ran. Actual command outputs, environment information and frozen-file comparisons are preserved as `audit-core-checks` in [core-checks.json](../research/core-checks.json), registered by hash in [evidence.json](../research/evidence.json).

## Predictions compared with observations

| Pre-check prediction | Actual observation | What was learned |
| --- | --- | --- |
| Official data check should pass; the partition should repeat without index overlap | The official check and independent dataset contract test passed | The installed environment can load the fixture data and reproduce its registered partition; this does not establish writer isolation |
| Registered protocol and archived fixture should match live bytes | Both comparisons passed | No scientific implementation or registered configuration edit was needed |
| Stale registration/cache should refuse execution; valid cached records should avoid refitting | Temporary-root refusal and reuse tests passed | These guard branches behave as expected; cached checkpoint contents still need a later hash audit |
| Unpublished comparisons should fail before fitting | Publication-gate test passed | The fixture respects the missing-marker gate; it does not authenticate the contents of an arbitrary marker |
| Pilot score construction should use validation labels and IDs | Synthetic mocked-pilot test passed | Scoring orchestration uses the requested validation indices; actual fitting and full-feature reload behavior remain to be checked in the pilot |
| Untrained checkpoint outputs should reload exactly | Real safetensors round trip on synthetic features passed | Serialization works in this CPU probe; this is not a trained or GPU checkpoint result |
| Logging should preserve long stdout, stderr and failure status | Actual child-process tests and the wrapper's real `check` run passed | The wrapper did not truncate these probes or mask the child failure |

The test-first run initially failed to import the then-unimplemented wrapper. This expected failure is preserved in [core-initial-failure.json](../research/core-initial-failure.json) and embedded in the audit. Adding the small standard-library runner resolved it; the later full suite passed. No scientific fixture fix or training rerun was involved. The audit also records that no result records existed after the checks. `audit-core-checks`.

## One-pass implementation review

Correctness and design: reuse the supplied modular data/network/experiment functions and preserve their bytes, rather than duplicate scientific logic. The wrapper delegates through an argument array, selects only known fixture operations, creates required output directories and returns the process status. Its exclusive log creation preserves existing evidence. Sources: `audit-fixture-source`, `audit-core-checks`, [src-python-subprocess](https://docs.python.org/3.14/library/subprocess.html).

Security and reliability: the wrapper uses no shell and needs no credentials. Logs intentionally retain complete output, so future callers should not print secrets. No new dependencies were introduced. The observed direct research package versions match the existing project pins; this phase does not upgrade a registered scientific environment. The repository's `uv.lock` records transitive resolution; the parent later verified installation with `uv sync --locked --extra cuda`. This corrects the original report's contrary statement; see [parent review](../../../reports/parent-review.md). Package/hardware observations are in the audit, not a claim of bitwise portability. [src-reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html); `audit-core-checks`.

Tests and methodology: non-training probes cover data, provenance, serialization, orchestration and logging; mocks explicitly bypass fitting. These checks cannot determine accuracy, convergence, GPU determinism, the validity of saved trained checkpoints or the size of the decay effect. The fixture cache checks protocol/code hashes but not every checkpoint on reuse. Later phases must inspect actual artifacts and recompute metrics. The random example split and known prior outcomes continue to limit the scientific interpretation. [preregistration.md](preregistration.md); `audit-fixture-source`.

Documentation links resolve locally, and the required core draft verifier returned `complete: true`, `errors: []`, `draft_only: true`. This verifier outcome does not accept the phase or validate a scientific conclusion. The controller remains responsible for publication and state transitions.

## Decision and next learning

Continue to controller core publication. Only the subsequent pilot may test actual fitting, full loss logging, trained checkpoint reloads and validation scoring. The inexpensive checks reduced specific implementation uncertainties while retaining the fixed study and avoiding switching costs. There is no evidence yet to support or falsify the directional decay hypothesis. Keep the exploratory classification and all failures; do not train again merely to repair documentation. The reproduction guide is [docs/reproduction.md](../docs/reproduction.md).
