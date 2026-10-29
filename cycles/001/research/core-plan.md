# Core engineering probe plan

Stage and analysis kind: exploration / exploratory. No training is authorized in this phase.

Project goal: complete the fixed engineering replication with reproducible artifacts. Subgoal: make the supplied implementation runnable and reviewable before controller publication. Immediate objective: preserve the fixture and registration, add non-training regression tests and a console-log wrapper, and document reproduction. Expected effort: approximately fifteen active minutes; inexpensive CPU checks avoid training and protocol switching costs.

Predictions recorded before running the new suite:

- The official `study.py check` should pass; repeated dataset construction should give identical, disjoint, exhaustive indices and constant-scaled pixels.
- The archived fixture and registered protocol hashes should match their live files. A deliberately stale cached result or registration should raise before fitting.
- Without a pilot-publication marker, comparisons should refuse execution before fitting. A mocked pilot should save only validation labels and IDs, even though the real checkpoint verifier forwards all feature rows.
- An untrained network should serialize and reload with exactly equal outputs on synthetic inputs. This probes serialization, not predictive accuracy or training determinism.
- The new wrapper should retain a long output line and stderr in its log and propagate failure status. Missing wrapper implementation should initially fail its tests.

Competing explanations: an environment mismatch could break imports; a split/provenance bug could fail invariants; synthetic mocks could hide defects in actual fitting; a logging implementation could truncate output or mask a child failure. Passing these probes narrows these particular failure classes, without establishing scientific validity or absence of all leakage. The later pilot remains necessary to test actual training and checkpoint reuse. No observed model outcomes will be written into this plan.

Selection rationale: reuse the supplied implementation and standard-library `unittest`/`subprocess`, rather than refactoring scientific code. Sources: `audit-fixture-source`, `src-splits`, `src-torch-adamw`, `src-reproducibility`, and the Python documentation linked in test and runner comments. Record actual checks and limitations separately in `core-checks.json` and the reproduction guide.
