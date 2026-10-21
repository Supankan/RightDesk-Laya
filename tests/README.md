# Verification

Phase 1 tests cover strict input validation, workflow references, probability
constraints, action combinations, unknown versus negative labels, YAML errors,
and CLI operation from a different working directory. Phase 2 adds synthetic
ABCD fixtures for first-message extraction, label separation, split preservation,
checksums, rejected cases, interrupted downloads, and overwrite protection.
Run `uv run pytest -q`. Tests require no network or downloaded dataset.

Phase 3 covers mapping exceptions, unknown labels, duplicate normalization,
test/dev precedence, seeded reproducibility, independence from labels and input
ordering, checksum failures, joins, and separate prediction/label artifacts.

Phases 4-5 add model behavior, train-only vocabulary checks, hand-calculated
metrics, confidence ties, clustered uncertainty, abstention handling, and a
complete synthetic benchmark with no calibration/test files. Install the
`baseline` extra to run the full suite: `uv run --extra baseline pytest -q`.
Without it, model/evaluation test modules are explicitly skipped.

Phase 6 adds Laya adapter contract tests and comparison/failure-path tests using
fake model responses. The suite does not download or load model weights. Install
both extras for the full suite: `uv run --extra baseline --extra ml pytest -q`.
The real-checkpoint development benchmark is a separate explicit command.

Implemented checks cover:

- Valid schemas and consistent route/clarify/review records.
- No overlap of conversations or duplicates across dataset splits.
- No future dialogue or ground-truth labels in model inputs.
- Common adapter output contracts and probability validation.
- Evaluation metrics on small, independently checked examples.
- Shadow decisions remaining separate from action execution.

Only implemented behavior is tested; no models or datasets are loaded by this suite.
## Phase 7 checks

`test_calibration.py`, `test_routing_policy.py`, and `test_calibration_run.py` check
group isolation, temperature transforms, threshold selection, independent validation,
frozen base models, artifact integrity, and route/review proposals. The integration
fixture has no final-test files and uses a fake Laya backend. Real calibration
inference is run explicitly with `rightdesk predict-calibration`.

## Phases 8-10 checks

`test_workbench.py` covers the SQLite catalog, checksummed inputs, idempotence,
source loss, Parquet exports, annotation validation/history, concurrent conflicts,
review queues, and agreement. Synthetic source bundles require no model inference.
`test_dashboard.py` uses Streamlit AppTest for empty states, comparisons, filters,
blind review, explicit judgments, next-case navigation, reveal, and stale forms.
All annotation writes in these tests go to temporary synthetic databases.

For the complete suite:

```powershell
uv sync --locked --extra baseline --extra ml --extra data --extra ui
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests apps
```

## Phases 11-12 checks

`test_clarification.py` covers bounded ambiguity, unsupported/tied inputs, question
history, route gate preservation, null usefulness, and review snapshots.
`test_simulator.py` covers persistent state, replies, virtual deadlines, agent
actions, retries, concurrent delivery, append-only streams, event replay/corruption,
synthetic lifecycle assertions, and input-only development replay.
Dashboard tests also exercise primary intake, clarification, service completion,
reopening, and stale-clock protection. Simulation databases and test reviews stay
inside temporary test directories.

## Phases 13-14 checks

`test_shadow.py` checks the read-only source handle, denied projection reads,
prefix-only input, future-reply exclusion, independent state, live arrivals,
at-most-one question proposal, atomic restart recovery, concurrent worker delivery,
immutable proposals, model-error review and changed event history. `test_final.py`
checks protocol/output overwrite protection and failure before test access if pins change.
Saved final-result and shadow UI states are exercised without inference.

`scripts/run_e2e.py` is an explicit real-model integration run, outside the fixture
suite. It starts and restarts actual worker processes and checks that the primary
state is unchanged, all events are observed, no enabled routes occur, no model
failures occur, and a third restart produces zero new proposals.

CI installs locked baseline/data/UI extras and runs fixture tests plus Ruff on Linux;
no model weights or dataset downloads are required. A local package build is also
checked at handoff. Real inference is an explicit separate operation.
