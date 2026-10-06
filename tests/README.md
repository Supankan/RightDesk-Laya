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

Later phases will add checks for:

- Valid schemas and consistent route/clarify/review records.
- No overlap of conversations or duplicates across dataset splits.
- No future dialogue or ground-truth labels in model inputs.
- Common adapter output contracts and probability validation.
- Evaluation metrics on small, independently checked examples.
- Shadow decisions remaining separate from action execution.

Only implemented behavior is tested; no models or datasets are loaded by this suite.
