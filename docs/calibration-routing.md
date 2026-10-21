# Calibration and routing policy (Phase 7)

Phase 7 adds probability calibration, threshold selection, independent validation,
and deterministic shadow route/review decisions. The base models are frozen.
Sufficiency and clarification remain unknown and disabled.

## Reproduce

First prepare the dataset and complete the baseline and Laya development commands
documented in the README. They create the frozen model artifacts used here.

```powershell
uv sync --locked --extra baseline --extra ml
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
uv run --extra baseline --extra ml rightdesk predict-calibration
uv run --extra baseline --extra ml rightdesk calibrate-routing
```

The first command captures predictions for all 1,063 reserved calibration cases.
It checks the completed development run, its artifact checksums, dataset manifest,
workflow, checkpoint/configuration pins, case membership, and saved model identity.
It loads the saved TF-IDF pipeline and training priors and reconstructs the saved
rules. It never calls base-model fit. Only our locally produced and verified joblib
artifact should be loaded; joblib artifacts can execute code when deserialized.
The model receives intake text, without reference labels or later replies.

Laya reuses one local Agent, with the same CPU/batch configuration as Phase 6.
Raw responses are retained. A failure writes `failed_run.json`; only `run.json`
marks a completed capture. No failed cases are silently removed. A missing model
does not trigger a download. An existing output directory is protected.

The second command validates the cache and performs the inexpensive statistical
analysis. It does not rerun inference. Paths can be overridden with `--config`,
`--adapter`, `--calibration`, `--source`, `--predictions`, and `--output`.
Choose a fresh output path for each run. The capture command supports `--config`,
`--adapter`, `--source`, and `--output`.

## Allocation and fitting

`configs/calibration/retail_v1.yaml` fixes the settings before validation:

| Role | Expected fraction | Actual cases | Use |
|---|---:|---:|---|
| Fit | 50% | 544 | Fit one temperature per probability model |
| Selection | 25% | 263 | Choose each frozen confidence threshold |
| Validation | 25% | 256 | Evaluate frozen candidates once |

A seeded SHA256 value of the normalized-text duplicate key determines the role.
This is label-independent and stable across input order. Identical normalized
messages stay together; ID and conversation overlap are checked. Fractions are
probabilities, so actual counts need not match the nominal proportions exactly.
The original train/development/calibration/test assignments stay intact.
No final-test tickets or labels are loaded by Phase 7.

For each of majority, TF-IDF, and Laya, temperature scaling applies:

```text
calibrated_p = softmax(log(max(raw_p, 1e-12)) / T)
```

Fit minimizes multiclass negative log likelihood on fit cases, with T in
[0.05, 20]. Identity and endpoints are considered alongside bounded scalar
optimization, so fit NLL cannot worsen relative to identity. The calibrator saves
the model/workflow identity, queue vocabulary, fit fingerprint, temperature, and
before/after fit NLL. Applying it rejects model/workflow mismatches and double
calibration. It preserves the selected queue, including explicit tied choices.

This is probability-based temperature scaling; it uses the SDK's rounded queue
probabilities rather than unavailable pre-softmax Laya logits. Validation can
show worse probability quality even when fit NLL improves. Calibration cannot
correct the base classifier's queue choices.

## Thresholds and independent validation

Seven variants are declared in advance: raw and temperature-scaled versions of
each probability model, plus native rules. The primary research target is 5%
observed ticket routing error; it is a configurable experiment setting rather
than an agreed production service level.

On selection cases, each probability variant chooses the threshold with the
largest ticket coverage satisfying the target and at least 30 distinct routed
text groups. A score exactly equal to the threshold routes. Equal scores stay
together. If no candidate qualifies, the threshold is null and the policy reviews
everything. Rules have no probabilities; their fixed native operating point is
checked against the same target and group minimum.

Every candidate is frozen before validation labels are used. Validation reports
ticket coverage, errors, per-queue metrics, reliability bins, Brier score, ECE,
NLL, and descriptive cluster-bootstrap intervals. Curves on validation are
descriptive; they do not choose replacement thresholds. Group bounds shown on
selection/search curves are not valid post-selection guarantees.

The independent gate also computes a one-sided Clopper-Pearson upper bound on
**group any-error probability**. A routed normalized-text group is a failure if
any routed ticket in it is wrong. This avoids treating duplicate tickets as
independent Bernoulli trials. The bound uses independent, exchangeable groups as
an assumption; it is not a bound on ticket-weighted error when group sizes vary.
For all seven variants, alpha is 0.05/7 (Bonferroni correction).

A policy is enabled for shadow proposals only if its selection candidate qualifies,
validation observed ticket error is within 5%, at least 30 validation groups route,
and the corrected group bound is within 5%. A failed gate leaves a review-only
policy. A null error rate means no cases routed, not zero risk. Even zero observed
errors leave a positive binomial upper bound. This conservative gate can reject
useful candidates when the validation sample is small.

Labels remain provisional eventual-owner references. Passing this research gate
does not validate intake sufficiency, label quality, future distribution changes,
or production safety. A failed gate is an experiment result, not an implementation
failure; do not relax settings after seeing validation and call that independent
confirmation. More validation evidence requires newly reserved/reviewed data.

## Policy and artifacts

`RoutingPolicy` binds an exact model revision and calibration version to a validated
workflow. It proposes route when an enabled model meets the frozen threshold, or
when an enabled native rule matches. Other cases propose review. Unsupported
languages and missing predictions produce explicit review reasons; case, workflow,
model, and calibration mismatches are rejected. The policy never executes routing.
Clarification and sufficiency thresholds are rejected by this Phase 7 policy.

Generated workflow snapshots have distinct policy versions. The original workflow
file is preserved because the dataset and historical runs bind its checksum.
Each variant saves `policy.json` containing the workflow and settings. Reconstruct
it with `WorkflowConfig.model_validate`, `RoutingSettings.model_validate`, and
`RoutingPolicy(workflow, settings)`. Calibrated predictions must first pass through
the saved `TemperatureCalibrator`; raw predictions cannot use a calibrated policy.

`artifacts/calibration-predictions-v1/` contains raw predictions and Laya responses.
`artifacts/calibration-v1/` contains the report, fit calibrators, role assignments,
selection curves, validation predictions and metrics, policy snapshots, and shadow
decisions. Its completion manifest records code/configuration/package versions,
provenance, and artifact hashes. Failed analysis uses temporary staging and does
not publish a completed output directory. Latency is not measured by cached
analysis; Phase 6 retains the separately labeled inference timing measurements.

## Completed local result

All four models completed 1,063 calibration predictions with zero failures.
The independent validation role contains 256 tickets. The frozen TF-IDF candidates
(raw threshold 0.352236; temperature threshold 0.455396, rounded here) both route
244/256 tickets: 95.31% candidate coverage with 9 errors, or 3.69% ticket error.
Their corrected group upper bound is 7.90%, exceeding the 5% gate. Therefore both
saved policies are disabled and propose review for all tickets. Full precision
thresholds are retained in the policy artifacts.

Laya and majority have no selection threshold satisfying the 5% target and group
minimum. Native rules also fail selection; their diagnostic validation operating
point is 53.12% coverage with 8.82% ticket error. All seven saved policies remain
review-only. Candidate measurements are useful research results, not enabled
automation coverage. More independent validation evidence or a separately agreed
research target is needed before enabling proposals; this run's gate is not relaxed.

Calibration preserves validation accuracy: TF-IDF 92.97%, Laya 57.81%.
These are a different partition from the Phase 6 development results.

| Model | Brier, raw → calibrated | ECE, raw → calibrated | NLL, raw → calibrated |
|---|---:|---:|---:|
| TF-IDF | 0.1383 → 0.0937 | 0.1577 → 0.0566 | 0.3507 → 0.2255 |
| Laya | 0.5372 → 0.5182 | 0.1331 → 0.1315 | 1.2171 → 1.1277 |

All 126 automated tests pass. The completed capture, analysis, prediction and
decision schemas, historical run checksums, and current code/artifact hashes are
verified. No final-test evaluation was performed.

## Verification and references

Tests exercise temperature fitting/serialization and identity checks, zero and tied
probabilities, group allocation, threshold ties and all-review outcomes, sparse
zero-error bounds, policy eligibility and failure paths, immutable model loading,
cache integrity, and a complete pipeline without final-test files. Perturbing only
validation labels must leave fitted calibrators and selected thresholds identical.

- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [SciPy bounded scalar optimization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html)
- [SciPy exact binomial confidence intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)
