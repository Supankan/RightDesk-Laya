# Baseline models and evaluation (Phases 4-5)

## Running the benchmark

```powershell
uv sync --locked --extra baseline
uv run --extra baseline rightdesk benchmark-baselines --config configs/experiments/baseline.yaml --output artifacts/baseline-development-v1
uv run --extra baseline pytest -q
```

The default run has already completed. Pick a fresh output directory to repeat
it; existing results are never overwritten. The command validates partition
checksums, provenance, workflow/mapping identity, target joins, and train/evaluation
case, conversation, and duplicate-message separation before fitting anything.

Only training and the requested evaluation role are loaded. Development is the
default. Calibration is rejected as a benchmark role. An explicit `--split test`
is supported for a later frozen comparison; it has not been run. This baseline-only
command records Laya as deferred. Phase 6's separate `benchmark-laya` command
compares the pinned checkpoint with these saved baseline artifacts.

This focused benchmark writes local artifacts. General experiment orchestration,
SQLite tracking, model serving, and calibrated policy decisions arrive later.

## Implemented models

All adapters implement `fit(tickets, targets)` and `predict_one(ticket)` and
return validated ModelPrediction records. Prediction takes TicketInput only;
IDs are for alignment and text is the sole classifier feature.

- **Majority:** fits training class frequencies, always predicts the most common
  queue, and reports those empirical priors. Ties use alphabetical queue order.
- **Rules:** versioned phrases in `configs/models/rules_v1.yaml`. Case-insensitive
  whole-word/phrase matching; exactly one matched queue routes, conflicting queues
  or no match abstain. There are no estimated probabilities. These fixed rules
  do not reason about negation or context and have not been tuned to these results.
- **TF-IDF/logistic regression:** word unigrams/bigrams, lowercase, sublinear TF,
  min_df=1, max_features=30,000; logistic regression C=1, lbfgs, max_iter=1,000.
  Vocabulary, IDF, and weights are fitted on training data only. Seed is 42.
  A convergence warning fails the fit. Absent training classes get probability
  zero in the full workflow distribution; at least two observed classes are required.

No model estimates intake sufficiency or useful questions. Those fields remain
unknown. Probability outputs are uncalibrated; no routing threshold is selected.

## Metric definitions

| Metric | Definition |
|---|---|
| Accuracy | Correct queue predictions / all cases; abstention counts as incorrect |
| Macro F1 | Mean per-queue F1 across all configured queues, with zero for undefined values |
| Precision / recall | Per-queue true positives / predicted count or reference support |
| Confusion matrix | Reference queues in rows; predicted queues plus review column |
| Automation coverage | Non-abstained cases / all cases at the stated selection rule |
| Routing error | Incorrect routed cases / routed cases; null when nothing routes |
| Brier score | Mean sum of squared probability errors across queues, range 0-2 |
| Calibration error | Top-label ECE using 10 equal-width confidence bins |
| Reliability bins | Count, mean confidence, and empirical accuracy per bin |
| Latency p50/p95 | Local single-ticket adapter call, including contract validation |

Rules have no Brier score, ECE, or probability-threshold curve. Their native
route/review point is reported alongside the all-review point. For probabilistic
models, the full curve uses every observed maximum-probability threshold, includes
the all-review point, and never divides a group of tied confidence values.
The Markdown report shows fixed diagnostic cutoffs 0.50, 0.70, 0.90, and 0.95;
these examples are not recommended operating policies.

Timing excludes fitting and file writing, includes one unmeasured warm-up call,
and limits BLAS to one thread. Timing is machine-dependent, not a throughput claim.

## Uncertainty

The evaluator reports 95% percentile cluster-bootstrap intervals for overall
accuracy, native coverage/routing error, and routing error along the curve.
It resamples normalized-message groups with replacement (500 replicates, seed 42)
and keeps all cases in a selected group together. Rate denominators are recomputed
for each replicate. Replicates with no routed cases are excluded from routing-error
intervals; their valid count is recorded. Fewer than two groups produces no interval.

These are descriptive, pointwise intervals. They are not simultaneous confidence
bands or evidence that a selected threshold meets a production error limit.
Sparse samples and zero observed errors can give degenerate bootstrap intervals;
always inspect routed counts and error counts. Hyperparameter-selection uncertainty,
label uncertainty, and future domain shifts are not covered.

## Initial development results

Training: 5,792 cases. Development: 863 cases across 859 normalized-message groups.
These references are provisional mapped scenario labels; they are not reviewed
judgments about the information present at intake.

| Model | Accuracy | Macro F1 | Coverage | Routed errors / routed cases |
|---|---:|---:|---:|---:|
| Majority | 23.06% | 0.0625 | 100.00% | 664 / 863 |
| Rules | 44.61% | 0.5944 | 48.67% | 35 / 420 |
| TF-IDF/logistic | 89.57% | 0.9030 | 100.00% | 90 / 863 |

Logistic regression is a strong baseline on this development set. Comparing its
full-coverage error directly against the selective rules router would compare
different operating points. The saved curve supports future matched-risk or
matched-coverage analysis after calibration. At the diagnostic 0.70 score cutoff,
it routes 588 cases with 5 errors; this is not an independently validated threshold.
Laya will need to demonstrate a useful advantage over this conventional classifier.

## Saved artifacts

`artifacts/baseline-development-v1/` contains:

- `report.md`: comparison, per-queue metrics, confusion matrices, and cutoff examples.
- `run.json`: configuration, reference semantics, split, dataset/code hashes,
  package/lockfile versions, timing method, metrics, and artifact checksums.
- Per-model `predictions.jsonl`: ModelPrediction nested under `prediction`, plus
  observed latency; labels remain separate in the prepared dataset.
- Per-model `metrics.json`, `risk_coverage.csv`, and `model.json`.
- `tfidf_logistic/pipeline.joblib`: the locally trained sklearn pipeline. It can
  be loaded with joblib for subsequent local experiments; only load trusted files.

All baseline tests use synthetic fixtures and small hand-checked calculations.
The end-to-end test creates no calibration/test files, ensuring a development run
does not depend on either partition. Model and metric tests require the baseline
extra; with only core dependencies, those modules are explicitly skipped.
