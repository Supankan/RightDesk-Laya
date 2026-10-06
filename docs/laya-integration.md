# Laya integration (Phase 6)

## Pinned configuration

`configs/models/laya_v1.yaml` selects the English base checkpoint
`convaiinnovations/laya` at revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. The English corpus makes this a
direct English-checkpoint experiment. No automatic model routing or selection
among specialist checkpoints is performed.

The checkpoint's encoder configuration and tokenizer are bundled at the same
revision. The explicit download command fetches those assets, the safetensors
weights, agent configuration, and model card. It excludes repository Python
files. A local manifest records the revision, source, license, and checksums.
The runtime uses the installed Laya SDK and verifies these files before loading.

The current execution configuration is CPU, four CPU threads, batch size four,
512 total tokens, and a 192-token question head. Fast kernels and compilation
are disabled. The process reuses one Agent instance across all batches.

## Commands

```powershell
uv sync --locked --extra baseline --extra ml
uv run --extra baseline --extra ml rightdesk download-laya
uv run --extra baseline --extra ml rightdesk benchmark-laya
```

Downloads happen only in the explicit download step. The benchmark requires a
verified local checkpoint. For a network-disabled run:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
uv run --extra baseline --extra ml rightdesk benchmark-laya
```

Use `--output artifacts/laya-another-run` for a fresh run; existing output
directories are protected. `--baseline` selects the prior baseline run,
`--adapter` the pinned model YAML, and `--config` the experiment YAML.
The experiment and adapter checkpoint identities must match.

## Prediction contract

The adapter supports the shared `predict_one(ticket)` interface and
`predict_batch(tickets)`. Its `fit(tickets, targets)` compatibility method only
checks the inputs and loads the checkpoint; it does not train. The Laya benchmark
calls `load()` directly and never loads the training examples or labels.

The only question is `owner_queue`, a choice among the workflow's six queue
descriptions. The state contains ticket text only. Case IDs, eventual-owner labels,
scenario details, and later replies are not sent to the model.

Returned probabilities must cover all six queues, be finite and bounded, and
agree with the returned choice. SDK probabilities have four decimal places, so
the adapter allows a sum discrepancy of at most `k * 0.00005 + 1e-6` before
normalizing. Anything larger fails. Raw SDK output is preserved separately so this
conversion is auditable. The SDK's entropy-style `confidence` and action score are
not used as routing probabilities or sufficiency judgments.

The SDK applies checkpoint-provided temperatures. These have not been fitted or
validated on RightDesk, so `calibration_version` remains null. The installed SDK
warns that it clamps the checkpoint's `choice:11+` temperature; this experiment
has six choices and uses the separate `choice:6-10` bucket.

Sufficiency and suggested-question fields remain null. Phase 6 measures queue
prediction; clarification and reviewed sufficiency evaluation remain later work.

## Failure and context handling

The adapter rejects explicitly non-English tickets and text exceeding a conservative
312-token budget, reserving room for the question head and special tokens. It
does not silently truncate long inputs or substitute another checkpoint. Empty
batches return an empty list; wrong output counts, missing queues, inconsistent
choices, and invalid probabilities raise LayaError.

A failed load leaves no completed report. Failures after benchmark output creation
write `failed_run.json` and preserve any partial responses. `run.json` is written
only after all requested cases have completed. No failed cases are dropped from
the denominator of a purported successful benchmark, and no fallback predictions
are fabricated. Consumers should require `run.json` as the completion record.

## Comparison and artifacts

The runner checks the baseline artifact hashes, dataset manifest, workflow, and
ordered case IDs. It evaluates exactly the baseline's requested split and uses
the same metrics and bootstrap seed/sample count. The development comparison
does not evaluate the calibration or final-test partitions.

`artifacts/laya-development-v1/` contains a combined report and copies of the
original baseline artifacts. `laya/` contains raw responses, validated predictions,
metrics, configuration/model metadata, and batch timings. `run.json` preserves
baseline provenance alongside the Laya revision, SDK/runtime versions, code
hashes, and artifact checksums.

Laya latency is measured per batch and divided by that batch's size. Its reported
percentiles describe amortized time per ticket, not individual request response
time. Baseline latencies are serial single-ticket timings; they are not directly
comparable. Model loading and one warm-up call are excluded from inference latency.

## Development result

The real CPU run completed all 863 development cases with zero failures.

| Model | Accuracy | Macro F1 | Coverage | Routing error | Brier |
|---|---:|---:|---:|---:|---:|
| Zero-shot Laya | 53.30% | 0.5152 | 100% | 46.70% (403/863) | 0.6010 |
| Trained TF-IDF + logistic regression | 89.57% | 0.9030 | 100% | 10.43% (90/863) | 0.1782 |

This configuration does not demonstrate a Laya advantage. The comparison uses
provisional eventual-owner targets and different training regimes: Laya uses
queue descriptions, while TF-IDF learns from 5,792 training examples. It does not
establish the performance of a fine-tuned Laya model or human-reviewed intake
decisions. Calibration and final-test partitions remain unevaluated.

All artifact and code hashes, checkpoint hashes, prediction contracts, and ordered
case IDs were verified after the run. The unit/integration suite passes 107 tests.

## Automated checks

Unit tests use an injected fake backend to check batch ordering, state boundaries,
probability rounding, malformed output, unsupported input, and model reuse.
Integration tests check exact baseline-case comparison and incomplete-run failure
records. The full real-checkpoint development run is separate from the test suite.
