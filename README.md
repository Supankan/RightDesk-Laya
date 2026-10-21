# RightDesk

A local workbench for evaluating whether an incoming support request should be
routed, clarified with a predefined question, or sent to a person for review.

## Status

Phases 1-14 implemented: validated contracts, reproducible ABCD preparation, provisional
queue mapping, isolated partitions, three working baselines, and a saved development
benchmark. The original import has 10,042 cases; the partitioned dataset has 8,722
active cases and 1,320 documented exclusions. The pinned Laya adapter supports
local batched inference and comparison with the saved baselines. Phase 7 adds
probability calibration, independent threshold validation, and deterministic shadow
route/review proposals. Phases 8-10 add the SQLite experiment catalog, Parquet exports,
working Streamlit dashboard, and revisioned human-review workflow. Phases 11-12 add
bounded clarification proposals and a persistent primary helpdesk simulator with
customer/agent actions and virtual time. Phases 13-14 add the independent shadow worker,
restartable observation, real-model end-to-end checks, a frozen final-test evaluation,
saved-result dashboard tabs, CI checks, and a reproducible handoff.

This is a completed **local research prototype**. Automatic routing stays disabled
because no candidate passed the calibration gate. Human review and question
usefulness remain unmeasured. See [results](docs/results.md) and [handoff](docs/handoff.md).

## Layout

```text
apps/dashboard.py              Experiment explorer and blind human review
configs/workflows/             Queue and question definitions
configs/experiments/           Experiment templates
src/rightdesk/schemas.py       Validated data contracts
src/rightdesk/config.py        YAML loading and experiment configuration
src/rightdesk/data/            Import, preprocessing, and splits
src/rightdesk/models/          Replaceable prediction adapters
src/rightdesk/policy/          Routing and clarification policies
src/rightdesk/evaluation/      Calibration, metrics, and reports
src/rightdesk/storage.py       SQLite catalog and append-only annotations
src/rightdesk/services.py      Shared application operations
src/rightdesk/review.py        Review queues, judgments, and reviewer agreement
src/rightdesk/exports.py       Checksummed Parquet snapshots
src/rightdesk/simulation/      Primary helpdesk, event stream, and scenario drivers
src/rightdesk/shadow/          Read-only observer and independent proposal database
scripts/run_e2e.py             Real-model process restart and isolation demonstration
src/rightdesk/cli.py           Configuration and data preparation CLI
data/                         Local datasets and source manifests
artifacts/                    Local predictions, models, and reports
docs/                         Architecture, schemas, and data plan
tests/                        Contracts, configuration, and ingestion tests
```

## Stack

- Python 3.12 and uv for the project environment.
- Pydantic for contracts, YAML for workflow and experiment configuration,
  and Typer for CLI commands.
- Laya and scikit-learn for model comparisons.
- NumPy, pandas, and Parquet through PyArrow for experiment data.
- SQLite for run metadata and reviewer annotations.
- Streamlit and Plotly for the local review dashboard.
- pytest and Ruff for development checks.

Optional extras separate the baseline, ML, data, and UI dependencies. `uv.lock`
records resolved versions. Core, development, and baseline dependencies have been
installed and tested. The baseline extra adds NumPy/scikit-learn without loading
Laya or PyTorch. The ML extra is installed for Phase 6's CPU Laya integration;
data/UI extras are installed and verified for Phases 8-10.

## Local setup and checks

```powershell
uv sync --locked --extra baseline
uv run rightdesk --help
uv run rightdesk validate-config configs/workflows/retail_support.yaml
uv run rightdesk validate-config configs/experiments/baseline.yaml --experiment
uv run --extra baseline pytest -q
uv run ruff check src tests apps
```

On this Windows workspace, uv is also available as `.\.bootstrap\Scripts\uv.exe`.
The installed environment can be used directly without uv:

```powershell
.\.venv\Scripts\rightdesk.exe validate-config configs/experiments/baseline.yaml --experiment
.\.venv\Scripts\python.exe -m pytest -q
```

Run these commands from the RightDesk directory. Paths inside YAML are relative
to that YAML's directory, so loading an absolute config path works from any CWD.
Validation checks the template structure and linked workflow; it does not assert
that a dataset, trained model, or calibrated routing policy is ready to execute.

To prepare and open the working dashboard after generating the experiments:

```powershell
uv sync --locked --extra baseline --extra ml --extra data --extra ui
uv run --no-sync rightdesk store bootstrap
uv run --no-sync streamlit run apps/dashboard.py
```

## ABCD ingestion

The download is an explicit operation, pinned to a full repository commit.
Import runs locally and verifies the archive and license checksums first:

```powershell
uv run rightdesk data download-abcd
uv run rightdesk data import-abcd
```

On this workspace these steps have already completed. Repeating the download
verifies and reuses its files. Import refuses to overwrite an existing output;
use `--output-dir` and `--manifest` for a separately named preparation.

Outputs under `data/processed/abcd-intake-v1/`:

- `tickets.jsonl`: TicketInput records containing only the first customer message.
- `references.jsonl`: source flow/subflow labels, IDs, turn indexes, and official splits.
- `rejections.jsonl`: preparation failures with source locations and reason codes.

Source and preparation manifests live under `data/manifests/`. Raw and processed
data, generated manifests, model weights, and experiment artifacts remain local
and are ignored by Git. A fresh checkout should run the download, import, and
partition commands below to generate its own files. Ingestion uses the standard library and Phase 1
dependencies; optional `data` and `ml` extras are not needed yet. No model runs.

See [data preparation details](docs/data-plan.md) for provenance, limitations,
and commands for importing into another directory.

## Queue mapping and partitions

```powershell
uv run rightdesk data prepare-abcd --config configs/data/abcd_partitions_v1.yaml
```

This has already run locally. Repeating it refuses to overwrite outputs; create
a new partition config with fresh output/manifest paths for a separate run.
The baseline experiment now points to this prepared dataset.

| Partition | Cases |
|---|---:|
| Training | 5,792 |
| Development | 863 |
| Calibration | 1,063 |
| Test | 1,004 |

Each partition contains `tickets.jsonl` and a separate `labels.jsonl`. The
top-level `assignments.jsonl` records all source cases, duplicate groups, targets,
and exclusions. `report.json` reports counts and integrity checks. The manifest
is `data/manifests/abcd-partitions-v1.json`.

Mapping choices are explicit in `configs/mappings/abcd_retail_v1.yaml` and marked
**provisional**. Source-derived target queues do not establish what was inferable
at intake. Every prepared label remains unreviewed with unknown intake judgments.

## Baseline benchmark (Phases 4-5)

```powershell
uv run --extra baseline rightdesk benchmark-baselines
```

Already run locally: `artifacts/baseline-development-v1/report.md` contains the
comparison. Each model has predictions, metrics, a risk/coverage CSV, and model
metadata. The fitted sklearn pipeline is saved as a local joblib artifact.
`run.json` records configuration, package versions, provenance, and output hashes.
Existing output directories are protected; use `--output artifacts/another-run`
to rerun. The command fits only on train and evaluates development by default.
Calibration is reserved; `--split test` is an explicit final-test operation.
Laya is listed as deferred in this baseline-only report. Its separate comparison
command reuses these saved results.

| Model | Development accuracy | Routing coverage | Error among routed cases |
|---|---:|---:|---:|
| Majority | 23.06% | 100.00% | 76.94% |
| Rules | 44.61% | 48.67% | 8.33% |
| TF-IDF + logistic regression | 89.57% | 100.00% | 10.43% |

Accuracy counts abstentions as incorrect. Coverage/error trade-offs, per-queue
metrics, probability quality, timing, and uncertainty are in the report. These
are uncalibrated development results against provisional eventual-owner targets.
See [evaluation details](docs/baseline-evaluation.md) for definitions and limits.

## Laya comparison (Phase 6)

```powershell
uv sync --locked --extra baseline --extra ml
uv run --extra baseline --extra ml rightdesk download-laya
uv run --extra baseline --extra ml rightdesk benchmark-laya
```

The pinned English checkpoint and bundled tokenizer/encoder assets are stored
under `artifacts/models/`. The benchmark uses one loaded Agent, batches of four,
and the same development cases as the saved baselines. It uses queue descriptions
without training on RightDesk examples. Probabilities, raw SDK responses, model
metadata, and a combined report are saved under `artifacts/laya-development-v1/`.

The real run completed all **863 development cases with zero failures**.
Zero-shot Laya achieved **53.30% accuracy** and **0.5152 macro F1**, compared with
**89.57%** and **0.9030** for trained TF-IDF + logistic regression. This configuration
does not show a Laya advantage. Phase 7 adds the separate calibration analysis;
final-test results remain reserved.

`run.json` marks a successful complete run. An inference failure leaves
`failed_run.json` and any partial outputs, without a completed run manifest.
Choose a fresh `--output` path for another run. Laya batch-amortized latency is
labeled separately from baseline single-ticket latency. See
[Laya integration](docs/laya-integration.md) for pins, limits, and failure behavior.

## Calibration and routing (Phase 7)

After the development benchmarks, capture reserved calibration predictions and fit
the routing policies:

```powershell
uv run --extra baseline --extra ml rightdesk predict-calibration
uv run --extra baseline --extra ml rightdesk calibrate-routing
```

Base models stay frozen. The 1,063 calibration cases are allocated by normalized
text group: 544 fit, 263 threshold selection, and 256 independent validation.
Temperature scaling, a common 5% observed-error selection target, and an independent
validation gate compare raw/calibrated probabilities and native rules. Failed
candidates produce review-only shadow policies. Phase 14 evaluates the final test separately.
Generated outputs live under `artifacts/calibration-predictions-v1/` and
`artifacts/calibration-v1/`. See [calibration and routing](docs/calibration-routing.md)
for reproducibility commands, statistical limits, and policy artifacts.

The completed validation run gives TF-IDF **95.31% candidate coverage** with
**3.69% ticket error** (244 routed, 9 wrong, out of 256 validation tickets).
Its corrected group-error upper bound is **7.90%**, above the 5% gate, so its
saved policies remain review-only. Laya has no selection threshold meeting the
target and group minimum. **No variant passed the complete gate.** Calibration
improved TF-IDF Brier score from 0.1383 to 0.0937; the detailed report is
`artifacts/calibration-v1/report.md`. All 1,063 inference cases completed without
failures. See the handoff for the current verification results.

## Experiment store, dashboard, and human review (Phases 8-10)

The local workbench at `http://127.0.0.1:8501` reads completed experiments from
`artifacts/workbench/rightdesk.sqlite3`. The catalog currently has **3 runs,
7,833 predictions, and 1,926 development/calibration tickets**. Browsing and
filtering use saved results without model inference. Final-test data is excluded.

The dashboard compares model metrics, candidate versus enabled policy coverage,
risk curves, reliability, errors, and individual tickets. The review workflow
supports blind intake judgments, explicit unknown values, multiple reviewers,
append-only revisions, conflict protection, reviewer agreement, and exports.
A **50-case development review queue is ready for human annotation**; human labels
have not been fabricated. Existing Phase 7 policies remain review-only.

```powershell
uv run --no-sync rightdesk store list-runs
uv run --no-sync rightdesk store export-run RUN_ID artifacts/workbench/exports/my-run
uv run --no-sync rightdesk review list-queues
```

See [workbench setup and review guide](docs/workbench.md) for commands, storage
schema, Parquet layouts, review rules, and verification details. The complete suite
includes dashboard interactions, concurrent review saves,
and primary ticket lifecycle checks.

## Clarification and primary helpdesk (Phases 11-12)

Predefined questions are selected only for supported queue ambiguities and within
a one-question budget. An offline run on 863 development tickets proposes 7 questions
for Laya and 0 for TF-IDF with the three-question v2 catalog. Human usefulness remains
unmeasured; question coverage is not evidence that a question helps.

The **Primary helpdesk** dashboard tab runs a separate incumbent using fixed phrase
and answer rules. It supports intake, customer replies, timeouts, agent assignment,
reassignment, resolution/reopening, virtual time, and a persistent public event stream.
It uses `artifacts/simulator/rightdesk.sqlite3`; existing experiments and review-only
candidate policies remain separate.

```powershell
uv run --no-sync rightdesk simulate list
uv run --no-sync rightdesk simulate create --name "Manual helpdesk"
uv run --no-sync rightdesk simulate demo --output artifacts/simulator/new-demo
uv run --no-sync rightdesk clarification evaluate RUN_ID MODEL_ID artifacts/clarification/new-run
```

The completed 14-scenario demo passes all lifecycle assertions: **81 events,
8 questions, 10 resolved, and 4 awaiting agent review**. A separate **50-ticket
ABCD development replay** creates 172 events, with 22 resolved and 28 awaiting review.
Source labels and later dialogue never drive these simulations. See
[clarification and simulator guide](docs/clarification-simulator.md) for commands,
contracts, limits, and reproducible artifacts.

## Independent observer and final evaluation (Phases 13-14)

The **Shadow observer** tab monitors persisted proposals and event backlog. A separate
CLI worker opens the primary database read-only and writes only to its own SQLite
database. Restarting resumes from an atomic cursor; failures produce review proposals.
Neither source labels, primary queue choices, nor future replies become model features.
Actual delivered question text is included only after its public event arrives.

```powershell
uv run --no-sync rightdesk shadow run SIMULATION_ID
uv run --no-sync rightdesk shadow run SIMULATION_ID --follow
uv run --no-sync python scripts/run_e2e.py artifacts/my-end-to-end-run
```

The real-model end-to-end demonstration runs the primary lifecycle, starts and restarts
separate observer processes, verifies no primary mutation, and exports comparisons.
The **Final results** tab reads the checksummed held-out report without exposing test
tickets in the reviewer workbench. The completed test is consumed; do not tune on it.
See [shadow architecture](docs/shadow.md), [results](docs/results.md), and
[handoff](docs/handoff.md) for commands, measurements and limits.

## Design notes

See [architecture](docs/architecture.md), [schemas](docs/schemas.md), and the
[data plan](docs/data-plan.md). The original workflow retains unset thresholds;
Phase 7 writes separate versioned policy snapshots after calibration and validation.
