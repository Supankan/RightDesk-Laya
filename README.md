# RightDesk

A local workbench for evaluating whether an incoming support request should be
routed, clarified with a predefined question, or sent to a person for review.

## Status

Phases 1-6 implemented: validated contracts, reproducible ABCD preparation, provisional
queue mapping, isolated partitions, three working baselines, and a saved development
benchmark. The original import has 10,042 cases; the partitioned dataset has 8,722
active cases and 1,320 documented exclusions. The pinned Laya adapter supports
local batched inference and comparison with the saved baselines. RightDesk-specific
calibration, policy execution, the general experiment store, and the dashboard
remain future work.

## Layout

```text
apps/dashboard.py              Streamlit placeholder
configs/workflows/             Queue and question definitions
configs/experiments/           Experiment templates
src/rightdesk/schemas.py       Validated data contracts
src/rightdesk/config.py        YAML loading and experiment configuration
src/rightdesk/data/            Import, preprocessing, and splits
src/rightdesk/models/          Replaceable prediction adapters
src/rightdesk/policy/          Routing and clarification policies
src/rightdesk/evaluation/      Calibration, metrics, and reports
src/rightdesk/storage.py       Planned SQLite/artifact storage
src/rightdesk/services.py      Shared application operations
src/rightdesk/cli.py           Configuration and data preparation CLI
data/                         Local datasets and source manifests
artifacts/                    Local predictions, models, and reports
docs/                         Architecture, schemas, and data plan
tests/                        Contracts, configuration, and ingestion tests
```

## Planned stack

- Python 3.12 and uv for the project environment.
- Pydantic for contracts, YAML for workflow and experiment configuration,
  and Typer for the future CLI commands.
- Laya and scikit-learn for model comparisons.
- NumPy, pandas, and Parquet through PyArrow for experiment data.
- SQLite for run metadata and reviewer annotations.
- Streamlit and Plotly for the local review dashboard.
- pytest and Ruff for development checks.

Optional extras separate the baseline, ML, data, and UI dependencies. `uv.lock`
records resolved versions. Core, development, and baseline dependencies have been
installed and tested. The baseline extra adds NumPy/scikit-learn without loading
Laya or PyTorch. The ML extra is installed for Phase 6's CPU Laya integration;
data/UI extras have not been runtime-verified.

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

To open the placeholder UI:

```powershell
uv sync --extra ui
uv run streamlit run apps/dashboard.py
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
does not show a Laya advantage. Calibration and final-test results remain reserved.

`run.json` marks a successful complete run. An inference failure leaves
`failed_run.json` and any partial outputs, without a completed run manifest.
Choose a fresh `--output` path for another run. Laya batch-amortized latency is
labeled separately from baseline single-ticket latency. See
[Laya integration](docs/laya-integration.md) for pins, limits, and failure behavior.

## Design notes

See [architecture](docs/architecture.md), [schemas](docs/schemas.md), and the
[data plan](docs/data-plan.md). Workflow and experiment files are templates;
thresholds remain unset until calibration evidence is available.
