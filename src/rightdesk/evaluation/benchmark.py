"""A focused baseline benchmark; the general experiment store arrives in Phase 8."""

import json
import platform
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from threadpoolctl import threadpool_limits

from rightdesk.config import load_experiment, load_workflow, resolve_config_path
from rightdesk.data.importers.abcd import checksum, write_json
from rightdesk.data.preprocessing import duplicate_key
from rightdesk.evaluation.metrics import evaluate
from rightdesk.evaluation.reports import write_report
from rightdesk.models.majority import MajorityModel
from rightdesk.models.rules import RulesModel
from rightdesk.models.sklearn_model import TfidfLogisticModel
from rightdesk.schemas import ReviewedLabel, TicketInput, WorkflowConfig


def load_partition(manifest_path: Path, manifest: dict, role: str, workflow: WorkflowConfig):
    records = []
    for name, schema in (("tickets.jsonl", TicketInput), ("labels.jsonl", ReviewedLabel)):
        entry = manifest["outputs"][f"{role}/{name}"]
        path = resolve_config_path(manifest_path, entry["path"])
        if checksum(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Partition checksum mismatch: {path}")
        indexed = {}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = schema.model_validate_json(line, context={"workflow": workflow})
                if record.case_id in indexed:
                    raise ValueError(f"Duplicate {role} case ID: {record.case_id}")
                indexed[record.case_id] = record
        records.append(indexed)
    tickets, labels = records
    if not tickets or tickets.keys() != labels.keys():
        raise ValueError(f"Incomplete {role} ticket/label join")
    if len(tickets) != manifest["report"]["partition_counts"][role]:
        raise ValueError(f"{role} count differs from manifest")
    keys = sorted(tickets)
    ordered = [tickets[key] for key in keys]
    if len({ticket.conversation_id for ticket in ordered}) != len(ordered):
        raise ValueError(f"Duplicate conversation in {role}")
    targets = [labels[key].eventual_owner_queue for key in keys]
    if any(target is None for target in targets):
        raise ValueError(f"Missing eventual-owner target in {role}")
    return ordered, targets


def benchmark_baselines(
    config_path: Path,
    output_dir: Path,
    split: str = "development",
    bootstrap_samples: int = 500,
) -> Path:
    if split not in {"development", "test"}:
        raise ValueError("Benchmark split must be development or test; calibration is reserved")
    if bootstrap_samples < 20:
        raise ValueError("At least 20 bootstrap samples are required")
    config_path, output_dir = config_path.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Benchmark output exists; choose a fresh output directory")
    config = load_experiment(config_path)
    workflow_path = resolve_config_path(config_path, config.workflow)
    workflow = load_workflow(workflow_path)
    if config.dataset.manifest is None or config.dataset.processed_path is None:
        raise ValueError("Configure a prepared dataset manifest and processed path first")
    manifest_path = resolve_config_path(config_path, config.dataset.manifest)
    processed_path = resolve_config_path(config_path, config.dataset.processed_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("dataset"), manifest.get("schema_version")) != ("abcd", 1):
        raise ValueError("Expected an ABCD partition manifest v1")
    if (manifest["workflow_id"], manifest["workflow_version"]) != (
        workflow.workflow_id,
        workflow.version,
    ):
        raise ValueError("Partition workflow mismatch")
    if manifest["queue_mapping_version"] != config.dataset.queue_mapping_version:
        raise ValueError("Queue mapping version mismatch")
    if manifest["provenance"]["workflow"]["sha256"] != checksum(workflow_path):
        raise ValueError("Workflow contents changed since partition preparation")
    for entry in manifest["provenance"].values():
        if checksum(resolve_config_path(manifest_path, entry["path"])) != entry["sha256"]:
            raise ValueError("Partition provenance checksum mismatch")
    for role in ("train", split):
        for name in ("tickets.jsonl", "labels.jsonl"):
            path = resolve_config_path(manifest_path, manifest["outputs"][f"{role}/{name}"]["path"])
            if path != processed_path / role / name:
                raise ValueError("Configured processed path does not match partition artifacts")
    train, train_targets = load_partition(manifest_path, manifest, "train", workflow)
    evaluation, eval_targets = load_partition(manifest_path, manifest, split, workflow)
    for attribute in ("case_id", "conversation_id"):
        if {getattr(t, attribute) for t in train} & {getattr(t, attribute) for t in evaluation}:
            raise ValueError(f"Train/evaluation {attribute} overlap")
    if {duplicate_key(t.text) for t in train} & {duplicate_key(t.text) for t in evaluation}:
        raise ValueError("Train/evaluation normalized-message overlap")
    model_configs = {model.id: model for model in config.models}
    required = {"majority", "rules", "tfidf_logistic"}
    if not required <= model_configs.keys() or model_configs["rules"].rules_path is None:
        raise ValueError("Configure all three baselines and a rules_path")
    rules_path = resolve_config_path(config_path, model_configs["rules"].rules_path)
    models = {
        "majority": MajorityModel(workflow),
        "rules": RulesModel(workflow, rules_path),
        "tfidf_logistic": TfidfLogisticModel(workflow, config.seed),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".baseline-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        reports = {}
        for name, model in models.items():
            folder = staging / name
            folder.mkdir()
            start = perf_counter()
            model.fit(train, train_targets)
            fit_seconds = perf_counter() - start
            predictions, latencies = [], []
            with threadpool_limits(limits=1):
                model.predict_one(evaluation[0])  # One unmeasured warm-up call; no state fitting.
                for ticket in evaluation:
                    start = perf_counter()
                    prediction = model.predict_one(ticket)
                    latencies.append((perf_counter() - start) * 1000)
                    predictions.append(prediction)
            model.save(folder)
            with (folder / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for prediction, latency in zip(predictions, latencies, strict=True):
                    stream.write(
                        json.dumps(
                            {"prediction": prediction.model_dump(), "latency_ms": latency},
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
            metrics = evaluate(
                predictions,
                eval_targets,
                [duplicate_key(ticket.text) for ticket in evaluation],
                sorted(workflow.queue_ids),
                latencies,
                config.seed,
                bootstrap_samples,
            )
            metrics["fit_seconds"] = fit_seconds
            metrics["model_revision"] = model.revision
            write_json(folder / "metrics.json", metrics)
            reports[name] = metrics
        package_root = Path(__file__).parents[1]
        sources = [
            package_root / "schemas.py",
            package_root / "config.py",
            package_root / "data/preprocessing.py",
            *sorted((package_root / "models").glob("*.py")),
            *sorted((package_root / "evaluation").glob("*.py")),
        ]
        run = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "created_at": datetime.now(UTC).isoformat(),
            "evaluation_split": split,
            "training_cases": len(train),
            "evaluation_cases": len(evaluation),
            "seed": config.seed,
            "models_run": list(models),
            "models_deferred": sorted(model_configs.keys() - models.keys()),
            "mapping_status": manifest["mapping_status"],
            "reference_label": "scenario-derived eventual owner, not reviewed intake correctness",
            "calibration_status": "not performed; no routing threshold selected",
            "latency_method": (
                "one warm-up; serial single-ticket calls incl. schema validation; one BLAS thread"
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: version(package)
                for package in ("numpy", "scikit-learn", "scipy", "pydantic", "joblib")
            },
            "config": config.model_dump(),
            "workflow": workflow.model_dump(),
            "dataset_manifest_sha256": checksum(manifest_path),
            "config_sha256": checksum(config_path),
            "rules_sha256": checksum(rules_path),
            "code_sha256": {
                path.relative_to(package_root).as_posix(): checksum(path) for path in sources
            },
            "metrics": reports,
        }
        lock_path = package_root.parents[1] / "uv.lock"
        if lock_path.exists():
            run["lockfile_sha256"] = checksum(lock_path)
        write_report(staging, run)
        run["artifacts"] = {
            path.relative_to(staging).as_posix(): checksum(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        write_json(staging / "run.json", run)
        staging.rename(output_dir)
    return output_dir / "report.md"
