"""Capture reserved-calibration predictions from frozen models, without refitting."""

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
from threadpoolctl import threadpool_limits

from rightdesk.config import load_experiment, load_workflow, resolve_config_path
from rightdesk.data.importers.abcd import checksum, write_json
from rightdesk.evaluation.benchmark import load_partition
from rightdesk.models.laya_model import LayaModel, load_laya_config
from rightdesk.models.majority import MajorityModel
from rightdesk.models.rules import RulesModel
from rightdesk.models.sklearn_model import TfidfLogisticModel
from rightdesk.schemas import ModelPrediction

MODEL_NAMES = ("majority", "rules", "tfidf_logistic", "laya")


def checked_run(directory: Path) -> dict:
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if not run.get("artifacts"):
        raise ValueError("Completed run must list artifact hashes")
    for name, digest in run["artifacts"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or checksum(path) != digest:
            raise ValueError(f"Run artifact checksum/path mismatch: {name}")
    return run


def calibration_context(experiment_path: Path, adapter_path: Path, source_dir: Path):
    source = checked_run(source_dir)
    if set(source["models_run"]) != set(MODEL_NAMES) or source["evaluation_split"] != "development":
        raise ValueError("Calibration requires the completed four-model development comparison")
    config = load_experiment(experiment_path)
    workflow_path = resolve_config_path(experiment_path, config.workflow)
    workflow = load_workflow(workflow_path)
    if workflow.model_dump() != source["workflow"] or config.dataset.manifest is None:
        raise ValueError("Source workflow mismatch or missing dataset manifest")
    manifest_path = resolve_config_path(experiment_path, config.dataset.manifest)
    if checksum(manifest_path) != source["dataset_manifest_sha256"]:
        raise ValueError("Source dataset manifest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["queue_mapping_version"] != config.dataset.queue_mapping_version:
        raise ValueError("Mapping version mismatch")
    if manifest["provenance"]["workflow"]["sha256"] != checksum(workflow_path):
        raise ValueError("Workflow changed since partition preparation")
    for entry in manifest["provenance"].values():
        if checksum(resolve_config_path(manifest_path, entry["path"])) != entry["sha256"]:
            raise ValueError("Partition provenance mismatch")
    if checksum(adapter_path) != source["laya_config_sha256"]:
        raise ValueError("Laya adapter configuration differs from development")
    adapter = load_laya_config(adapter_path)
    configured = next((model for model in config.models if model.id == "laya"), None)
    if configured is None or (configured.checkpoint, configured.revision) != (
        adapter.checkpoint,
        adapter.revision,
    ):
        raise ValueError("Experiment and adapter checkpoint pins differ")
    tickets, targets = load_partition(manifest_path, manifest, "calibration", workflow)
    development_ids = {
        json.loads(line)["prediction"]["case_id"]
        for line in (source_dir / "laya/predictions.jsonl").read_text(encoding="utf-8").splitlines()
    }
    if development_ids & {ticket.case_id for ticket in tickets}:
        raise ValueError("Calibration/development case overlap")
    return config, workflow, source, manifest_path, tickets, targets


def frozen_baselines(source_dir, workflow, source):
    for package in ("scikit-learn", "numpy"):
        if version(package) != source["packages"][package]:
            raise ValueError(f"Frozen pipeline runtime version mismatch: {package}")
    models = {}
    majority = MajorityModel(workflow)
    metadata = json.loads((source_dir / "majority/model.json").read_text())
    majority.probabilities = metadata["probabilities"]
    models["majority"] = majority
    rules_metadata = json.loads((source_dir / "rules/model.json").read_text())
    # Load the exact saved rules, rather than a possibly edited live rules YAML.
    # RulesConfig accepts JSON as YAML, so the saved config is extracted by the caller.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="rightdesk-rules-") as temporary:
        path = Path(temporary) / "rules.json"
        path.write_text(json.dumps(rules_metadata["rules"]), encoding="utf-8")
        models["rules"] = RulesModel(workflow, path)
    tfidf = TfidfLogisticModel(workflow)
    # Only load our integrity-checked local artifact; joblib is not an untrusted interchange format.
    tfidf.pipeline = joblib.load(source_dir / "tfidf_logistic/pipeline.joblib")
    models["tfidf_logistic"] = tfidf
    for name, model in models.items():
        metadata = json.loads((source_dir / name / "model.json").read_text())
        model.revision, model.metadata, model.fitted = metadata["revision"], metadata, True
    return models


def capture_calibration(
    experiment_path: Path, adapter_path: Path, source_dir: Path, output_dir: Path, progress=print
) -> Path:
    if output_dir.exists():
        raise ValueError("Calibration prediction output exists; choose a fresh directory")
    config, workflow, source, manifest_path, tickets, _ = calibration_context(
        experiment_path, adapter_path, source_dir
    )
    models = frozen_baselines(source_dir, workflow, source)
    model = LayaModel(workflow, adapter_path)
    expected_revision = source["metrics"]["laya"]["model_revision"]
    if model.revision != expected_revision:
        raise ValueError("Laya revision differs from development")
    output_dir.mkdir(parents=True)
    completed = 0
    try:
        for name, baseline in models.items():
            directory = output_dir / name
            directory.mkdir()
            with (directory / "predictions.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                with threadpool_limits(limits=1):
                    for ticket in tickets:
                        stream.write(
                            json.dumps(baseline.predict_one(ticket).model_dump(), allow_nan=False)
                            + "\n"
                        )
        progress(f"Frozen baselines: {len(tickets)} calibration cases each; no refitting.")
        progress("Loading the verified local Laya checkpoint for calibration predictions...")
        start = perf_counter()
        model.load()
        load_seconds = perf_counter() - start
        directory = output_dir / "laya"
        directory.mkdir()
        model.predict_one(tickets[0])
        with (
            (directory / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream,
            (directory / "raw_responses.jsonl").open("w", encoding="utf-8", newline="\n") as raw,
        ):
            for offset in range(0, len(tickets), model.config.batch_size):
                chunk = tickets[offset : offset + model.config.batch_size]
                predictions = model.predict_batch(chunk)
                for ticket, prediction, response in zip(
                    chunk, predictions, model.last_raw, strict=True
                ):
                    stream.write(json.dumps(prediction.model_dump(), allow_nan=False) + "\n")
                    raw.write(
                        json.dumps(
                            {"case_id": ticket.case_id, "response": response}, allow_nan=False
                        )
                        + "\n"
                    )
                    completed += 1
                stream.flush()
                raw.flush()
                if offset % (model.config.batch_size * 10) == 0 or completed == len(tickets):
                    progress(f"Laya calibration: {completed}/{len(tickets)} cases")
        run = {
            "schema_version": 1,
            "purpose": "reserved_calibration_predictions",
            "created_at": datetime.now(UTC).isoformat(),
            "cases": len(tickets),
            "models_run": list(MODEL_NAMES),
            "failed_cases": 0,
            "workflow": workflow.model_dump(),
            "dataset_manifest_sha256": checksum(manifest_path),
            "source_run_sha256": checksum(source_dir / "run.json"),
            "experiment_sha256": checksum(experiment_path),
            "adapter_sha256": checksum(adapter_path),
            "model_revisions": {
                name: source["metrics"][name]["model_revision"] for name in MODEL_NAMES
            },
            "laya_load_seconds": load_seconds,
            "laya_model": model.metadata,
            "packages": {p: version(p) for p in ("numpy", "scikit-learn", "laya", "torch")},
            "prediction_code_sha256": checksum(Path(__file__)),
            "artifacts": {
                p.relative_to(output_dir).as_posix(): checksum(p)
                for p in sorted(output_dir.rglob("*"))
                if p.is_file()
            },
        }
        write_json(output_dir / "run.json", run)
    except Exception as exc:
        write_json(
            output_dir / "failed_run.json",
            {
                "status": "failed",
                "completed_laya_cases": completed,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    return output_dir / "run.json"


def load_calibration_predictions(directory, source_dir, workflow, source, manifest_path, tickets):
    run = checked_run(directory)
    if run.get("purpose") != "reserved_calibration_predictions" or run.get("failed_cases") != 0:
        raise ValueError("Expected completed calibration prediction capture")
    if (
        run["source_run_sha256"] != checksum(source_dir / "run.json")
        or run["dataset_manifest_sha256"] != checksum(manifest_path)
        or run["workflow"] != workflow.model_dump()
        or run["cases"] != len(tickets)
        or set(run["models_run"]) != set(MODEL_NAMES)
    ):
        raise ValueError("Calibration capture provenance mismatch")
    predictions = {}
    for name in MODEL_NAMES:
        rows = [
            ModelPrediction.model_validate_json(line, context={"workflow": workflow})
            for line in (directory / name / "predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if [p.case_id for p in rows] != [t.case_id for t in tickets]:
            raise ValueError("Calibration prediction order/membership mismatch")
        if any(
            p.model_revision != source["metrics"][name]["model_revision"]
            or p.calibration_version is not None
            for p in rows
        ):
            raise ValueError("Calibration prediction model identity mismatch")
        if any((p.queue_probabilities is None) != (name == "rules") for p in rows):
            raise ValueError("Calibration prediction probability contract mismatch")
        predictions[name] = rows
    return predictions
