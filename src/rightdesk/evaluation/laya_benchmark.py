"""Compare a pinned Laya checkpoint with an existing, integrity-checked baseline run."""

import json
import shutil
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Callable

from rightdesk.config import load_experiment, load_workflow, resolve_config_path
from rightdesk.data.importers.abcd import checksum, write_json
from rightdesk.data.preprocessing import duplicate_key
from rightdesk.evaluation.benchmark import load_partition
from rightdesk.evaluation.metrics import evaluate
from rightdesk.evaluation.reports import write_report
from rightdesk.models.laya_model import LayaModel


def benchmark_laya(
    experiment_path: Path,
    adapter_path: Path,
    baseline_dir: Path,
    output_dir: Path,
    progress: Callable[[str], None] = print,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Comparison output exists; choose a fresh directory")
    baseline_path = baseline_dir / "run.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if set(baseline["models_run"]) != {"majority", "rules", "tfidf_logistic"}:
        raise ValueError("Expected the three-model baseline run")
    for name, digest in baseline["artifacts"].items():
        if checksum(baseline_dir / name) != digest:
            raise ValueError(f"Baseline artifact checksum mismatch: {name}")
    config = load_experiment(experiment_path)
    workflow = load_workflow(resolve_config_path(experiment_path, config.workflow))
    if workflow.model_dump() != baseline["workflow"]:
        raise ValueError("Baseline and Laya workflows must match")
    if config.dataset.manifest is None:
        raise ValueError("Dataset manifest is required")
    manifest_path = resolve_config_path(experiment_path, config.dataset.manifest)
    if checksum(manifest_path) != baseline["dataset_manifest_sha256"]:
        raise ValueError("Baseline and Laya datasets must match")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = baseline["evaluation_split"]
    if split not in ("development", "test"):
        raise ValueError("Calibration data cannot be benchmarked")
    tickets, targets = load_partition(manifest_path, manifest, split, workflow)
    for name in baseline["models_run"]:
        with (baseline_dir / name / "predictions.jsonl").open(encoding="utf-8") as stream:
            ids = [json.loads(line)["prediction"]["case_id"] for line in stream]
        if ids != [ticket.case_id for ticket in tickets]:
            raise ValueError("Baseline and Laya case order/membership differ")
    model = LayaModel(workflow, adapter_path)
    configured = next((entry for entry in config.models if entry.id == "laya"), None)
    if configured is None or (configured.checkpoint, configured.revision) != (
        model.config.checkpoint,
        model.config.revision,
    ):
        raise ValueError("Experiment and adapter checkpoint pins differ")
    progress("Loading the verified local Laya checkpoint...")
    start = perf_counter()
    model.load()
    load_seconds = perf_counter() - start
    output_dir.mkdir(parents=True)
    folder = output_dir / "laya"
    folder.mkdir()
    predictions, latencies, batches = [], [], []
    try:
        model.predict_one(tickets[0])  # Warm-up excluded from latency measurements.
        with (
            (folder / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream,
            (folder / "raw_responses.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as raw_stream,
        ):
            for offset in range(0, len(tickets), model.config.batch_size):
                chunk = tickets[offset : offset + model.config.batch_size]
                start = perf_counter()
                results = model.predict_batch(chunk)
                elapsed_ms = (perf_counter() - start) * 1000
                batches.append({"offset": offset, "size": len(chunk), "elapsed_ms": elapsed_ms})
                for ticket, prediction, raw in zip(chunk, results, model.last_raw, strict=True):
                    amortized = elapsed_ms / len(chunk)
                    predictions.append(prediction)
                    latencies.append(amortized)
                    stream.write(
                        json.dumps(
                            {
                                "prediction": prediction.model_dump(),
                                "latency_ms": amortized,
                                "latency_kind": "batch_amortized",
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    raw_stream.write(
                        json.dumps({"case_id": ticket.case_id, "response": raw}, allow_nan=False)
                        + "\n"
                    )
                stream.flush()
                raw_stream.flush()
                if offset % (model.config.batch_size * 10) == 0 or offset + len(chunk) == len(
                    tickets
                ):
                    progress(f"Laya: {offset + len(chunk)}/{len(tickets)} cases")
        metrics = evaluate(
            predictions,
            targets,
            [duplicate_key(t.text) for t in tickets],
            sorted(workflow.queue_ids),
            latencies,
            baseline["seed"],
            baseline["metrics"]["majority"]["uncertainty"]["samples"],
        )
        metrics.update(
            model_revision=model.revision,
            fit_seconds=0.0,
            load_seconds=load_seconds,
            latency_method="batch-amortized milliseconds per ticket, not single-request latency",
            failed_cases=0,
        )
        model.save(folder)
        write_json(folder / "metrics.json", metrics)
        write_json(folder / "batch_timings.json", {"batches": batches})
        run = {
            **baseline,
            "created_at": datetime.now(UTC).isoformat(),
            "report_title": "RightDesk baseline and Laya comparison",
            "models_run": [*baseline["models_run"], "laya"],
            "models_deferred": [],
            "metrics": {**baseline["metrics"], "laya": metrics},
            "baseline_source": str(baseline_path.resolve()),
            "baseline_run_sha256": checksum(baseline_path),
            "laya_config_sha256": checksum(adapter_path),
            "laya_model": model.metadata,
            "config": config.model_dump(),
            "config_sha256": checksum(experiment_path),
            "latency_method": (
                "Baseline: serial one-ticket calls. Laya: batch-amortized; not directly comparable."
            ),
            "timing_note": (
                "Laya p95 is batch-amortized ms per case. Baseline p95 is single-ticket latency; "
                "do not compare them as response times."
            ),
            "packages": {
                **baseline["packages"],
                **{p: version(p) for p in ("laya", "torch", "transformers", "huggingface-hub")},
            },
        }
        root = Path(__file__).parents[1]
        run["code_sha256"] = {
            p.relative_to(root).as_posix(): checksum(p)
            for p in (
                root / "models/laya_model.py",
                root / "models/base.py",
                root / "schemas.py",
                root / "evaluation/laya_benchmark.py",
                root / "evaluation/metrics.py",
                root / "evaluation/reports.py",
                root / "evaluation/benchmark.py",
            )
        }
        run["baseline_code_sha256"] = baseline["code_sha256"]
        run["lockfile_sha256"] = checksum(root.parents[1] / "uv.lock")
        for name in baseline["models_run"]:
            shutil.copytree(baseline_dir / name, output_dir / name)
        write_report(output_dir, run)
        run["artifacts"] = {
            p.relative_to(output_dir).as_posix(): checksum(p)
            for p in sorted(output_dir.rglob("*"))
            if p.is_file()
        }
        write_json(output_dir / "run.json", run)
    except Exception as exc:
        write_json(
            output_dir / "failed_run.json",
            {
                "status": "failed",
                "completed_cases": len(predictions),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "note": "Partial outputs are not a completed benchmark; no cases silently dropped.",
            },
        )
        raise
    return output_dir / "report.md"
