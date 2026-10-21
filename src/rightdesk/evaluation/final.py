"""Frozen held-out evaluation. No model fitting, threshold selection, or gate changes."""

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from threadpoolctl import threadpool_limits

from rightdesk.data.importers.abcd import checksum, write_json
from rightdesk.data.preprocessing import duplicate_key
from rightdesk.evaluation.benchmark import load_partition
from rightdesk.evaluation.calibration import TemperatureCalibrator, routed_mask, routing_summary
from rightdesk.evaluation.calibration_predictions import checked_run, frozen_baselines
from rightdesk.evaluation.metrics import evaluate
from rightdesk.models.laya_model import LayaModel
from rightdesk.policy.routing import RoutingPolicy, RoutingSettings
from rightdesk.schemas import WorkflowConfig

NAMES = ("majority", "rules", "tfidf_logistic", "laya")
VARIANTS = tuple(
    (name, kind)
    for name in NAMES
    for kind in (("native",) if name == "rules" else ("raw", "temperature"))
)


def freeze_protocol(source: Path, calibration: Path, manifest: Path, adapter: Path, output: Path):
    """Write a dated, content-addressed plan without opening any test records or labels."""
    if output.exists():
        raise ValueError("Protocol already exists; do not overwrite a frozen protocol")
    development, calibrated = checked_run(source), checked_run(calibration)
    if development["evaluation_split"] != "development" or calibrated["phase"] != 7:
        raise ValueError("Expected development models and Phase 7 calibration")
    if calibrated["source_run_sha256"] != checksum(source / "run.json"):
        raise ValueError("Calibration and development source mismatch")
    if development["dataset_manifest_sha256"] != checksum(manifest):
        raise ValueError("Dataset manifest mismatch")
    if development["workflow"] != calibrated["workflow"]:
        raise ValueError("Workflow mismatch")
    expected = {f"{n}_{k}" for n, k in VARIANTS}
    if set(calibrated["variants"]) != expected:
        raise ValueError("Saved policy variants differ from the final protocol")
    for name in expected:
        if not (calibration / name / "policy.json").is_file():
            raise ValueError(f"Missing frozen policy: {name}")
    if development["laya_config_sha256"] != checksum(adapter):
        raise ValueError("Adapter changed")
    root = Path(__file__).parents[1]
    protocol = {
        "schema_version": 1,
        "purpose": "frozen_final_test",
        "frozen_at": datetime.now(UTC).isoformat(),
        "source": str(source.resolve()),
        "calibration": str(calibration.resolve()),
        "manifest": str(manifest.resolve()),
        "adapter": str(adapter.resolve()),
        "input_sha256": {
            "source": checksum(source / "run.json"),
            "calibration": checksum(calibration / "run.json"),
            "manifest": checksum(manifest),
            "adapter": checksum(adapter),
        },
        "workflow": development["workflow"],
        "variants": [f"{name}_{kind}" for name, kind in VARIANTS],
        "split": "test",
        "seed": 42,
        "bootstrap_samples": 500,
        "base_models_refitted": False,
        "thresholds_reselected": False,
        "gate_changes_allowed": False,
        "reference": "provisional eventual owner, not reviewed intake sufficiency",
        "timing": "warm serial single calls for baselines; batch elapsed / batch size for Laya",
        "interpretation": "Native prediction quality is descriptive; enabled policy is separate",
        "code_sha256": {
            p.relative_to(root).as_posix(): checksum(p)
            for folder in ("evaluation", "models", "policy", "data")
            for p in sorted((root / folder).rglob("*.py"))
        },
        "packages": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "laya", "torch")},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, protocol)
    return output


def final_context(protocol_path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["purpose"] != "frozen_final_test" or protocol["variants"] != [
        f"{n}_{k}" for n, k in VARIANTS
    ]:
        raise ValueError("Unsupported final protocol")
    paths = {key: Path(protocol[key]) for key in ("source", "calibration", "manifest", "adapter")}
    for key, path in paths.items():
        if (
            checksum(path / "run.json" if key in ("source", "calibration") else path)
            != protocol["input_sha256"][key]
        ):
            raise ValueError(f"Frozen input changed: {key}")
    root = Path(__file__).parents[1]
    for name, digest in protocol["code_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or checksum(path) != digest:
            raise ValueError(f"Frozen evaluation code changed: {name}")
    for package, expected in protocol["packages"].items():
        if version(package) != expected:
            raise ValueError(f"Frozen runtime changed: {package}")
    source = checked_run(paths["source"])
    calibration = checked_run(paths["calibration"])
    workflow = WorkflowConfig.model_validate(protocol["workflow"])
    if workflow.model_dump() != source["workflow"] or source["workflow"] != calibration["workflow"]:
        raise ValueError("Frozen workflow changed")
    return protocol, paths, workflow, source, calibration


def final_evaluate(protocol_path: Path, output: Path, progress=print):
    if output.exists():
        raise ValueError("Final output exists; final evaluation never overwrites results")
    protocol, paths, workflow, source, calibration = final_context(protocol_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    # First test access occurs only after all frozen inputs and code have been checked.
    tickets, targets = load_partition(paths["manifest"], manifest, "test", workflow)
    groups, queues = [duplicate_key(t.text) for t in tickets], sorted(workflow.queue_ids)
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    predictions, timings = {}, {}
    try:
        models = frozen_baselines(paths["source"], workflow, source)
        for name, model in models.items():
            model.predict_one(tickets[0])
            rows, latency = [], []
            with threadpool_limits(limits=1):
                for ticket in tickets:
                    start = perf_counter()
                    rows.append(model.predict_one(ticket))
                    latency.append((perf_counter() - start) * 1000)
            predictions[name], timings[name] = rows, latency
            progress(f"Final {name}: {len(rows)} cases, frozen model, no refitting", flush=True)
        model = LayaModel(workflow, paths["adapter"])
        if model.revision != source["metrics"]["laya"]["model_revision"]:
            raise ValueError("Laya revision mismatch")
        progress("Loading local Laya checkpoint for frozen final evaluation...", flush=True)
        start = perf_counter()
        model.load()
        load_seconds = perf_counter() - start
        model.predict_one(tickets[0])
        rows, latency = [], []
        with (output / "laya_raw_responses.jsonl").open("w", encoding="utf-8") as raw:
            for offset in range(0, len(tickets), model.config.batch_size):
                chunk = tickets[offset : offset + model.config.batch_size]
                start = perf_counter()
                rows.extend(model.predict_batch(chunk))
                elapsed = (perf_counter() - start) * 1000 / len(chunk)
                latency.extend([elapsed] * len(chunk))
                for ticket, response in zip(chunk, model.last_raw, strict=True):
                    raw.write(json.dumps({"case_id": ticket.case_id, "response": response}) + "\n")
                raw.flush()
                if offset % 40 == 0 or len(rows) == len(tickets):
                    progress(f"Final Laya: {len(rows)}/{len(tickets)}", flush=True)
        predictions["laya"], timings["laya"] = rows, latency
        reports = {}
        for name, kind in VARIANTS:
            variant = f"{name}_{kind}"
            rows = predictions[name]
            if kind == "temperature":
                calibrator = TemperatureCalibrator.model_validate_json(
                    (paths["calibration"] / f"{name}_calibrator.json").read_text()
                )
                rows = [calibrator.transform(p, workflow) for p in rows]
            payload = json.loads((paths["calibration"] / variant / "policy.json").read_text())
            policy_workflow = WorkflowConfig.model_validate(payload["workflow"])
            settings = RoutingSettings.model_validate(payload["settings"])
            if settings.enabled != calibration["variants"][variant]["enabled"]:
                raise ValueError("Policy gate differs from calibration report")
            policy = RoutingPolicy(policy_workflow, settings)
            decisions = [policy.decide(t, p) for t, p in zip(tickets, rows, strict=True)]
            report = evaluate(
                rows,
                targets,
                groups,
                queues,
                timings[name],
                seed=protocol["seed"],
                bootstrap_samples=protocol["bootstrap_samples"],
            )
            candidate_mask = routed_mask(
                rows, policy_workflow.policy.routing_threshold, settings.native_rules
            )
            report["frozen_candidate"] = routing_summary(rows, targets, groups, candidate_mask)
            report["enabled_policy"] = routing_summary(
                rows, targets, groups, [d.action == "route" for d in decisions]
            )
            report["policy_enabled"] = settings.enabled
            report["model_revision"] = rows[0].model_revision
            report["calibration_version"] = rows[0].calibration_version
            directory = output / variant
            directory.mkdir()
            write_json(directory / "metrics.json", report)
            write_json(directory / "policy.json", payload)
            with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
                for p, d, ms in zip(rows, decisions, timings[name], strict=True):
                    stream.write(
                        json.dumps(
                            {
                                "prediction": p.model_dump(),
                                "decision": d.model_dump(),
                                "latency_ms": ms,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
            reports[variant] = report
        run = {
            "schema_version": 1,
            "phase": 14,
            "purpose": "frozen_final_test",
            "completed_at": datetime.now(UTC).isoformat(),
            "evaluation_split": "test",
            "evaluation_cases": len(tickets),
            "protocol_sha256": checksum(protocol_path),
            "failed_cases": 0,
            "base_models_refitted": False,
            "thresholds_reselected": False,
            "gate_changes": False,
            "laya_load_seconds": load_seconds,
            "metrics": reports,
            "protocol": protocol,
        }
        write_final_report(output, run)
        run["artifacts"] = {
            p.relative_to(output).as_posix(): checksum(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        write_json(output / "run.json", run)
        return output / "report.md"
    except Exception as exc:
        write_json(
            output / "failed_run.json",
            {"status": "failed", "error": str(exc), "error_type": type(exc).__name__},
        )
        raise


def write_final_report(output, run):
    lines = [
        "# RightDesk frozen final evaluation",
        "",
        f"Held-out test: {run['evaluation_cases']} cases. No refitting, retuning, or gate changes.",
        "Targets are provisional eventual-owner queues, not human judgments of intake sufficiency.",
        "Native predictions and candidate thresholds are diagnostics. Enabled policy is separate.",
        "",
        "| Variant | Accuracy | Macro F1 | Brier | ECE | Native coverage | Enabled coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def number(value):
        return "N/A" if value is None else f"{value:.4f}"

    for name, report in run["metrics"].items():
        lines.append(
            f"| {name} | {report['accuracy']:.2%} | {report['macro_f1']:.4f} | "
            f"{number(report['brier_score'])} | {number(report['calibration_error'])} | "
            f"{report['operating_point']['automation_coverage']:.2%} | "
            f"{report['enabled_policy']['automation_coverage']:.2%} |"
        )
    lines += [
        "",
        "Full per-queue metrics, confusion matrices, reliability bins, descriptive risk curves,",
        "cluster bootstrap intervals, and frozen candidate risk are in each metrics.json.",
        "Intervals are pointwise descriptive estimates, not safety guarantees.",
        "Timing uses warm serial calls for baselines and amortized batch time for Laya;",
        "these timings do not establish a like-for-like online latency advantage.",
        "No final-test result enables a failed calibration gate.",
        "Question usefulness, human agreement, operational benefit and safety remain unmeasured.",
        "",
        "The protocol, hashes, predictions, SDK responses and policies are retained for audit.",
        "The held-out test is consumed and must not be reused for tuning.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
