"""Fit on calibration-fit, select on selection, validate once on untouched groups."""

import json
import tempfile
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from rightdesk.config import _load
from rightdesk.data.importers.abcd import checksum, write_json
from rightdesk.data.preprocessing import duplicate_key
from rightdesk.evaluation.calibration import (
    CalibrationConfig,
    TemperatureCalibrator,
    assign_roles,
    probability_matrix,
    routed_mask,
    routing_summary,
    select_threshold,
)
from rightdesk.evaluation.calibration_predictions import (
    calibration_context,
    load_calibration_predictions,
)
from rightdesk.evaluation.metrics import evaluate
from rightdesk.policy.routing import RoutingPolicy, RoutingSettings
from rightdesk.schemas import WorkflowConfig


def _jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _quality(predictions, targets, groups, queues, config):
    report = evaluate(
        predictions,
        targets,
        groups,
        queues,
        [0.0] * len(targets),
        config.seed,
        config.bootstrap_samples,
    )
    # This analysis reuses cached predictions; no inference timing is measured.
    report.pop("latency_p50_ms")
    report.pop("latency_p95_ms")
    if predictions[0].queue_probabilities is not None:
        probabilities = probability_matrix(predictions, queues)
        truth = [queues.index(target) for target in targets]
        report["negative_log_likelihood"] = float(
            -np.log(np.maximum(probabilities[np.arange(len(truth)), truth], 1e-12)).mean()
        )
        report["selection_method"] = "Descriptive confidence curve; not used to retune the policy"
    else:
        report["negative_log_likelihood"] = None
    return report


def _report(directory, run):
    def pct(value):
        return "N/A" if value is None else f"{100 * value:.2f}%"

    lines = [
        "# RightDesk calibration and routing validation",
        "",
        f"Reserved calibration cases: {run['cases']}. Roles: {run['role_counts']}.",
        "Final-test cases were not loaded or evaluated. Base models were not refitted.",
        "Targets remain provisional scenario-derived eventual queues, not reviewed intake labels.",
        "",
        f"Common selection target: {pct(run['config']['target_error'])} observed ticket error, "
        f"at least {run['config']['min_routed_groups']} routed normalized-text groups.",
        "Temperatures use fit labels only. Thresholds maximize coverage on selection only.",
        "Thresholds are frozen before validation. "
        "No replacement threshold is chosen after failure.",
        "",
        "## Frozen candidates on independent validation",
        "",
        "| Variant | Threshold | Routed / cases | Coverage | Errors | Ticket error | "
        "Routed groups | Group error upper bound | Gate passed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, variant in run["variants"].items():
        point = variant["validation"]
        threshold = variant["threshold"]
        threshold_text = (
            "native"
            if variant["native_rules"]
            else "review only"
            if threshold is None
            else f"{threshold:.6f}"
        )
        lines.append(
            f"| {name} | {threshold_text} "
            f"| {point['auto_count']} / {point['cases']} | {pct(point['automation_coverage'])} "
            f"| {point['error_count']} | {pct(point['routing_error'])} | {point['routed_groups']} "
            f"| {pct(point['group_error_upper'])} | {'yes' if variant['enabled'] else 'no'} |"
        )
    lines += [
        "",
        "## Gate and interpretation",
        "",
        "A routed group is counted as wrong if any routed ticket in that normalized-text group "
        "is wrong. The one-sided Clopper-Pearson upper bound treats groups as independent "
        "Bernoulli units. It describes group any-error probability, not ticket-weighted risk.",
        f"Bounds use alpha={run['config']['family_alpha']} / {len(run['variants'])} variants "
        "(Bonferroni family correction). This assumes exchangeable independent groups; "
        "corpus shifts and provisional labels limit interpretation.",
        "A shadow policy is enabled only if selection requirements pass, validation ticket error "
        "is within target, enough validation groups route, and the corrected group bound is "
        "within target. Otherwise it proposes review for every ticket.",
        "Candidate coverage above is evaluated before this gate. It must not be presented as "
        "coverage achieved by a disabled policy. No policy is approved for production.",
        "",
        "## Probability quality on validation",
        "",
        "| Variant | Accuracy | Brier | ECE (10 bins) | NLL |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, variant in run["variants"].items():
        metrics = variant["quality"]

        def fmt(value):
            return "N/A" if value is None else f"{value:.4f}"

        lines.append(
            f"| {name} | {pct(metrics['accuracy'])} | {fmt(metrics['brier_score'])} "
            f"| {fmt(metrics['calibration_error'])} | {fmt(metrics['negative_log_likelihood'])} |"
        )
    lines += [
        "",
        "Calibration changes confidence and can change selective coverage; it preserves "
        "the predicted queue. It cannot repair Laya classification errors.",
        "Probability quality can worsen on validation despite a better fit-set NLL.",
        "Raw and temperature variants are predefined comparisons; validation is not used to "
        "choose a replacement calibrator or threshold.",
        "",
        "## Enabled policy coverage",
        "",
    ]
    for name, variant in run["variants"].items():
        point = variant["enabled_policy_validation"]
        lines.append(
            f"- {name}: {point['auto_count']}/{point['cases']} routed; "
            f"{point['review_count']} review; gate: {variant['gate_reason']}."
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "Each variant has its frozen threshold, selection curve, validation metrics, "
        "policy configuration, validation predictions, and shadow decisions. "
        "Calibrators and role assignments are saved separately. run.json records "
        "completion, provenance, settings, package versions, code and output hashes.",
        "Cluster bootstrap intervals in quality metrics are descriptive and may be "
        "degenerate at zero observed errors; the gate uses the separate nonzero "
        "binomial upper bound.",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def calibrate_routing(
    experiment_path: Path,
    adapter_path: Path,
    calibration_path: Path,
    source_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Calibration output exists; choose a fresh directory")
    config = _load(calibration_path, CalibrationConfig)
    _, workflow, source, manifest_path, tickets, targets = calibration_context(
        experiment_path, adapter_path, source_dir
    )
    predictions = load_calibration_predictions(
        predictions_dir, source_dir, workflow, source, manifest_path, tickets
    )
    roles = assign_roles(tickets, config)
    indices = {
        role: [i for i, value in enumerate(roles) if value == role]
        for role in ("fit", "selection", "validation")
    }
    groups = [duplicate_key(ticket.text) for ticket in tickets]

    def select(values, role):
        return [values[i] for i in indices[role]]

    # Allocation must isolate IDs, normalized text, and conversations.
    for attribute in (groups, [t.conversation_id for t in tickets]):
        allocated = {}
        for key, role in zip(attribute, roles, strict=True):
            if key in allocated and allocated[key] != role:
                raise ValueError("Calibration role group/conversation overlap")
            allocated[key] = role
    queues = sorted(workflow.queue_ids)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".calibration-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        _jsonl(
            staging / "assignments.jsonl",
            [
                {"case_id": ticket.case_id, "group": group, "role": role}
                for ticket, group, role in zip(tickets, groups, roles, strict=True)
            ],
        )
        variants = {}
        for name, raw in predictions.items():
            variants[f"{name}_raw" if name != "rules" else "rules_native"] = (raw, None)
            if name == "rules":
                continue
            calibrator = TemperatureCalibrator.fit(
                select(raw, "fit"), select(targets, "fit"), f"{config.version}:{name}"
            )
            write_json(staging / f"{name}_calibrator.json", calibrator.model_dump())
            calibrated = [calibrator.transform(p, workflow) for p in raw]
            variants[f"{name}_temperature"] = (calibrated, calibrator.version)
        # Freeze ALL candidate thresholds before examining ANY validation targets.
        frozen = {}
        for name, (rows, calibration_version) in variants.items():
            chosen, truth, group = (
                select(values, "selection") for values in (rows, targets, groups)
            )
            native = name == "rules_native"
            if native:
                point = routing_summary(chosen, truth, group, routed_mask(chosen, None, True))
                eligible = (
                    point["routing_error"] is not None
                    and point["routing_error"] <= config.target_error
                    and point["routed_groups"] >= config.min_routed_groups
                )
                selection = {
                    "threshold": None,
                    "selection": point,
                    "curve": [],
                    "reason": "native_rules_pass" if eligible else "native_rules_fail",
                }
            else:
                selection = select_threshold(chosen, truth, group, config)
                eligible = selection["threshold"] is not None
            frozen[name] = (selection, eligible, native, calibration_version)
        reports = {}
        for name, (rows, _) in variants.items():
            selection, eligible, native, calibration_version = frozen[name]
            chosen, truth, group = (
                select(values, "validation") for values in (rows, targets, groups)
            )
            point = routing_summary(
                chosen,
                truth,
                group,
                routed_mask(chosen, selection["threshold"], native),
                config.family_alpha / len(variants),
            )
            enabled = bool(
                eligible
                and point["routing_error"] is not None
                and point["routing_error"] <= config.target_error
                and point["routed_groups"] >= config.min_routed_groups
                and point["group_error_upper"] <= config.target_error
            )
            workflow_data = workflow.model_dump()
            workflow_data["policy"].update(
                version=f"{config.version}:{name}",
                routing_threshold=selection["threshold"],
                calibration_version=calibration_version,
                max_questions=0,
                clarification_enabled=False,
                sufficient_information_threshold=None,
            )
            policy_workflow = WorkflowConfig.model_validate(workflow_data)
            settings = RoutingSettings(
                model_revision=chosen[0].model_revision,
                calibration_version=calibration_version,
                enabled=enabled,
                native_rules=native,
            )
            policy = RoutingPolicy(policy_workflow, settings)
            decisions = [policy.decide(tickets[i], rows[i]) for i in indices["validation"]]
            quality = _quality(chosen, truth, group, queues, config)
            # The descriptive curve contains the same tie-preserving routed prefix.
            matching_point = next(
                row
                for row in quality["risk_coverage_curve"]
                if row["auto_count"] == point["auto_count"]
            )
            point["routing_error_ci95"] = matching_point["routing_error_ci95"]
            point["bootstrap_valid_replicates"] = matching_point["bootstrap_valid_replicates"]
            actual = routing_summary(
                chosen,
                truth,
                group,
                [decision.action == "route" for decision in decisions],
                config.family_alpha / len(variants),
            )
            reports[name] = {
                "threshold": selection["threshold"],
                "native_rules": native,
                "selection_eligible": eligible,
                "selection": selection["selection"],
                "validation": point,
                "enabled": enabled,
                "gate_reason": (
                    "passed"
                    if enabled
                    else "selection_requirements_failed"
                    if not eligible
                    else "independent_validation_gate_failed"
                ),
                "enabled_policy_validation": actual,
                "quality": quality,
            }
            folder = staging / name
            folder.mkdir()
            write_json(folder / "selection.json", selection)
            write_json(folder / "metrics.json", reports[name])
            write_json(
                folder / "policy.json",
                {"workflow": policy_workflow.model_dump(), "settings": settings.model_dump()},
            )
            _jsonl(folder / "validation_predictions.jsonl", [p.model_dump() for p in chosen])
            _jsonl(folder / "decisions.jsonl", [decision.model_dump() for decision in decisions])
        root = Path(__file__).parents[1]
        sources = [
            root / "schemas.py",
            root / "config.py",
            root / "cli.py",
            root / "policy/routing.py",
            root / "evaluation/metrics.py",
            root / "evaluation/benchmark.py",
            root / "evaluation/calibration.py",
            root / "evaluation/calibration_predictions.py",
            Path(__file__),
            root / "data/preprocessing.py",
        ]
        run = {
            "schema_version": 1,
            "phase": 7,
            "created_at": datetime.now(UTC).isoformat(),
            "cases": len(tickets),
            "role_counts": dict(Counter(roles)),
            "group_counts": {role: len(set(select(groups, role))) for role in indices},
            "config": config.model_dump(),
            "variants": reports,
            "dataset_manifest_sha256": checksum(manifest_path),
            "source_run_sha256": checksum(source_dir / "run.json"),
            "prediction_run_sha256": checksum(predictions_dir / "run.json"),
            "calibration_config_sha256": checksum(calibration_path),
            "experiment_config_sha256": checksum(experiment_path),
            "adapter_config_sha256": checksum(adapter_path),
            "workflow": workflow.model_dump(),
            "code_sha256": {p.relative_to(root).as_posix(): checksum(p) for p in sources},
            "lockfile_sha256": checksum(root.parents[1] / "uv.lock"),
            "packages": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "pydantic")},
            "reference_label": "provisional eventual owner; no reviewed intake sufficiency",
            "test_evaluated": False,
            "base_models_refitted": False,
        }
        _report(staging, run)
        run["artifacts"] = {
            p.relative_to(staging).as_posix(): checksum(p)
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        }
        write_json(staging / "run.json", run)
        staging.rename(output_dir)
    return output_dir / "report.md"
