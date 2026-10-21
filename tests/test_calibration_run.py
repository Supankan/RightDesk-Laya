"""Complete Phase 7 without test files; frozen inference and role isolation are required."""

import json
from pathlib import Path

import pytest
import yaml
from test_benchmark import benchmark_config as source_benchmark_config
from test_laya_benchmark import fake_load

from rightdesk.config import load_workflow
from rightdesk.data.importers.abcd import checksum
from rightdesk.evaluation import calibration_run
from rightdesk.evaluation.benchmark import benchmark_baselines
from rightdesk.evaluation.calibration import CalibrationConfig, assign_roles
from rightdesk.evaluation.calibration_predictions import capture_calibration
from rightdesk.evaluation.calibration_run import calibrate_routing
from rightdesk.evaluation.laya_benchmark import benchmark_laya
from rightdesk.models.laya_model import LayaModel
from rightdesk.models.sklearn_model import TfidfLogisticModel
from rightdesk.policy.routing import RoutingPolicy, RoutingSettings
from rightdesk.schemas import (
    ModelPrediction,
    PolicyDecision,
    ReviewedLabel,
    TicketInput,
    WorkflowConfig,
)

benchmark_config = source_benchmark_config
ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "configs/models/laya_v1.yaml"


@pytest.fixture
def prepared(benchmark_config, tmp_path, monkeypatch):
    workflow = load_workflow(ROOT / "configs/workflows/retail_support.yaml")
    manifest_path = tmp_path / "partitions.json"
    manifest = json.loads(manifest_path.read_text())
    folder = tmp_path / "calibration"
    folder.mkdir()
    tickets, labels = [], []
    for index in range(240):
        target = "account" if index % 5 else "returns_refunds"
        ticket = TicketInput(
            case_id=f"cal-{index}",
            conversation_id=f"cal-{index}",
            text=f"{'reset password' if target == 'account' else 'refund please'} example {index}",
            language="en",
            channel="chat",
        )
        label = ReviewedLabel.for_workflow(
            {
                "case_id": ticket.case_id,
                "workflow_id": workflow.workflow_id,
                "workflow_version": workflow.version,
                "eventual_owner_queue": target,
            },
            workflow,
        )
        tickets.append(ticket)
        labels.append(label)
    for name, rows in [("tickets.jsonl", tickets), ("labels.jsonl", labels)]:
        path = folder / name
        path.write_text("\n".join(row.model_dump_json() for row in rows) + "\n", encoding="utf-8")
        manifest["outputs"][f"calibration/{name}"] = {
            "path": f"calibration/{name}",
            "sha256": checksum(path),
            "size_bytes": path.stat().st_size,
        }
    manifest["report"]["partition_counts"]["calibration"] = len(tickets)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(LayaModel, "load", fake_load)
    benchmark_baselines(benchmark_config, tmp_path / "baseline", bootstrap_samples=20)
    benchmark_laya(
        benchmark_config, ADAPTER, tmp_path / "baseline", tmp_path / "source", lambda message: None
    )
    calibration = CalibrationConfig(
        version="test-v1",
        target_error=0.3,
        min_routed_groups=2,
        family_alpha=0.2,
        bootstrap_samples=20,
    )
    calibration_path = tmp_path / "calibration.yaml"
    calibration_path.write_text(yaml.safe_dump(calibration.model_dump()), encoding="utf-8")

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Phase 7 must not refit the base model")

    monkeypatch.setattr(TfidfLogisticModel, "fit", forbidden_fit)
    capture_calibration(
        benchmark_config,
        ADAPTER,
        tmp_path / "source",
        tmp_path / "predictions",
        lambda message: None,
    )
    return benchmark_config, calibration_path, tickets


def execute(prepared, tmp_path, output="result"):
    config, calibration, _ = prepared
    return calibrate_routing(
        config,
        ADAPTER,
        calibration,
        tmp_path / "source",
        tmp_path / "predictions",
        tmp_path / output,
    )


def test_complete_run_has_checked_policies_decisions_and_no_final_test(prepared, tmp_path):
    report = execute(prepared, tmp_path)
    run = json.loads((report.parent / "run.json").read_text())
    assert not run["test_evaluated"]
    assert not run["base_models_refitted"]
    assert sum(run["role_counts"].values()) == 240
    assert len(run["variants"]) == 7
    assert not (tmp_path / "test").exists()
    for name, digest in run["artifacts"].items():
        assert checksum(report.parent / name) == digest
    for name, info in run["variants"].items():
        folder = report.parent / name
        artifact = json.loads((folder / "policy.json").read_text())
        workflow = WorkflowConfig.model_validate(artifact["workflow"])
        policy = RoutingPolicy(workflow, RoutingSettings.model_validate(artifact["settings"]))
        rows = [
            ModelPrediction.model_validate_json(line, context={"workflow": workflow})
            for line in (folder / "validation_predictions.jsonl").read_text().splitlines()
        ]
        decisions = [
            PolicyDecision.model_validate_json(line, context={"workflow": workflow})
            for line in (folder / "decisions.jsonl").read_text().splitlines()
        ]
        assert len(rows) == len(decisions) == run["role_counts"]["validation"]
        assert (
            sum(d.action == "route" for d in decisions)
            == info["enabled_policy_validation"]["auto_count"]
        )
        assert all(d.question_id is None for d in decisions)
        assert policy.settings.enabled == info["enabled"]
    assert any(info["enabled"] for info in run["variants"].values())
    with pytest.raises(ValueError, match="exists"):
        execute(prepared, tmp_path)


def test_validation_labels_cannot_change_fit_or_selected_threshold(prepared, tmp_path, monkeypatch):
    first = execute(prepared, tmp_path)
    original = calibration_run.calibration_context
    config = CalibrationConfig.model_validate(yaml.safe_load(prepared[1].read_text()))

    def change_validation_only(*args, **kwargs):
        values = list(original(*args, **kwargs))
        roles = assign_roles(values[-2], config)
        values[-1] = [
            "technical_support" if role == "validation" else target
            for target, role in zip(values[-1], roles, strict=True)
        ]
        return tuple(values)

    monkeypatch.setattr(calibration_run, "calibration_context", change_validation_only)
    second = execute(prepared, tmp_path, "changed-validation")
    for name in ["majority", "tfidf_logistic", "laya"]:
        assert (first.parent / f"{name}_calibrator.json").read_bytes() == (
            second.parent / f"{name}_calibrator.json"
        ).read_bytes()
    for folder in first.parent.iterdir():
        if folder.is_dir():
            assert (folder / "selection.json").read_bytes() == (
                second.parent / folder.name / "selection.json"
            ).read_bytes()
    changed = json.loads((second.parent / "run.json").read_text())
    assert not any(info["enabled"] for info in changed["variants"].values())


def test_corrupt_prediction_cache_is_rejected_before_report(prepared, tmp_path):
    with (tmp_path / "predictions/laya/predictions.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        execute(prepared, tmp_path)
    assert not (tmp_path / "result").exists()


def test_capture_failure_never_creates_completed_manifest(prepared, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected inference failure")

    monkeypatch.setattr(LayaModel, "predict_batch", fail)
    with pytest.raises(RuntimeError, match="injected"):
        capture_calibration(
            prepared[0], ADAPTER, tmp_path / "source", tmp_path / "failed", lambda message: None
        )
    assert (tmp_path / "failed/failed_run.json").exists()
    assert not (tmp_path / "failed/run.json").exists()
