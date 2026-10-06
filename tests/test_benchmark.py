import json
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

import yaml

from rightdesk.config import load_experiment, load_workflow
from rightdesk.data.importers.abcd import checksum
from rightdesk.evaluation.benchmark import benchmark_baselines
from rightdesk.schemas import ReviewedLabel, TicketInput

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def benchmark_config(tmp_path):
    workflow_path = ROOT / "configs/workflows/retail_support.yaml"
    workflow = load_workflow(workflow_path)
    manifest = {
        "schema_version": 1,
        "dataset": "abcd",
        "workflow_id": workflow.workflow_id,
        "workflow_version": workflow.version,
        "queue_mapping_version": "1",
        "mapping_status": "provisional",
        "outputs": {},
        "provenance": {"workflow": {"path": str(workflow_path), "sha256": checksum(workflow_path)}},
        "report": {"partition_counts": {"train": 6, "development": 2}},
    }
    texts = {
        "train": [
            "reset password",
            "forgot username",
            "account locked",
            "refund please",
            "return my product",
            "refund payment",
        ],
        "development": ["forgot my password", "refund the damaged item"],
    }
    for role, messages in texts.items():
        directory = tmp_path / role
        directory.mkdir()
        tickets, labels = [], []
        for index, message in enumerate(messages):
            case_id = f"{role}-{index}"
            ticket = TicketInput(
                case_id=case_id, conversation_id=case_id, text=message, channel="chat"
            )
            label = ReviewedLabel.for_workflow(
                {
                    "case_id": case_id,
                    "workflow_id": workflow.workflow_id,
                    "workflow_version": workflow.version,
                    "eventual_owner_queue": "account"
                    if index < len(messages) / 2
                    else "returns_refunds",
                },
                workflow,
            )
            tickets.append(ticket.model_dump_json())
            labels.append(label.model_dump_json())
        for name, rows in (("tickets.jsonl", tickets), ("labels.jsonl", labels)):
            path = directory / name
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            manifest["outputs"][f"{role}/{name}"] = {
                "path": f"{role}/{name}",
                "sha256": checksum(path),
                "size_bytes": path.stat().st_size,
            }
    (tmp_path / "partitions.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = load_experiment(ROOT / "configs/experiments/baseline.yaml").model_dump()
    config["workflow"] = str(workflow_path)
    config["dataset"].update(manifest="partitions.json", processed_path=".")
    next(model for model in config["models"] if model["id"] == "rules")["rules_path"] = str(
        ROOT / "configs/models/rules_v1.yaml"
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_benchmark_runs_without_any_calibration_or_test_files(benchmark_config, tmp_path):
    report = benchmark_baselines(benchmark_config, tmp_path / "run", bootstrap_samples=20)
    assert report.exists()
    run = json.loads((report.parent / "run.json").read_text())
    assert run["models_run"] == ["majority", "rules", "tfidf_logistic"]
    assert run["models_deferred"] == ["laya"]
    assert run["evaluation_split"] == "development"
    for name, digest in run["artifacts"].items():
        assert checksum(report.parent / name) == digest
    assert (report.parent / "tfidf_logistic/pipeline.joblib").exists()
    with pytest.raises(ValueError, match="exists"):
        benchmark_baselines(benchmark_config, tmp_path / "run", bootstrap_samples=20)


def test_calibration_cannot_be_used_as_benchmark(benchmark_config, tmp_path):
    with pytest.raises(ValueError, match="calibration is reserved"):
        benchmark_baselines(benchmark_config, tmp_path / "run", split="calibration")


def test_corrupt_data_is_rejected_before_fitting(benchmark_config, tmp_path):
    (tmp_path / "train/tickets.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        benchmark_baselines(benchmark_config, tmp_path / "run")
    assert not (tmp_path / "run").exists()
