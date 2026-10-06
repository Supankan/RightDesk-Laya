import gzip
import io
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from rightdesk.cli import app
from rightdesk.config import load_workflow
from rightdesk.data.importers import abcd
from rightdesk.data.mapping import QueueMapping, load_mapping
from rightdesk.data.preprocessing import duplicate_key
from rightdesk.data.splits import assign_partitions, prepare_partitions, verified_intake
from rightdesk.schemas import ReviewedLabel, TicketInput

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "configs/workflows/retail_support.yaml"
MAPPING = ROOT / "configs/mappings/abcd_retail_v1.yaml"


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def record(identifier, text):
        return {
            "convo_id": identifier,
            "original": [["customer", text]],
            "scenario": {"flow": "account_access", "subflow": "recover_password"},
        }

    dataset = {
        "train": [record(i, f"Unique request {i}") for i in range(200)]
        + [record(200, "TEST SHARED"), record(201, "DEV SHARED"), record(202, "Unique request 1")],
        "dev": [record(300, "dev shared"), record(301, "test shared")],
        "test": [record(400, "test shared"), record(401, "TEST   SHARED")],
    }
    archive = gzip.compress(json.dumps(dataset).encode())
    monkeypatch.setattr(
        abcd,
        "urlopen",
        lambda url, timeout: io.BytesIO(archive if url.endswith(".gz") else b"MIT fixture"),
    )
    source = abcd.download_abcd(tmp_path / "data")
    intake = abcd.import_abcd(source, tmp_path / "intake", tmp_path / "intake.json")
    config = tmp_path / "partitions.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": "1",
                "intake_manifest": "intake.json",
                "workflow": str(WORKFLOW),
                "mapping": str(MAPPING),
                "seed": 42,
                "calibration_fraction": 0.15,
                "output_directory": "out",
                "manifest": "partitions.json",
            }
        )
    )
    return config, intake


def test_duplicate_normalization_does_not_remove_punctuation():
    assert duplicate_key(" ＨＥＬＬＯ\tworld ") == duplicate_key("hello WORLD")
    assert duplicate_key("hello!") != duplicate_key("hello")


def test_group_priority_determinism_and_labels_do_not_select_splits(prepared):
    _, intake = prepared
    tickets, refs = verified_intake(intake)
    rows = assign_partitions(tickets, refs, 42, 0.15)
    by_id = {row["case_id"]: row for row in rows}
    assert by_id["abcd:200:intake"]["exclusion_reason"] == "duplicate_of_test"
    assert by_id["abcd:301:intake"]["exclusion_reason"] == "duplicate_of_test"
    assert by_id["abcd:201:intake"]["exclusion_reason"] == "duplicate_of_development"
    assert by_id["abcd:400:intake"]["partition"] == "test"
    assert by_id["abcd:401:intake"]["partition"] == "test"
    assert by_id["abcd:1:intake"]["partition"] == by_id["abcd:202:intake"]["partition"]
    for ref in refs.values():
        ref.flow, ref.subflow = "changed", "labels"
    assert rows == assign_partitions(dict(reversed(list(tickets.items()))), refs, 42, 0.15)
    changed = assign_partitions(tickets, refs, 99, 0.15)
    assert rows != changed
    assert [r for r in rows if r["source_split"] != "train"] == [
        r for r in changed if r["source_split"] != "train"
    ]


def test_end_to_end_partition_artifacts(prepared, monkeypatch, tmp_path):
    config, _ = prepared
    monkeypatch.chdir(tmp_path.parent)
    result = CliRunner().invoke(app, ["data", "prepare-abcd", "--config", str(config)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "partitions.json").read_text())
    report = manifest["report"]
    assert report["official_test_preserved"]
    assert report["excluded_count"] == 3
    assert report["cross_partition_duplicate_groups"] == 0
    assert report["partition_counts"]["test"] == 2
    workflow = load_workflow(WORKFLOW)
    groups = {}
    cases = set()
    for role in ("train", "development", "calibration", "test"):
        inputs = (tmp_path / "out" / role / "tickets.jsonl").read_text().splitlines()
        labels = (tmp_path / "out" / role / "labels.jsonl").read_text().splitlines()
        for raw, target in zip(inputs, labels, strict=True):
            ticket = TicketInput.model_validate_json(raw)
            label = ReviewedLabel.model_validate_json(target, context={"workflow": workflow})
            assert label.case_id == ticket.case_id
            assert label.review_status == "unreviewed"
            assert label.sufficient_information_at_intake is None
            assert label.eventual_owner_queue == "account"
            assert ticket.case_id not in cases
            cases.add(ticket.case_id)
            key = duplicate_key(ticket.text)
            assert groups.setdefault(key, role) == role
    for item in manifest["outputs"].values():
        assert abcd.checksum(tmp_path / item["path"]) == item["sha256"]
    with pytest.raises(abcd.IngestionError, match="exists"):
        prepare_partitions(config)


def test_mapping_overrides_and_unknowns():
    workflow = load_workflow(WORKFLOW)
    mapping = load_mapping(MAPPING, workflow)
    lookup = mapping.lookup()
    assert lookup[("order_issue", "status_mystery_fee")] == "billing_subscriptions"
    assert lookup[("purchase_dispute", "out_of_stock_one_item")] == "product_policy"
    assert lookup[("purchase_dispute", "mistimed_billing_already_returned")] == "returns_refunds"
    assert ("account_access", "new_subflow") not in lookup
    data = mapping.model_dump()
    data["flows"][0]["target_queue"] = "nonexistent"
    with pytest.raises(ValueError, match="unknown target"):
        QueueMapping.model_validate(data).validate_workflow(workflow)
    data = mapping.model_dump()
    data["flows"].append(data["flows"][0])
    with pytest.raises(ValidationError, match="unique"):
        QueueMapping.model_validate(data)


def test_unmapped_labels_reported_without_publishing(prepared, tmp_path):
    config, _ = prepared
    mapping = yaml.safe_load(MAPPING.read_text())
    mapping["flows"] = [row for row in mapping["flows"] if row["flow"] != "account_access"]
    (tmp_path / "mapping.yaml").write_text(yaml.safe_dump(mapping))
    settings = yaml.safe_load(config.read_text())
    settings["mapping"] = "mapping.yaml"
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(abcd.IngestionError, match="account_access/recover_password"):
        prepare_partitions(config)
    assert not (tmp_path / "out").exists()


def test_corrupt_intake_rejected(prepared, tmp_path):
    config, _ = prepared
    with (tmp_path / "intake/tickets.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(abcd.IngestionError, match="checksum"):
        prepare_partitions(config)


def test_missing_join_and_conversation_collisions(prepared, tmp_path):
    _, intake = prepared
    metadata = json.loads(intake.read_text())
    path = tmp_path / "intake/references.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row["conversation_id"] = "wrong"
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n")
    metadata["outputs"]["references.jsonl"].update(
        sha256=abcd.checksum(path), size_bytes=path.stat().st_size
    )
    intake.write_text(json.dumps(metadata))
    with pytest.raises(abcd.IngestionError, match="conversation mismatch"):
        verified_intake(intake)
