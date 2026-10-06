"""Use synthetic source-shaped fixtures, never the downloaded test split."""

import gzip
import io
import json

import pytest
from typer.testing import CliRunner

from rightdesk.cli import app
from rightdesk.data.importers import abcd
from rightdesk.schemas import TicketInput


def conversation(identifier=1):
    return {
        "convo_id": identifier,
        "scenario": {"flow": "hidden_flow", "subflow": "hidden_subflow", "personal": "SECRET"},
        "original": [
            ["agent", "Welcome"],
            ["customer", "  Hello!\n"],
            ["customer", "LATER_SECRET"],
            ["agent", "AGENT_SECRET"],
        ],
        "delexed": [{"text": "DELEXED_SECRET", "targets": ["TARGET_SECRET"]}],
    }


@pytest.fixture
def dataset():
    return {split: [conversation(index)] for index, split in enumerate(abcd.SPLITS)}


def download_fixture(tmp_path, monkeypatch, dataset):
    archive = gzip.compress(json.dumps(dataset).encode())

    def fetch(url, timeout):
        assert abcd.REVISION in url
        return io.BytesIO(archive if url.endswith(".gz") else b"MIT fixture license")

    monkeypatch.setattr(abcd, "urlopen", fetch)
    return abcd.download_abcd(tmp_path / "data")


def test_separation_provenance_and_official_splits(tmp_path, monkeypatch, dataset):
    source = download_fixture(tmp_path, monkeypatch, dataset)
    output, manifest = tmp_path / "prepared", tmp_path / "prepared.json"

    def no_network(*args, **kwargs):
        raise AssertionError("Import must not access network")

    monkeypatch.setattr(abcd, "urlopen", no_network)
    monkeypatch.chdir(tmp_path.parent)
    abcd.import_abcd(source, output, manifest)
    tickets = [
        TicketInput.model_validate_json(line)
        for line in (output / "tickets.jsonl").read_text().splitlines()
    ]
    references = [
        json.loads(line) for line in (output / "references.jsonl").read_text().splitlines()
    ]
    assert len(tickets) == 3
    assert all(ticket.text == "  Hello!\n" for ticket in tickets)
    assert "SECRET" not in (output / "tickets.jsonl").read_text()
    assert "hidden_flow" not in (output / "tickets.jsonl").read_text()
    assert [ref["source_split"] for ref in references] == list(abcd.SPLITS)
    assert all(ref["intake_turn_index"] == 1 for ref in references)
    assert [ref["case_id"] for ref in references] == [ticket.case_id for ticket in tickets]
    assert all(set(ref) == set(abcd.SourceReference.model_fields) for ref in references)
    metadata = json.loads(manifest.read_text())
    assert metadata["queue_mapping_version"] is None
    assert metadata["calibration_split"] is None
    for item in metadata["outputs"].values():
        path = manifest.parent / item["path"]
        assert abcd.checksum(path) == item["sha256"]
        assert path.stat().st_size == item["size_bytes"]
    assert abcd.checksum(source) == metadata["source_manifest_sha256"]


@pytest.mark.parametrize(
    "turns,reason",
    [
        ([["customer", " "], ["customer", "later useful message"]], "empty_or_invalid"),
        ([["agent", "hello"]], "no_customer"),
        ([[[], "bad"]], "invalid_turn"),
        (None, "missing_original"),
    ],
)
def test_bad_intake_is_reported_not_replaced(tmp_path, monkeypatch, dataset, turns, reason):
    dataset["train"][0]["original"] = turns
    source = download_fixture(tmp_path, monkeypatch, dataset)
    abcd.import_abcd(source, tmp_path / "out", tmp_path / "manifest.json")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["counts"]["train"] == {"source": 1, "accepted": 0, "rejected": 1}
    rejected = json.loads((tmp_path / "out/rejections.jsonl").read_text())
    assert reason in rejected["reason"]
    assert rejected["source_index"] == 0


def test_duplicate_conversations_fail_without_publishing(tmp_path, monkeypatch, dataset):
    dataset["test"][0]["convo_id"] = dataset["train"][0]["convo_id"]
    source = download_fixture(tmp_path, monkeypatch, dataset)
    with pytest.raises(abcd.IngestionError, match="Duplicate"):
        abcd.import_abcd(source, tmp_path / "out", tmp_path / "manifest.json")
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "manifest.json").exists()
    assert not list(tmp_path.glob(".abcd-import-*"))


def test_download_reuse_and_corruption(tmp_path, monkeypatch, dataset):
    source = download_fixture(tmp_path, monkeypatch, dataset)

    def no_network(*args, **kwargs):
        raise AssertionError("Cached download must not access network")

    monkeypatch.setattr(abcd, "urlopen", no_network)
    assert abcd.download_abcd(tmp_path / "data") == source
    metadata = abcd.load_source_manifest(source)
    (source.parent / metadata.archive.path).write_bytes(b"corrupted")
    with pytest.raises(abcd.IngestionError, match="checksum"):
        abcd.download_abcd(tmp_path / "data")
    with pytest.raises(abcd.IngestionError, match="checksum"):
        abcd.import_abcd(source, tmp_path / "out", tmp_path / "manifest.json")


def test_download_failure_does_not_publish(tmp_path, monkeypatch):
    def fail(url, timeout):
        raise OSError("network unavailable")

    monkeypatch.setattr(abcd, "urlopen", fail)
    with pytest.raises(OSError):
        abcd.download_abcd(tmp_path)
    assert not list((tmp_path / "manifests").glob("*.json"))
    assert not (tmp_path / "raw/abcd" / abcd.REVISION).exists()
    assert not list((tmp_path / "raw/abcd").glob(".abcd-download-*"))


def test_cli_and_existing_output_protection(tmp_path, monkeypatch, dataset):
    source = download_fixture(tmp_path, monkeypatch, dataset)
    output, manifest = tmp_path / "out", tmp_path / "manifest.json"
    args = [
        "data",
        "import-abcd",
        "--source-manifest",
        str(source),
        "--output-dir",
        str(output),
        "--manifest",
        str(manifest),
    ]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    before = manifest.read_bytes()
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "exists" in result.output
    assert before == manifest.read_bytes()


@pytest.mark.parametrize("shape", [[], {}, {"train": [], "dev": [], "test": []}])
def test_unexpected_dataset_shape(tmp_path, monkeypatch, shape):
    source = download_fixture(tmp_path, monkeypatch, shape)
    with pytest.raises(abcd.IngestionError):
        abcd.import_abcd(source, tmp_path / "out", tmp_path / "manifest.json")


def test_pin_required(tmp_path):
    with pytest.raises(abcd.IngestionError, match="commit SHA"):
        abcd.download_abcd(tmp_path, "master")


def test_future_turns_never_affect_intake():
    record = conversation()
    expected = abcd.extract_case(record, "train", 0)
    record["original"][2:] = [None, {"untrusted": "future"}]
    record["delexed"] = "irrelevant"
    record["scenario"]["personal"] = {"more": "hidden data"}
    assert abcd.extract_case(record, "train", 0) == expected
