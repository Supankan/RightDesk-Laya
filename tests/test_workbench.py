"""Stored experiments and annotations remain reproducible, isolated, and transactional."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from typer.testing import CliRunner

from rightdesk.cli import app
from rightdesk.config import load_workflow
from rightdesk.data.importers.abcd import checksum
from rightdesk.data.preprocessing import duplicate_key
from rightdesk.evaluation.metrics import evaluate
from rightdesk.exports import export_run
from rightdesk.policy.routing import RoutingPolicy, RoutingSettings
from rightdesk.review import (
    QueueRequest,
    ReviewSubmission,
    agreement,
    blind_case,
    create_queue,
    export_reviews,
    queue_items,
    review_progress,
    submit_review,
)
from rightdesk.schemas import ModelPrediction, ReviewedLabel, TicketInput, WorkflowConfig
from rightdesk.services import (
    import_dataset,
    model_summaries,
    prediction_rows,
    register_run,
    reviewed_evaluation,
    verify_run,
)
from rightdesk.storage import Store, encode

ROOT = Path(__file__).resolve().parents[1]


def write_lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(encode(value) + "\n" for value in values), encoding="utf-8")


def finish_run(directory, run):
    run["artifacts"] = {
        p.relative_to(directory).as_posix(): checksum(p)
        for p in directory.rglob("*")
        if p.is_file() and p.name != "run.json"
    }
    (directory / "run.json").write_text(encode(run), encoding="utf-8")


@pytest.fixture
def workbench(tmp_path):
    workflow_path = ROOT / "configs/workflows/retail_support.yaml"
    workflow = load_workflow(workflow_path)
    texts = [
        "password reset",
        "refund request",
        "hello",
        " HELLO ",
        "checkout timeout",
        "billing charge",
    ]
    targets = [
        "account",
        "returns_refunds",
        "account",
        "returns_refunds",
        "technical_support",
        "billing_subscriptions",
    ]
    manifest = {
        "schema_version": 1,
        "dataset": "abcd",
        "workflow_id": workflow.workflow_id,
        "workflow_version": workflow.version,
        "provenance": {"workflow": {"path": str(workflow_path), "sha256": checksum(workflow_path)}},
        "report": {"partition_counts": {"development": 6, "calibration": 3}},
        "outputs": {"test/tickets.jsonl": {"path": "TEST_MUST_NEVER_BE_READ"}},
    }
    tickets_by_role = {}
    for role, messages in [
        ("development", texts),
        ("calibration", ["cal zero", "cal one", "cal two"]),
    ]:
        tickets = [
            TicketInput(
                case_id=f"{role}-{i}",
                conversation_id=f"{role}-{i}",
                text=text,
                channel="chat",
                language="en",
            )
            for i, text in enumerate(messages)
        ]
        labels = [
            ReviewedLabel.for_workflow(
                {
                    "case_id": ticket.case_id,
                    "workflow_id": workflow.workflow_id,
                    "workflow_version": workflow.version,
                    "eventual_owner_queue": targets[i] if role == "development" else "account",
                },
                workflow,
            )
            for i, ticket in enumerate(tickets)
        ]
        tickets_by_role[role] = tickets
        for name, rows in [("tickets.jsonl", tickets), ("labels.jsonl", labels)]:
            path = tmp_path / role / name
            write_lines(path, [row.model_dump() for row in rows])
            manifest["outputs"][f"{role}/{name}"] = {
                "path": f"{role}/{name}",
                "sha256": checksum(path),
                "size_bytes": path.stat().st_size,
            }
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(encode(manifest), encoding="utf-8")
    store = Store(tmp_path / "workbench.sqlite3")
    store.initialize()
    dataset_id = import_dataset(store, manifest_path, workflow_path)
    source = tmp_path / "benchmark"
    source.mkdir()
    metrics = {}
    for name in ("tfidf_logistic", "laya"):
        rows = []
        for i, ticket in enumerate(tickets_by_role["development"]):
            chosen = targets[i] if i != 3 else "account"
            if name == "laya" and i == 2:
                chosen = "returns_refunds"
            score = 0.3 if i in (2, 3) else 0.9
            row = ModelPrediction.for_workflow(
                {
                    "case_id": ticket.case_id,
                    "workflow_id": workflow.workflow_id,
                    "workflow_version": workflow.version,
                    "model_revision": name + "-test-v1",
                    "queue_probabilities": {
                        q: score if q == chosen else (1 - score) / 5 for q in workflow.queue_ids
                    },
                },
                workflow,
            )
            rows.append(row)
        folder = source / name
        write_lines(
            folder / "predictions.jsonl",
            [{"prediction": r.model_dump(), "latency_ms": 1.0} for r in rows],
        )
        metrics[name] = evaluate(
            rows,
            targets,
            [duplicate_key(t.text) for t in tickets_by_role["development"]],
            sorted(workflow.queue_ids),
            [1.0] * 6,
            bootstrap_samples=20,
        )
    base = {
        "schema_version": 1,
        "dataset_manifest_sha256": dataset_id,
        "workflow": workflow.model_dump(),
        "created_at": "2026-10-21T00:00:00+00:00",
        "evaluation_split": "development",
        "models_run": list(metrics),
        "metrics": metrics,
    }
    finish_run(source, base)
    run_id = register_run(store, source)
    # A minimal genuine-contract Phase 7 snapshot with a disabled policy.
    calibration_dir = tmp_path / "calibration-run"
    folder = calibration_dir / "tfidf_temperature"
    folder.mkdir(parents=True)
    data = workflow.model_dump()
    data["policy"].update(
        version="test-policy", routing_threshold=0.7, calibration_version="test-calibration"
    )
    bound = WorkflowConfig.model_validate(data)
    settings = RoutingSettings(
        model_revision="tfidf-test", calibration_version="test-calibration", enabled=False
    )
    ticket = tickets_by_role["calibration"][2]
    p = ModelPrediction.for_workflow(
        {
            "case_id": ticket.case_id,
            "workflow_id": workflow.workflow_id,
            "workflow_version": workflow.version,
            "model_revision": "tfidf-test",
            "calibration_version": "test-calibration",
            "queue_probabilities": {q: 0.9 if q == "account" else 0.02 for q in workflow.queue_ids},
        },
        bound,
    )
    decision = RoutingPolicy(bound, settings).decide(ticket, p)
    quality = evaluate(
        [p], ["account"], ["group"], sorted(workflow.queue_ids), [0.0], bootstrap_samples=20
    )
    point = {"auto_count": 1, "automation_coverage": 1.0, "routing_error": 0.0}
    actual = {"auto_count": 0, "automation_coverage": 0.0, "routing_error": None}
    variant = {
        "quality": quality,
        "validation": point,
        "enabled_policy_validation": actual,
        "enabled": False,
        "threshold": 0.7,
    }
    (folder / "policy.json").write_text(
        encode({"workflow": bound.model_dump(), "settings": settings.model_dump()})
    )
    (folder / "metrics.json").write_text(encode(variant))
    write_lines(folder / "validation_predictions.jsonl", [p.model_dump()])
    write_lines(folder / "decisions.jsonl", [decision.model_dump()])
    write_lines(
        calibration_dir / "assignments.jsonl",
        [
            {"case_id": t.case_id, "role": role}
            for t, role in zip(
                tickets_by_role["calibration"], ["fit", "selection", "validation"], strict=True
            )
        ],
    )
    finish_run(
        calibration_dir,
        {
            "schema_version": 1,
            "phase": 7,
            "workflow": workflow.model_dump(),
            "dataset_manifest_sha256": dataset_id,
            "role_counts": {"validation": 1},
            "created_at": "2026-10-21T01:00:00+00:00",
            "variants": {"tfidf_temperature": variant},
        },
    )
    calibration_id = register_run(store, calibration_dir)
    return {
        "store": store,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "calibration_id": calibration_id,
        "source": source,
        "manifest": manifest_path,
        "workflow": workflow_path,
    }


def submission(workbench, reviewer="alice", case="development-0", revision=0, **values):
    return ReviewSubmission(
        dataset_id=workbench["dataset_id"],
        case_id=case,
        reviewer_id=reviewer,
        expected_revision=revision,
        **values,
    )


def test_catalog_idempotence_and_sealed_test_partition(workbench):
    store = workbench["store"]
    assert len(store.list_runs()) == 2
    assert store.query("SELECT COUNT(*) n FROM cases")[0]["n"] == 9
    assert register_run(store, workbench["source"]) == workbench["run_id"]
    assert (
        import_dataset(store, workbench["manifest"], workbench["workflow"])
        == workbench["dataset_id"]
    )
    assert len(store.list_runs()) == 2
    assert {
        r["partition_role"] for r in store.query("SELECT DISTINCT partition_role FROM cases")
    } == {"development", "calibration"}
    assert verify_run(store, workbench["run_id"])["status"] == "verified"


def test_candidate_and_enabled_policy_coverage_are_distinct(workbench):
    summary = model_summaries(workbench["store"], workbench["calibration_id"])[0]
    assert summary["candidate_coverage"] == 1
    assert summary["coverage"] == 0
    assert summary["gate_passed"] is False
    rows = prediction_rows(workbench["store"], workbench["calibration_id"], "tfidf_temperature")
    assert rows[0]["action"] == "review"


def test_corrupt_source_rejected_and_cached_data_survives_source_loss(workbench):
    store = workbench["store"]
    path = workbench["source"] / "run.json"
    original = path.read_bytes()
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        verify_run(store, workbench["run_id"])
    path.write_bytes(original)
    path.rename(path.with_name("source-unavailable.json"))
    assert verify_run(store, workbench["run_id"])["status"] == "source_missing"
    assert len(prediction_rows(store, workbench["run_id"], "laya")) == 6


def test_failed_dataset_import_leaves_no_partial_rows(workbench, tmp_path):
    manifest = json.loads(workbench["manifest"].read_text())
    path = workbench["manifest"].parent / "calibration/labels.jsonl"
    path.write_text("{}\n")
    store = Store(tmp_path / "empty.sqlite3")
    store.initialize()
    with pytest.raises(ValueError, match="checksum"):
        import_dataset(store, workbench["manifest"], workbench["workflow"])
    assert store.query("SELECT COUNT(*) n FROM datasets")[0]["n"] == 0
    assert manifest["outputs"]


def test_unknown_negative_revision_history_and_blind_boundaries(workbench):
    store = workbench["store"]
    before = blind_case(store, workbench["dataset_id"], "development-0", "alice")
    assert set(before) == {"ticket", "partition", "own_review"}
    assert before["own_review"] is None
    request = submission(workbench, sufficient_information_at_intake=False)
    assert submit_review(store, request) == 1
    own = blind_case(store, workbench["dataset_id"], "development-0", "alice")["own_review"]
    assert own["label"]["acceptable_queues_at_intake"] is None
    assert own["label"]["sufficient_information_at_intake"] is False
    assert (
        submit_review(store, submission(workbench, revision=1, acceptable_queues_at_intake=[])) == 2
    )
    assert len(store.latest_reviews(workbench["dataset_id"])) == 1
    history = [
        json.loads(line)
        for line in export_reviews(store, workbench["dataset_id"], True).splitlines()
    ]
    assert len(history) == 2
    assert history[0]["label"]["acceptable_queues_at_intake"] is None
    assert history[1]["label"]["acceptable_queues_at_intake"] == []
    with pytest.raises(ValueError, match="another session"):
        submit_review(store, request)
    with store.connection(write=True) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM reviews")


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"acceptable_queues_at_intake": ["unknown_queue"]},
        {"useful_question_ids": ["invented"]},
        {"acceptable_queues_at_intake": ["account", "account"]},
    ],
)
def test_invalid_annotations_never_enter_store(workbench, values):
    with pytest.raises(ValueError):
        submit_review(workbench["store"], submission(workbench, **values))
    assert not workbench["store"].latest_reviews(workbench["dataset_id"])


def test_concurrent_submissions_cannot_overwrite(workbench):
    request = submission(workbench, sufficient_information_at_intake=True)

    def save():
        try:
            return submit_review(workbench["store"], request)
        except ValueError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(list(pool.map(lambda _: save(), range(2))), key=str) == [1, "conflict"]
    assert len(workbench["store"].latest_reviews(workbench["dataset_id"])) == 1


def test_queue_sampling_is_deterministic_deduplicated_and_reviewer_specific(workbench):
    store = workbench["store"]
    request = QueueRequest(title="Random", dataset_id=workbench["dataset_id"], limit=50)
    queue = create_queue(store, request)
    assert create_queue(store, request) == queue
    assert len(queue_items(store, queue, "alice")) == 5  # Two greetings are one normalized group.
    case = queue_items(store, queue, "alice")[0]["case_id"]
    submit_review(store, submission(workbench, case=case, sufficient_information_at_intake=False))
    assert review_progress(store, queue, "alice")["completed"] == 1
    assert review_progress(store, queue, "bob")["completed"] == 0
    disagreement = create_queue(
        store,
        QueueRequest(
            title="Disagreements",
            dataset_id=workbench["dataset_id"],
            strategy="disagreement",
            run_id=workbench["run_id"],
            model_id="tfidf_logistic",
            comparison_model="laya",
        ),
    )
    assert [r["case_id"] for r in queue_items(store, disagreement, "alice")] == ["development-2"]


def test_agreement_excludes_unknowns_and_does_not_invent_consensus(workbench):
    store = workbench["store"]
    for i, (left, right) in enumerate([(True, True), (False, True), (True, False), (None, None)]):
        for reviewer, suff in [("alice", left), ("bob", right)]:
            submit_review(
                store,
                submission(
                    workbench,
                    reviewer=reviewer,
                    case=f"development-{i}",
                    sufficient_information_at_intake=suff,
                    acceptable_queues_at_intake=[] if i == 3 else ["account"],
                ),
            )
    pair = agreement(store, workbench["dataset_id"])["pairs"][0]
    assert pair["sufficiency_pairs"] == 3
    assert pair["sufficiency_agreement"] == pytest.approx(1 / 3)
    assert pair["sufficiency_kappa"] == pytest.approx(-0.5)
    assert pair["queue_pairs"] == 4
    assert pair["mean_queue_jaccard"] == 1
    evaluated = reviewed_evaluation(store, workbench["run_id"], "tfidf_logistic")
    assert evaluated["individual_judgments"] == 8
    assert evaluated["distinct_reviewed_cases"] == 4
    assert evaluated["acceptable_queue_rate"] == 0.5
    original = json.loads(
        store.one(
            "SELECT label_json FROM reference_labels WHERE dataset_id=? AND case_id=?",
            (workbench["dataset_id"], "development-0"),
        )["label_json"]
    )
    assert original["review_status"] == "unreviewed"


def test_parquet_snapshot_roundtrip_hashes_and_separate_labels(workbench, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    output = tmp_path / "export"
    manifest = export_run(workbench["store"], workbench["run_id"], output)
    payload = json.loads(manifest.read_text())
    for name, digest in payload["artifacts"].items():
        assert checksum(output / name) == digest
    predictions = pq.read_table(output / "predictions.parquet")
    assert predictions.num_rows == 12
    assert "eventual_owner_queue" not in predictions.column_names
    values = predictions.to_pylist()[0]["queue_probabilities"]
    assert sum(value for _, value in values) == pytest.approx(1)
    assert pq.read_table(output / "tickets.parquet").num_rows == 6
    assert pq.read_table(output / "reference_labels.parquet").num_rows == 6
    with pytest.raises(ValueError, match="exists"):
        export_run(workbench["store"], workbench["run_id"], output)


def test_unrelated_database_is_not_adopted_and_cli_lists_catalog(workbench, tmp_path):
    path = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE other(value TEXT)")
    with pytest.raises(ValueError, match="unrelated"):
        Store(path).initialize()
    result = CliRunner().invoke(
        app, ["store", "list-runs", "--database", str(workbench["store"].path)]
    )
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.output)) == 2
