"""Shared workbench operations. This module never imports or invokes a model backend."""

import json
import math
from pathlib import Path

from rightdesk.config import load_workflow, resolve_config_path
from rightdesk.data.importers.abcd import checksum
from rightdesk.data.preprocessing import duplicate_key
from rightdesk.policy.routing import RoutingPolicy, RoutingSettings
from rightdesk.schemas import (
    ModelPrediction,
    PolicyDecision,
    ReviewedLabel,
    TicketInput,
    WorkflowConfig,
)
from rightdesk.storage import Store, encode, utc_now


def checked_artifacts(directory: Path) -> dict:
    directory = directory.resolve()
    manifest_path = directory / "run.json"
    run = json.loads(manifest_path.read_text(encoding="utf-8"))
    if run.get("schema_version") != 1 or not run.get("artifacts"):
        raise ValueError("Expected a completed run manifest with artifact hashes")
    if (directory / "failed_run.json").exists():
        raise ValueError("Failed runs cannot be registered")
    for name, digest in run["artifacts"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or checksum(path) != digest:
            raise ValueError(f"Run artifact checksum/path mismatch: {name}")
    return run


def import_dataset(store: Store, manifest_path: Path, workflow_path: Path) -> str:
    """Import only development/calibration intake and separate proxy labels, atomically."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workflow = load_workflow(workflow_path)
    if (manifest.get("schema_version"), manifest.get("dataset")) != (1, "abcd"):
        raise ValueError("Expected an ABCD partition manifest v1")
    if (manifest["workflow_id"], manifest["workflow_version"]) != (
        workflow.workflow_id,
        workflow.version,
    ):
        raise ValueError("Dataset/workflow mismatch")
    if checksum(workflow_path) != manifest["provenance"]["workflow"]["sha256"]:
        raise ValueError("Dataset workflow checksum mismatch")
    for entry in manifest["provenance"].values():
        if checksum(resolve_config_path(manifest_path, entry["path"])) != entry["sha256"]:
            raise ValueError("Dataset provenance checksum mismatch")
    dataset_id = checksum(manifest_path)
    cases, labels, seen_groups, seen_conversations = [], [], {}, set()
    for role in ("development", "calibration"):
        tables = {}
        for name, schema in (("tickets.jsonl", TicketInput), ("labels.jsonl", ReviewedLabel)):
            entry = manifest["outputs"][f"{role}/{name}"]
            path = resolve_config_path(manifest_path, entry["path"])
            if checksum(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"Dataset checksum mismatch: {role}/{name}")
            rows = [
                schema.model_validate_json(line, context={"workflow": workflow})
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            indexed = {row.case_id: row for row in rows}
            if len(indexed) != len(rows):
                raise ValueError("Duplicate dataset case ID")
            tables[name] = indexed
        tickets, references = tables["tickets.jsonl"], tables["labels.jsonl"]
        if (
            tickets.keys() != references.keys()
            or len(tickets) != manifest["report"]["partition_counts"][role]
        ):
            raise ValueError("Incomplete dataset ticket/reference join")
        for case_id, ticket in tickets.items():
            reference = references[case_id]
            if reference.review_status != "unreviewed":
                raise ValueError("Source references must remain separate from human reviews")
            if reference.eventual_owner_queue is None:
                raise ValueError("ABCD source reference must identify an eventual owner")
            group = duplicate_key(ticket.text)
            if group in seen_groups and seen_groups[group] != role:
                raise ValueError("Development/calibration duplicate-group overlap")
            if ticket.conversation_id in seen_conversations:
                raise ValueError("Duplicate conversation in imported dataset")
            seen_groups[group] = role
            seen_conversations.add(ticket.conversation_id)
            cases.append((dataset_id, case_id, role, group, encode(ticket.model_dump())))
            labels.append((dataset_id, case_id, encode(reference.model_dump())))
    with store.connection(write=True) as connection:
        if connection.execute(
            "SELECT 1 FROM datasets WHERE dataset_id=?", (dataset_id,)
        ).fetchone():
            return dataset_id
        connection.execute(
            "INSERT INTO datasets VALUES(?,?,?,?,?)",
            (
                dataset_id,
                encode(manifest),
                encode(workflow.model_dump()),
                str(manifest_path),
                utc_now(),
            ),
        )
        connection.executemany("INSERT INTO cases VALUES(?,?,?,?,?)", cases)
        connection.executemany("INSERT INTO reference_labels VALUES(?,?,?)", labels)
    return dataset_id


def _required(run, name):
    if name not in run["artifacts"]:
        raise ValueError(f"Required artifact is absent from manifest: {name}")


def register_run(store: Store, directory: Path) -> str:
    directory = directory.resolve()
    run = checked_artifacts(directory)
    source_sha = checksum(directory / "run.json")
    run_id = "run-" + source_sha[:20]
    dataset_id = run["dataset_manifest_sha256"]
    dataset = store.dataset(dataset_id)
    base_workflow = WorkflowConfig.model_validate(dataset["workflow"])
    if run["workflow"] != base_workflow.model_dump():
        raise ValueError("Run/dataset workflow mismatch")
    calibration = run.get("phase") == 7
    if calibration:
        names = list(run["variants"])
        split, kind = "calibration_validation", "calibration"
        _required(run, "assignments.jsonl")
        assignments = [
            json.loads(line) for line in (directory / "assignments.jsonl").read_text().splitlines()
        ]
        expected_ids = [row["case_id"] for row in assignments if row["role"] == "validation"]
        if (
            len(expected_ids) != len(set(expected_ids))
            or len(expected_ids) != run["role_counts"]["validation"]
        ):
            raise ValueError("Invalid calibration validation membership")
    else:
        if "metrics" not in run or run.get("evaluation_split") != "development":
            raise ValueError(
                "Register a development benchmark or Phase 7 validation run; final test is sealed"
            )
        names = list(run["models_run"])
        split, kind = "development", "benchmark"
        expected_ids = None
    cases = {
        row["case_id"]: row
        for row in store.query(
            """SELECT c.*,r.label_json FROM cases c JOIN reference_labels r
           USING(dataset_id,case_id) WHERE c.dataset_id=?""",
            (dataset_id,),
        )
    }
    model_rows, prediction_rows = [], []
    for name in names:
        workflow, policy = base_workflow, None
        if calibration:
            for filename in (
                "policy.json",
                "validation_predictions.jsonl",
                "decisions.jsonl",
                "metrics.json",
            ):
                _required(run, f"{name}/{filename}")
            policy = json.loads((directory / name / "policy.json").read_text())
            workflow = WorkflowConfig.model_validate(policy["workflow"])
            routing = RoutingPolicy(workflow, RoutingSettings.model_validate(policy["settings"]))
            metrics = run["variants"][name]
            if json.loads((directory / name / "metrics.json").read_text()) != metrics:
                raise ValueError("Variant metrics differ from run manifest")
            path = directory / name / "validation_predictions.jsonl"
            predictions = [
                ModelPrediction.model_validate_json(line, context={"workflow": workflow})
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            decisions = [
                PolicyDecision.model_validate_json(line, context={"workflow": workflow})
                for line in (directory / name / "decisions.jsonl").read_text().splitlines()
            ]
            latencies = [None] * len(predictions)
            if (
                len(decisions) != len(predictions)
                or metrics["enabled"] != policy["settings"]["enabled"]
            ):
                raise ValueError("Incomplete decisions or inconsistent policy gate")
        else:
            _required(run, f"{name}/predictions.jsonl")
            metrics = run["metrics"][name]
            wrapped = [
                json.loads(line)
                for line in (directory / name / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            predictions = [
                ModelPrediction.for_workflow(row["prediction"], workflow) for row in wrapped
            ]
            latencies = [row.get("latency_ms") for row in wrapped]
            if any(
                value is not None and (not math.isfinite(value) or value < 0) for value in latencies
            ):
                raise ValueError("Invalid recorded latency")
            decisions = [None] * len(predictions)
        ids = [row.case_id for row in predictions]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Empty or duplicate prediction IDs")
        if expected_ids is None:
            expected_ids = sorted(
                case_id for case_id, row in cases.items() if row["partition_role"] == "development"
            )
        if ids != expected_ids:
            raise ValueError("Run prediction order/membership differs from evaluation partition")
        correct = 0
        for prediction, latency, decision in zip(predictions, latencies, decisions, strict=True):
            case = cases.get(prediction.case_id)
            expected_role = "calibration" if calibration else "development"
            if case is None or case["partition_role"] != expected_role:
                raise ValueError("Prediction case has the wrong dataset role")
            target = json.loads(case["label_json"])["eventual_owner_queue"]
            correct += prediction.selected_queue() == target
            if decision is not None:
                expected = routing.decide(
                    TicketInput.model_validate_json(case["ticket_json"]), prediction
                )
                if expected.model_dump() != decision.model_dump():
                    raise ValueError("Saved decision differs from frozen policy")
            prediction_rows.append(
                (
                    run_id,
                    name,
                    dataset_id,
                    prediction.case_id,
                    encode(prediction.model_dump()),
                    latency,
                    encode(decision.model_dump()) if decision else None,
                )
            )
        quality = metrics["quality"] if calibration else metrics
        if quality["cases"] != len(ids) or not math.isclose(
            quality["accuracy"], correct / len(ids), abs_tol=1e-12
        ):
            raise ValueError("Saved metrics do not match prediction/reference records")
        if calibration:
            actual = metrics["enabled_policy_validation"]
            if actual["auto_count"] != sum(d.action == "route" for d in decisions):
                raise ValueError("Enabled policy coverage differs from decisions")
        model_rows.append(
            (
                run_id,
                name,
                encode(metrics),
                encode(workflow.model_dump()),
                encode(policy) if policy else None,
            )
        )
    with store.connection(write=True) as connection:
        if connection.execute("SELECT 1 FROM runs WHERE source_sha256=?", (source_sha,)).fetchone():
            return run_id
        connection.execute(
            "INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                dataset_id,
                source_sha,
                str(directory),
                kind,
                split,
                run["created_at"],
                utc_now(),
                encode(run),
            ),
        )
        connection.executemany("INSERT INTO models VALUES(?,?,?,?,?)", model_rows)
        connection.executemany("INSERT INTO predictions VALUES(?,?,?,?,?,?,?)", prediction_rows)
    return run_id


def verify_run(store: Store, run_id: str) -> dict:
    saved = store.run(run_id)
    directory = Path(saved["source_path"])
    if not (directory / "run.json").exists():
        return {"run_id": run_id, "status": "source_missing", "cached_records_available": True}
    if checksum(directory / "run.json") != saved["source_sha256"]:
        raise ValueError("Source run manifest changed after registration")
    checked_artifacts(directory)
    return {"run_id": run_id, "status": "verified", "cached_records_available": True}


def model_summaries(store: Store, run_id: str) -> list[dict]:
    summaries = []
    for model in store.models(run_id):
        metrics = model["metrics"]
        calibration = "quality" in metrics
        quality = metrics["quality"] if calibration else metrics
        candidate = metrics["validation"] if calibration else None
        actual = metrics["enabled_policy_validation"] if calibration else quality["operating_point"]
        summaries.append(
            {
                "model": model["model_id"],
                "cases": quality["cases"],
                "accuracy": quality["accuracy"],
                "macro_f1": quality["macro_f1"],
                "brier": quality["brier_score"],
                "ece": quality["calibration_error"],
                "coverage": actual["automation_coverage"],
                "routing_error": actual["routing_error"],
                "candidate_coverage": candidate["automation_coverage"] if candidate else None,
                "candidate_error": candidate["routing_error"] if candidate else None,
                "gate_passed": metrics["enabled"] if calibration else None,
                "threshold": metrics["threshold"] if calibration else None,
                "group_error_upper": candidate.get("group_error_upper") if candidate else None,
                "coverage_kind": "enabled shadow policy"
                if calibration
                else "model prediction coverage",
            }
        )
    return summaries


def prediction_rows(store: Store, run_id: str, model_id: str) -> list[dict]:
    rows = store.query(
        """SELECT p.*,c.ticket_json,c.partition_role,c.group_key,r.label_json
        FROM predictions p JOIN cases c USING(dataset_id,case_id)
        JOIN reference_labels r USING(dataset_id,case_id)
        WHERE p.run_id=? AND p.model_id=? ORDER BY p.case_id""",
        (run_id, model_id),
    )
    result = []
    for row in rows:
        prediction = json.loads(row["prediction_json"])
        ticket, reference = json.loads(row["ticket_json"]), json.loads(row["label_json"])
        decision = json.loads(row["decision_json"]) if row["decision_json"] else None
        chosen = prediction["predicted_queue"]
        probabilities = prediction["queue_probabilities"]
        if chosen is None and probabilities is not None:
            chosen = min(probabilities, key=lambda q: (-probabilities[q], q))
        result.append(
            {
                "case_id": row["case_id"],
                "text": ticket["text"],
                "partition": row["partition_role"],
                "predicted_queue": chosen,
                "reference_queue": reference["eventual_owner_queue"],
                "correct": chosen == reference["eventual_owner_queue"],
                "confidence": max(probabilities.values()) if probabilities is not None else None,
                "action": decision["action"] if decision else "model prediction",
                "latency_ms": row["latency_ms"],
                "prediction": prediction,
                "decision": decision,
            }
        )
    return result


def reviewed_evaluation(store: Store, run_id: str, model_id: str) -> dict:
    """Descriptive assessment of latest individual judgments; no invented consensus."""
    run = store.run(run_id)
    predictions = {row["case_id"]: row for row in prediction_rows(store, run_id, model_id)}
    judgments = []
    for review in store.latest_reviews(run["dataset_id"]):
        if review["case_id"] in predictions:
            judgments.append((json.loads(review["label_json"]), predictions[review["case_id"]]))
    known = [
        (label, row) for label, row in judgments if label["acceptable_queues_at_intake"] is not None
    ]
    hits = sum(
        row["predicted_queue"] in label["acceptable_queues_at_intake"] for label, row in known
    )
    routed = [(label, row) for label, row in known if row["decision"] and row["action"] == "route"]
    return {
        "individual_judgments": len(judgments),
        "distinct_reviewed_cases": len({label["case_id"] for label, _ in judgments}),
        "queue_judgments": len(known),
        "acceptable_queue_rate": hits / len(known) if known else None,
        "routed_judgments": len(routed),
        "routed_unacceptable_rate": sum(
            row["predicted_queue"] not in label["acceptable_queues_at_intake"]
            for label, row in routed
        )
        / len(routed)
        if routed
        else None,
        "sufficiency_known": sum(
            label["sufficient_information_at_intake"] is not None for label, _ in judgments
        ),
        "limitation": (
            "Individual latest reviewer judgments on a selected subset; "
            "no adjudicated consensus or population estimate."
        ),
    }
