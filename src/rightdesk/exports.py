"""Portable, checksummed Parquet snapshots with inputs and labels in separate tables."""

import json
import tempfile
from pathlib import Path

from rightdesk.data.importers.abcd import checksum
from rightdesk.storage import Store, encode, utc_now


def export_run(store: Store, run_id: str, output: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = output.resolve()
    if output.exists():
        raise ValueError("Export output exists; choose a fresh directory")
    # One read transaction gives a consistent snapshot, including review history.
    with store.connection() as connection:
        run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if run is None:
            raise ValueError("Run not found")
        dataset = connection.execute(
            "SELECT * FROM datasets WHERE dataset_id=?", (run["dataset_id"],)
        ).fetchone()
        models = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM models WHERE run_id=? ORDER BY model_id", (run_id,)
            )
        ]
        predictions = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM predictions WHERE run_id=? ORDER BY model_id,case_id", (run_id,)
            )
        ]
        cases = [
            dict(row)
            for row in connection.execute(
                """SELECT c.* FROM cases c WHERE c.dataset_id=?
            AND EXISTS(SELECT 1 FROM predictions p WHERE p.run_id=? AND p.case_id=c.case_id)
            ORDER BY c.case_id""",
                (run["dataset_id"], run_id),
            )
        ]
        labels = [
            dict(row)
            for row in connection.execute(
                """SELECT r.* FROM reference_labels r
            WHERE r.dataset_id=? AND EXISTS(SELECT 1 FROM predictions p
            WHERE p.run_id=? AND p.case_id=r.case_id) ORDER BY r.case_id""",
                (run["dataset_id"], run_id),
            )
        ]
        reviews = [
            dict(row)
            for row in connection.execute(
                """SELECT r.* FROM reviews r
            WHERE r.dataset_id=? AND EXISTS(SELECT 1 FROM predictions p
            WHERE p.run_id=? AND p.case_id=r.case_id) ORDER BY r.annotation_id""",
                (run["dataset_id"], run_id),
            )
        ]
    prediction_records = []
    for row in predictions:
        prediction = json.loads(row["prediction_json"])
        prediction_records.append(
            {
                "run_id": run_id,
                "model_id": row["model_id"],
                "case_id": row["case_id"],
                "dataset_id": row["dataset_id"],
                **prediction,
                "queue_probabilities": list(prediction["queue_probabilities"].items())
                if prediction["queue_probabilities"] is not None
                else None,
                "latency_ms": row["latency_ms"],
                "decision_json": row["decision_json"],
            }
        )
    schema = pa.schema(
        [
            ("run_id", pa.string()),
            ("model_id", pa.string()),
            ("case_id", pa.string()),
            ("dataset_id", pa.string()),
            ("workflow_id", pa.string()),
            ("workflow_version", pa.string()),
            ("queue_probabilities", pa.map_(pa.string(), pa.float64())),
            ("predicted_queue", pa.string()),
            ("abstention_reason", pa.string()),
            ("sufficient_information_probability", pa.float64()),
            ("suggested_question_id", pa.string()),
            ("model_revision", pa.string()),
            ("calibration_version", pa.string()),
            ("latency_ms", pa.float64()),
            ("decision_json", pa.string()),
        ]
    )
    ticket_records = []
    for row in cases:
        ticket_records.append(
            {
                "dataset_id": row["dataset_id"],
                "partition": row["partition_role"],
                "group_key": row["group_key"],
                **json.loads(row["ticket_json"]),
            }
        )
    label_records = [
        {
            "dataset_id": row["dataset_id"],
            "case_id": row["case_id"],
            "label_json": row["label_json"],
        }
        for row in labels
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output.parent) as temporary:
        staging = Path(temporary)
        pq.write_table(
            pa.Table.from_pylist(prediction_records, schema=schema), staging / "predictions.parquet"
        )
        pq.write_table(pa.Table.from_pylist(ticket_records), staging / "tickets.parquet")
        pq.write_table(pa.Table.from_pylist(label_records), staging / "reference_labels.parquet")
        for filename, payload in [
            ("run.json", json.loads(run["manifest_json"])),
            ("dataset.json", json.loads(dataset["manifest_json"])),
            ("workflow.json", json.loads(dataset["workflow_json"])),
            ("models.json", models),
        ]:
            (staging / filename).write_text(encode(payload) + "\n", encoding="utf-8")
        (staging / "review_history.jsonl").write_text(
            "".join(
                encode(
                    {
                        **{key: value for key, value in row.items() if key != "label_json"},
                        "label": json.loads(row["label_json"]),
                    }
                )
                + "\n"
                for row in reviews
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "source_run_sha256": run["source_sha256"],
            "dataset_id": run["dataset_id"],
            "exported_at": utc_now(),
            "counts": {
                "predictions": len(predictions),
                "tickets": len(cases),
                "reviews": len(reviews),
            },
            "artifacts": {p.name: checksum(p) for p in sorted(staging.iterdir()) if p.is_file()},
        }
        (staging / "export.json").write_text(encode(manifest) + "\n", encoding="utf-8")
        staging.rename(output)
    return output / "export.json"
