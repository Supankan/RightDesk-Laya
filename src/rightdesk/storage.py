"""Transactional local experiment catalog and append-only human annotations."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    workflow_json TEXT NOT NULL,
    source_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cases (
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    case_id TEXT NOT NULL,
    partition_role TEXT NOT NULL CHECK(partition_role IN ('development','calibration')),
    group_key TEXT NOT NULL,
    ticket_json TEXT NOT NULL,
    PRIMARY KEY(dataset_id, case_id)
);
CREATE TABLE IF NOT EXISTS reference_labels (
    dataset_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    label_json TEXT NOT NULL,
    PRIMARY KEY(dataset_id, case_id),
    FOREIGN KEY(dataset_id,case_id) REFERENCES cases(dataset_id,case_id)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    source_sha256 TEXT NOT NULL UNIQUE,
    source_path TEXT NOT NULL,
    kind TEXT NOT NULL,
    split TEXT NOT NULL,
    created_at TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    model_id TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    workflow_json TEXT NOT NULL,
    policy_json TEXT,
    PRIMARY KEY(run_id,model_id)
);
CREATE TABLE IF NOT EXISTS predictions (
    run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    prediction_json TEXT NOT NULL,
    latency_ms REAL,
    decision_json TEXT,
    PRIMARY KEY(run_id,model_id,case_id),
    FOREIGN KEY(run_id,model_id) REFERENCES models(run_id,model_id),
    FOREIGN KEY(dataset_id,case_id) REFERENCES cases(dataset_id,case_id)
);
CREATE TABLE IF NOT EXISTS review_queues (
    queue_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    title TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_items (
    queue_id TEXT NOT NULL REFERENCES review_queues(queue_id),
    position INTEGER NOT NULL,
    dataset_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    PRIMARY KEY(queue_id,case_id),
    UNIQUE(queue_id,position),
    FOREIGN KEY(dataset_id,case_id) REFERENCES cases(dataset_id,case_id)
);
CREATE TABLE IF NOT EXISTS reviews (
    annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK(revision > 0),
    label_json TEXT NOT NULL,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(dataset_id,case_id,reviewer_id,revision),
    FOREIGN KEY(dataset_id,case_id) REFERENCES cases(dataset_id,case_id)
);
CREATE INDEX IF NOT EXISTS predictions_cases ON predictions(dataset_id,case_id);
CREATE INDEX IF NOT EXISTS cases_groups ON cases(dataset_id,group_key);
CREATE INDEX IF NOT EXISTS reviews_latest ON reviews(dataset_id,case_id,reviewer_id,revision DESC);
CREATE TRIGGER IF NOT EXISTS reviews_no_update BEFORE UPDATE ON reviews
BEGIN SELECT RAISE(ABORT, 'Reviews are append-only'); END;
CREATE TRIGGER IF NOT EXISTS reviews_no_delete BEFORE DELETE ON reviews
BEGIN SELECT RAISE(ABORT, 'Reviews are append-only'); END;
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class Store:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()

    @contextmanager
    def connection(self, write=False):
        if not self.path.exists():
            raise ValueError("Initialize the experiment store first")
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                raise ValueError("Unsupported store schema version")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise ValueError("Unsupported store schema version")
            if version == 0:
                # Do not adopt an unrelated SQLite file.
                tables = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if tables:
                    raise ValueError("Refusing to initialize an unrelated database")
                connection.executescript(
                    "BEGIN IMMEDIATE;\n" + DDL + f"\nPRAGMA user_version={SCHEMA_VERSION};\nCOMMIT;"
                )
            connection.execute("PRAGMA journal_mode=WAL")
        finally:
            connection.close()
        return self.path

    def query(self, sql, parameters=()):
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(sql, parameters).fetchall()]

    def one(self, sql, parameters=()):
        rows = self.query(sql, parameters)
        if not rows:
            raise ValueError("Requested record was not found")
        return rows[0]

    def list_runs(self):
        return self.query("""SELECT r.run_id,r.dataset_id,r.kind,r.split,r.created_at,r.imported_at,
            COUNT(DISTINCT m.model_id) AS model_count,
            (SELECT COUNT(DISTINCT p.case_id) FROM predictions p WHERE p.run_id=r.run_id) AS cases
            FROM runs r JOIN models m ON m.run_id=r.run_id GROUP BY r.run_id
            ORDER BY r.created_at DESC,r.run_id""")

    def dataset(self, dataset_id):
        row = self.one("SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,))
        row["manifest"] = json.loads(row.pop("manifest_json"))
        row["workflow"] = json.loads(row.pop("workflow_json"))
        return row

    def run(self, run_id):
        row = self.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row["manifest"] = json.loads(row.pop("manifest_json"))
        return row

    def models(self, run_id):
        rows = self.query("SELECT * FROM models WHERE run_id=? ORDER BY model_id", (run_id,))
        for row in rows:
            row["metrics"] = json.loads(row.pop("metrics_json"))
            row["workflow"] = json.loads(row.pop("workflow_json"))
            policy = row.pop("policy_json")
            row["policy"] = json.loads(policy) if policy else None
        return rows

    def latest_reviews(self, dataset_id):
        return self.query(
            """SELECT r.* FROM reviews r WHERE dataset_id=? AND revision=(
            SELECT MAX(s.revision) FROM reviews s WHERE s.dataset_id=r.dataset_id
            AND s.case_id=r.case_id AND s.reviewer_id=r.reviewer_id)
            ORDER BY r.case_id,r.reviewer_id""",
            (dataset_id,),
        )
