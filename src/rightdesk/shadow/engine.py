"""Read-only event input, separate durable cursor, and atomic shadow proposals."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from rightdesk.data.importers.abcd import checksum, write_json
from rightdesk.schemas import PolicyDecision, TicketInput
from rightdesk.simulation.engine import PublicEvent
from rightdesk.storage import encode, utc_now


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


class PublicStream:
    """Never opens a write handle or reads primary projections, labels, or scenario plans."""

    def __init__(self, path: Path, simulation_id: str):
        self.path = path.resolve()
        self.simulation_id = simulation_id

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")

        def authorize(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_READ:
                return (
                    sqlite3.SQLITE_OK if table in ("events", "simulations") else sqlite3.SQLITE_DENY
                )
            if action in (
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_TRANSACTION,
            ):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        try:
            connection.execute("BEGIN")
            yield connection
        finally:
            connection.close()

    def snapshot(self):
        with self.connection() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM simulations WHERE simulation_id=?", (self.simulation_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Public simulation not found")
            snapshot = json.loads(row[0])
            return {key: snapshot[key] for key in ("workflow", "clarification")}

    def events(self, after=0, limit=500):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError("Invalid public event cursor/limit")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE simulation_id=? AND event_id>? "
                "ORDER BY event_id LIMIT ?",
                (self.simulation_id, after, limit),
            ).fetchall()
            return [
                PublicEvent(
                    **{k: v for k, v in dict(row).items() if k != "payload_json"},
                    payload=json.loads(row["payload_json"]),
                )
                for row in rows
            ]

    def backlog(self, after):
        with self.connection() as connection:
            row = connection.execute(
                "SELECT count(*),max(event_id) FROM events WHERE simulation_id=? AND event_id>?",
                (self.simulation_id, after),
            ).fetchone()
            return {"unobserved_events": row[0], "latest_unobserved_event_id": row[1]}


DDL = """
CREATE TABLE IF NOT EXISTS observers (
 observer_id TEXT PRIMARY KEY, binding_json TEXT NOT NULL, cursor INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observed_cases (
 observer_id TEXT NOT NULL REFERENCES observers(observer_id),case_id TEXT NOT NULL,
 state_json TEXT NOT NULL,PRIMARY KEY(observer_id,case_id)
);
CREATE TABLE IF NOT EXISTS observed_events (
 observer_id TEXT NOT NULL REFERENCES observers(observer_id),event_id INTEGER NOT NULL,
 event_json TEXT NOT NULL, event_sha256 TEXT NOT NULL,
 PRIMARY KEY(observer_id,event_id)
);
CREATE TABLE IF NOT EXISTS proposals (
 observer_id TEXT NOT NULL,event_id INTEGER NOT NULL,model_id TEXT NOT NULL,
 case_id TEXT NOT NULL,input_json TEXT NOT NULL,input_sha256 TEXT NOT NULL,
 prediction_json TEXT,decision_json TEXT NOT NULL,error_json TEXT,latency_ms REAL NOT NULL,
 observed_at TEXT NOT NULL,PRIMARY KEY(observer_id,event_id,model_id),
 FOREIGN KEY(observer_id,event_id) REFERENCES observed_events(observer_id,event_id)
);
CREATE TRIGGER IF NOT EXISTS proposals_no_update BEFORE UPDATE ON proposals
 BEGIN SELECT RAISE(ABORT,'Shadow proposals are immutable'); END;
CREATE TRIGGER IF NOT EXISTS proposals_no_delete BEFORE DELETE ON proposals
 BEGIN SELECT RAISE(ABORT,'Shadow proposals are immutable'); END;
CREATE TRIGGER IF NOT EXISTS observed_events_no_update BEFORE UPDATE ON observed_events
 BEGIN SELECT RAISE(ABORT,'Observed events are immutable'); END;
CREATE TRIGGER IF NOT EXISTS observed_events_no_delete BEFORE DELETE ON observed_events
 BEGIN SELECT RAISE(ABORT,'Observed events are immutable'); END;
"""


class ShadowStore:
    def __init__(self, path: Path):
        self.path = path.resolve()

    @contextmanager
    def connection(self, write=False):
        if not self.path.exists():
            raise ValueError("Shadow database is not initialized")
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            existing = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if existing not in (0, 1) or (existing == 0 and tables):
                raise ValueError("Unsupported shadow database schema")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(DDL)
            connection.execute("PRAGMA user_version=1")

    def bind(self, stream, predictor, name="shadow-v1"):
        if self.path == stream.path:
            raise ValueError("Shadow and primary databases must be different")
        binding = {
            "primary_path": str(stream.path),
            "simulation_id": stream.simulation_id,
            "snapshot": stream.snapshot(),
            "predictor": predictor.metadata,
            "name": name,
            "input_contract": "arrived_customer_and_delivered_questions_v1",
        }
        observer_id = "shadow-" + digest(binding)[:24]
        with self.connection(True) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO observers VALUES(?,?,0,?,?)",
                (observer_id, encode(binding), utc_now(), utc_now()),
            )
        return observer_id

    def observers(self):
        if not self.path.exists():
            return []
        with self.connection() as connection:
            return [
                dict(row)
                for row in connection.execute("SELECT * FROM observers ORDER BY created_at")
            ]

    def capture(self, observer_id):
        with self.connection() as connection:
            observer = connection.execute(
                "SELECT * FROM observers WHERE observer_id=?", (observer_id,)
            ).fetchone()
            if observer is None:
                raise ValueError("Shadow observer not found")
            result = {"observer": dict(observer)}
            for key, table in (
                ("events", "observed_events"),
                ("proposals", "proposals"),
                ("cases", "observed_cases"),
            ):
                result[key] = [
                    dict(row)
                    for row in connection.execute(
                        f"SELECT * FROM {table} WHERE observer_id=? ORDER BY "
                        + ("case_id" if key == "cases" else "event_id"),
                        (observer_id,),
                    )
                ]
            return result


def project(state, event):
    """Prefix-only feature projection. Primary choices and future events never enter text."""
    if state is None:
        if event.kind != "ticket_created" or event.case_version != 1:
            raise ValueError("Missing initial public event")
        ticket = TicketInput.model_validate(event.payload["ticket"])
        if ticket.case_id != event.case_id:
            raise ValueError("Public case identity mismatch")
        return {
            "ticket": ticket.model_dump(),
            "version": 1,
            "clock": event.clock,
            "transcript": [ticket.text],
            "primary_asked": [],
            "proposed": {},
        }
    state = json.loads(encode(state))
    if event.case_version != state["version"] + 1 or event.clock < state["clock"]:
        raise ValueError("Non-contiguous public event stream")
    state.update(version=event.case_version, clock=event.clock)
    if event.kind == "customer_replied":
        text = event.payload["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty public customer response")
        state["transcript"].append("Customer: " + text)
    elif event.kind == "question_sent":
        state["primary_asked"].append(event.payload["question_id"])
        state["transcript"].append("Support question: " + event.payload["text"])
    return state


class ShadowWorker:
    def __init__(self, stream: PublicStream, store: ShadowStore, predictor, name="shadow-v1"):
        if store.path == stream.path:
            raise ValueError("Shadow and primary databases must be different")
        self.stream, self.store, self.predictor = stream, store, predictor
        self.observer_id = store.bind(stream, predictor, name)
        self.verify_prefix()

    def verify_prefix(self):
        captured = self.store.capture(self.observer_id)
        cursor = captured["observer"]["cursor"]
        binding = json.loads(captured["observer"]["binding_json"])
        if binding["snapshot"] != self.stream.snapshot():
            raise ValueError("Primary workflow snapshot changed")
        stored = captured["events"]
        actual, after = [], 0
        while after < cursor:
            chunk = self.stream.events(after)
            chunk = [e for e in chunk if e.event_id <= cursor]
            if not chunk:
                raise ValueError("Primary event history was removed")
            actual.extend(chunk)
            after = chunk[-1].event_id
        if [(r["event_id"], r["event_sha256"]) for r in stored] != [
            (e.event_id, digest(e.model_dump())) for e in actual
        ]:
            raise ValueError("Observed primary event prefix changed")
        return True

    def run(self, max_events=10000):
        """Drain up to max_events; restart resumes from the committed event cursor."""
        if type(max_events) is not int or not 1 <= max_events <= 1000000:
            raise ValueError("Invalid event budget")
        processed, inferred = 0, 0
        while processed < max_events:
            with self.store.connection() as connection:
                cursor = connection.execute(
                    "SELECT cursor FROM observers WHERE observer_id=?", (self.observer_id,)
                ).fetchone()[0]
            events = self.stream.events(cursor, min(500, max_events - processed))
            if not events:
                break
            for event in events:
                with self.store.connection() as connection:
                    observed_cursor = connection.execute(
                        "SELECT cursor FROM observers WHERE observer_id=?", (self.observer_id,)
                    ).fetchone()[0]
                    if observed_cursor != cursor:
                        break
                    row = connection.execute(
                        "SELECT state_json FROM observed_cases WHERE observer_id=? AND case_id=?",
                        (self.observer_id, event.case_id),
                    ).fetchone()
                state = project(json.loads(row[0]) if row else None, event)
                results = []
                if event.kind in ("ticket_created", "customer_replied"):
                    data = {**state["ticket"], "text": "\n\n".join(state["transcript"])}
                    ticket = TicketInput.model_validate(data)
                    for name in self.predictor.names:
                        asked = sorted(
                            set(state["primary_asked"] + state["proposed"].get(name, []))
                        )
                        started, error, prediction = perf_counter(), None, None
                        try:
                            prediction, decision = self.predictor.predict(name, ticket, asked)
                            decision = PolicyDecision.for_workflow(
                                decision, self.predictor.workflows[name]
                            )
                            if decision.case_id != ticket.case_id:
                                raise ValueError("Shadow decision case mismatch")
                        except Exception as exc:
                            error = {"type": type(exc).__name__, "message": str(exc)}
                            prediction = None
                            decision = self.predictor.failure(name, ticket)
                        elapsed = (perf_counter() - started) * 1000
                        if decision.action == "clarify":
                            state["proposed"].setdefault(name, []).append(decision.question_id)
                        results.append((name, ticket, prediction, decision, error, elapsed))
                # Inference happens outside the write transaction. A concurrent winner forces retry.
                with self.store.connection(True) as connection:
                    current = connection.execute(
                        "SELECT cursor FROM observers WHERE observer_id=?", (self.observer_id,)
                    ).fetchone()[0]
                    if current != cursor:
                        break
                    connection.execute(
                        "INSERT INTO observed_events VALUES(?,?,?,?)",
                        (
                            self.observer_id,
                            event.event_id,
                            event.model_dump_json(),
                            digest(event.model_dump()),
                        ),
                    )
                    connection.execute(
                        "INSERT INTO observed_cases VALUES(?,?,?) ON CONFLICT(observer_id,case_id) "
                        "DO UPDATE SET state_json=excluded.state_json",
                        (self.observer_id, event.case_id, encode(state)),
                    )
                    for name, ticket, prediction, decision, error, elapsed in results:
                        connection.execute(
                            "INSERT INTO proposals VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                self.observer_id,
                                event.event_id,
                                name,
                                event.case_id,
                                ticket.model_dump_json(),
                                digest(ticket.model_dump()),
                                prediction.model_dump_json() if prediction else None,
                                decision.model_dump_json(),
                                encode(error) if error else None,
                                elapsed,
                                utc_now(),
                            ),
                        )
                    connection.execute(
                        "UPDATE observers SET cursor=?,updated_at=? WHERE observer_id=?",
                        (event.event_id, utc_now(), self.observer_id),
                    )
                cursor = event.event_id
                processed += 1
                inferred += len(results)
        with self.store.connection() as connection:
            cursor = connection.execute(
                "SELECT cursor FROM observers WHERE observer_id=?", (self.observer_id,)
            ).fetchone()[0]
        return {
            "observer_id": self.observer_id,
            "processed_events": processed,
            "new_proposals": inferred,
            "cursor": cursor,
            **self.stream.backlog(cursor),
        }


def export_shadow(store: ShadowStore, observer_id: str, output: Path):
    if output.exists():
        raise ValueError("Shadow export exists; choose a fresh directory")
    captured = store.capture(observer_id)
    binding = json.loads(captured["observer"]["binding_json"])
    events = [PublicEvent.model_validate_json(row["event_json"]) for row in captured["events"]]
    # Match the next primary decision after the same input event, without crossing a new input.
    pending, matched = {}, {}
    for event in events:
        if event.kind in ("ticket_created", "customer_replied"):
            pending[event.case_id] = event.event_id
        elif event.kind == "incumbent_decided" and event.case_id in pending:
            matched[pending.pop(event.case_id)] = event.payload["decision"]
    models = {}
    for row in captured["proposals"]:
        model = models.setdefault(
            row["model_id"],
            {
                "proposals": 0,
                "failures": 0,
                "route": 0,
                "clarify": 0,
                "review": 0,
                "paired": 0,
                "action_agreements": 0,
                "queue_pairs": 0,
                "queue_agreements": 0,
            },
        )
        decision = json.loads(row["decision_json"])
        model["proposals"] += 1
        model["failures"] += row["error_json"] is not None
        model[decision["action"]] += 1
        primary = matched.get(row["event_id"])
        if primary:
            model["paired"] += 1
            model["action_agreements"] += decision["action"] == {"ask": "clarify"}.get(
                primary["action"], primary["action"]
            )
            prediction = json.loads(row["prediction_json"]) if row["prediction_json"] else None
            if primary["action"] == "route" and prediction:
                from rightdesk.schemas import ModelPrediction, WorkflowConfig

                workflow_data = (
                    binding["predictor"]
                    .get("policies", {})
                    .get(row["model_id"], {})
                    .get("workflow", binding["snapshot"]["workflow"])
                )
                chosen = ModelPrediction.for_workflow(
                    prediction, WorkflowConfig.model_validate(workflow_data)
                ).selected_queue()
                if chosen is not None:
                    model["queue_pairs"] += 1
                    model["queue_agreements"] += chosen == primary["queue"]
    output.mkdir(parents=True)
    for name in ("events", "proposals", "cases"):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in captured[name]:
                stream.write(encode(row) + "\n")
    report = {
        "schema_version": 1,
        "phase": 13,
        "observer": captured["observer"],
        "binding": binding,
        "models": models,
        "events": len(events),
        "cases": len(captured["cases"]),
        "exported_at": utc_now(),
        "comparison": "Same arrived-input checkpoint; agreement is not correctness or benefit",
        "question_usefulness": None,
        "artifacts": {p.name: checksum(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write_json(output / "run.json", report)
    return output / "run.json"
