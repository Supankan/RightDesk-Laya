"""Persistent primary helpdesk with virtual time, idempotent commands, and public events."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import Field, model_validator

from rightdesk.policy.clarification import ClarificationConfig
from rightdesk.schemas import Contract, Identifier, NonEmpty, TicketInput, WorkflowConfig
from rightdesk.simulation.config import (
    IncumbentConfig,
    PrimaryRouter,
    SimulationConfig,
    load_simulation,
)
from rightdesk.storage import encode, utc_now

DDL = """
CREATE TABLE simulations (
 simulation_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
 snapshot_json TEXT NOT NULL, source_json TEXT NOT NULL, clock INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE tickets (
 simulation_id TEXT NOT NULL REFERENCES simulations(simulation_id), case_id TEXT NOT NULL,
 state_json TEXT NOT NULL, PRIMARY KEY(simulation_id,case_id)
);
CREATE TABLE events (
 event_id INTEGER PRIMARY KEY AUTOINCREMENT,
 simulation_id TEXT NOT NULL REFERENCES simulations(simulation_id),
 case_id TEXT NOT NULL, case_version INTEGER NOT NULL, clock INTEGER NOT NULL,
 occurred_at TEXT NOT NULL, kind TEXT NOT NULL, actor TEXT NOT NULL, payload_json TEXT NOT NULL,
 command_id TEXT NOT NULL, UNIQUE(simulation_id,case_id,case_version)
);
CREATE TABLE commands (
 simulation_id TEXT NOT NULL REFERENCES simulations(simulation_id), command_id TEXT NOT NULL,
 request_sha256 TEXT NOT NULL, result_json TEXT NOT NULL, PRIMARY KEY(simulation_id,command_id)
);
CREATE INDEX events_cursor ON events(simulation_id,event_id);
CREATE TRIGGER events_no_update BEFORE UPDATE ON events
 BEGIN SELECT RAISE(ABORT,'Events are append-only'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events
 BEGIN SELECT RAISE(ABORT,'Events are append-only'); END;
CREATE TRIGGER commands_no_update BEFORE UPDATE ON commands
 BEGIN SELECT RAISE(ABORT,'Commands are immutable'); END;
CREATE TRIGGER commands_no_delete BEFORE DELETE ON commands
 BEGIN SELECT RAISE(ABORT,'Commands are immutable'); END;
"""


class CustomerMessage(Contract):
    text: NonEmpty
    clock: Annotated[int, Field(ge=0)]


class TicketState(Contract):
    initial_ticket: TicketInput
    version: Annotated[int, Field(ge=1)]
    status: Literal["new", "awaiting_customer", "awaiting_agent", "assigned", "resolved"]
    assigned_queue: Identifier | None = None
    agent_id: NonEmpty | None = None
    asked_question_ids: list[Identifier] = Field(default_factory=list)
    pending_question_id: Identifier | None = None
    reply_deadline: int | None = None
    service_due: int | None = None
    messages: list[CustomerMessage]
    created_clock: int
    updated_clock: int

    @model_validator(mode="after")
    def valid_projection(self) -> Self:
        if (self.status in ("assigned", "resolved")) != (self.assigned_queue is not None):
            raise ValueError("Assigned/resolved tickets require exactly one queue")
        if (self.status == "assigned") != (self.service_due is not None):
            raise ValueError("Only assigned tickets have a service deadline")
        waiting = self.status == "awaiting_customer"
        if waiting != (self.pending_question_id is not None and self.reply_deadline is not None):
            raise ValueError("Only waiting tickets have a pending question/deadline")
        if not waiting and (
            self.pending_question_id is not None or self.reply_deadline is not None
        ):
            raise ValueError("Stale pending clarification")
        return self


class PublicEvent(Contract):
    event_id: int
    simulation_id: NonEmpty
    case_id: NonEmpty
    case_version: int
    clock: int
    occurred_at: NonEmpty
    kind: NonEmpty
    actor: Literal["primary", "customer", "agent", "system"]
    payload: dict
    command_id: NonEmpty


def apply_event(state: TicketState | None, event: PublicEvent) -> TicketState:
    """One projector is used by both writes and verification replay."""
    if state is None:
        if event.kind != "ticket_created" or event.case_version != 1:
            raise ValueError("Stream must start with ticket_created version 1")
        ticket = TicketInput.model_validate(event.payload["ticket"])
        if ticket.case_id != event.case_id:
            raise ValueError("Event/ticket identity mismatch")
        return TicketState(
            initial_ticket=ticket,
            version=1,
            status="new",
            messages=[CustomerMessage(text=ticket.text, clock=event.clock)],
            created_clock=event.clock,
            updated_clock=event.clock,
        )
    if event.case_version != state.version + 1 or event.clock < state.updated_clock:
        raise ValueError("Non-contiguous case version or reversed time")
    data = state.model_dump()
    data.update(version=event.case_version, updated_clock=event.clock)
    payload = event.payload
    if event.kind == "question_sent":
        data["asked_question_ids"].append(payload["question_id"])
        data.update(
            status="awaiting_customer",
            pending_question_id=payload["question_id"],
            reply_deadline=payload["reply_deadline"],
        )
    elif event.kind == "customer_replied":
        data["messages"].append({"text": payload["text"], "clock": event.clock})
        data.update(status="awaiting_agent", pending_question_id=None, reply_deadline=None)
    elif event.kind in ("ticket_assigned", "ticket_reassigned"):
        data.update(
            status="assigned",
            assigned_queue=payload["queue"],
            agent_id=payload.get("agent_id"),
            service_due=payload["service_due"],
            pending_question_id=None,
            reply_deadline=None,
        )
    elif event.kind in ("agent_review_needed", "question_timed_out", "ticket_reopened"):
        data.update(
            status="awaiting_agent",
            assigned_queue=None,
            agent_id=None,
            service_due=None,
            pending_question_id=None,
            reply_deadline=None,
        )
    elif event.kind == "ticket_resolved":
        data.update(status="resolved", service_due=None)
    elif event.kind != "incumbent_decided":
        raise ValueError(f"Unknown event kind: {event.kind}")
    return TicketState.model_validate(data)


class Simulator:
    def __init__(self, database: Path):
        self.path = Path(database).resolve()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("Unsupported simulator database version")
            if version == 0:
                if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table'").fetchone():
                    raise ValueError("Refusing to adopt an unrelated simulator database")
                connection.executescript("BEGIN IMMEDIATE;" + DDL + "PRAGMA user_version=1;COMMIT;")
            else:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if not {"simulations", "events", "tickets", "commands"} <= tables:
                    raise ValueError("This is not a simulator database")
            connection.execute("PRAGMA journal_mode=WAL")
        finally:
            connection.close()
        return self.path

    @contextmanager
    def connection(self, write=False):
        if not self.path.exists():
            raise ValueError("Initialize the simulator first")
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("Unsupported simulator database version")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create(self, config_path: Path, name: str, source: dict | None = None):
        if not name.strip():
            raise ValueError("Simulation name is required")
        config, workflow, incumbent, clarification = load_simulation(config_path)
        snapshot = {
            "config": config.model_dump(),
            "workflow": workflow.model_dump(),
            "incumbent": incumbent.model_dump(),
            "clarification": clarification.model_dump(),
        }
        simulation_id = "sim-" + uuid4().hex[:20]
        with self.connection(True) as connection:
            connection.execute(
                "INSERT INTO simulations VALUES(?,?,?,?,?,0)",
                (
                    simulation_id,
                    name,
                    utc_now(),
                    encode(snapshot),
                    encode(source or {"kind": "manual_intake"}),
                ),
            )
        return simulation_id

    @staticmethod
    def _session(connection, simulation_id):
        row = connection.execute(
            "SELECT * FROM simulations WHERE simulation_id=?", (simulation_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Simulation not found")
        return dict(row)

    @staticmethod
    def _components(session):
        snapshot = json.loads(session["snapshot_json"])
        config = SimulationConfig.model_validate(snapshot["config"])
        workflow = WorkflowConfig.model_validate(snapshot["workflow"])
        router = PrimaryRouter(
            workflow,
            IncumbentConfig.model_validate(snapshot["incumbent"]),
            ClarificationConfig.model_validate(snapshot["clarification"]),
        )
        return config, workflow, router

    @staticmethod
    def _state(connection, simulation_id, case_id):
        row = connection.execute(
            "SELECT state_json FROM tickets WHERE simulation_id=? AND case_id=?",
            (simulation_id, case_id),
        ).fetchone()
        if row is None:
            raise ValueError("Simulation ticket not found")
        return TicketState.model_validate_json(row[0])

    def _command(self, simulation_id, command_id, request, operation):
        if not command_id.strip():
            raise ValueError("Command ID is required for idempotency")
        digest = hashlib.sha256(encode(request).encode()).hexdigest()
        with self.connection(True) as connection:
            session = self._session(connection, simulation_id)
            previous = connection.execute(
                "SELECT * FROM commands WHERE simulation_id=? AND command_id=?",
                (simulation_id, command_id),
            ).fetchone()
            if previous:
                if previous["request_sha256"] != digest:
                    raise ValueError("Command ID was already used with a different request")
                return json.loads(previous["result_json"])
            result = operation(connection, session)
            connection.execute(
                "INSERT INTO commands VALUES(?,?,?,?)",
                (simulation_id, command_id, digest, encode(result)),
            )
            return result

    def _emit(
        self, connection, session, state, case_id, kind, payload, command_id, actor="primary"
    ):
        config = self._components(session)[0]
        clock = session["clock"]
        occurred = (datetime.fromisoformat(config.start_at) + timedelta(seconds=clock)).isoformat()
        version = state.version + 1 if state else 1
        cursor = connection.execute(
            """INSERT INTO events(simulation_id,case_id,case_version,clock,
            occurred_at,kind,actor,payload_json,command_id) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                session["simulation_id"],
                case_id,
                version,
                clock,
                occurred,
                kind,
                actor,
                encode(payload),
                command_id,
            ),
        )
        event = PublicEvent(
            event_id=cursor.lastrowid,
            simulation_id=session["simulation_id"],
            case_id=case_id,
            case_version=version,
            clock=clock,
            occurred_at=occurred,
            kind=kind,
            actor=actor,
            payload=payload,
            command_id=command_id,
        )
        updated = apply_event(state, event)
        connection.execute(
            """INSERT INTO tickets VALUES(?,?,?) ON CONFLICT(simulation_id,case_id)
            DO UPDATE SET state_json=excluded.state_json""",
            (session["simulation_id"], case_id, updated.model_dump_json()),
        )
        return updated

    def _dispatch(self, connection, session, state, command_id, reply=None):
        config, workflow, router = self._components(session)
        choice = router.decide(state.initial_ticket, state.asked_question_ids, reply)
        case_id = state.initial_ticket.case_id
        state = self._emit(
            connection,
            session,
            state,
            case_id,
            "incumbent_decided",
            {"decision": choice.model_dump(), "observed_messages": len(state.messages)},
            command_id,
        )
        if choice.action == "route":
            return self._emit(
                connection,
                session,
                state,
                case_id,
                "ticket_assigned",
                {"queue": choice.queue, "service_due": session["clock"] + config.service_seconds},
                command_id,
            )
        if choice.action == "ask":
            question = next(
                question for question in workflow.questions if question.id == choice.question_id
            )
            return self._emit(
                connection,
                session,
                state,
                case_id,
                "question_sent",
                {
                    "question_id": question.id,
                    "text": question.text,
                    "reply_deadline": session["clock"] + config.customer_timeout_seconds,
                },
                command_id,
            )
        return self._emit(
            connection,
            session,
            state,
            case_id,
            "agent_review_needed",
            {"basis": choice.basis},
            command_id,
        )

    def intake(self, simulation_id: str, ticket: TicketInput, command_id: str):
        ticket = TicketInput.model_validate(ticket)

        def operation(connection, session):
            if connection.execute(
                "SELECT 1 FROM tickets WHERE simulation_id=? AND case_id=?",
                (simulation_id, ticket.case_id),
            ).fetchone():
                raise ValueError("Case already exists; retry using the original command ID")
            state = self._emit(
                connection,
                session,
                None,
                ticket.case_id,
                "ticket_created",
                {"ticket": ticket.model_dump()},
                command_id,
                "customer",
            )
            return self._dispatch(connection, session, state, command_id).model_dump()

        return self._command(
            simulation_id,
            command_id,
            {"operation": "intake", "ticket": ticket.model_dump()},
            operation,
        )

    def reply(
        self, simulation_id: str, case_id: str, text: str, expected_version: int, command_id: str
    ):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Customer reply cannot be empty")
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("A positive integer ticket version is required")

        def operation(connection, session):
            state = self._state(connection, simulation_id, case_id)
            if state.version != expected_version:
                raise ValueError("Ticket changed; reload before replying")
            late = state.status == "awaiting_agent" and bool(state.asked_question_ids)
            if state.status != "awaiting_customer" and not late:
                raise ValueError("Ticket is not waiting for a clarification response")
            state = self._emit(
                connection,
                session,
                state,
                case_id,
                "customer_replied",
                {"text": text, "late": late},
                command_id,
                "customer",
            )
            if late:
                return self._emit(
                    connection,
                    session,
                    state,
                    case_id,
                    "agent_review_needed",
                    {"basis": "late_or_additional_reply"},
                    command_id,
                ).model_dump()
            return self._dispatch(connection, session, state, command_id, reply=text).model_dump()

        request = {
            "operation": "reply",
            "case_id": case_id,
            "text": text,
            "expected_version": expected_version,
        }
        return self._command(simulation_id, command_id, request, operation)

    def agent(
        self,
        simulation_id: str,
        case_id: str,
        action: str,
        queue: str | None,
        agent_id: str,
        expected_version: int,
        command_id: str,
    ):
        if action not in ("assign", "reassign", "resolve", "reopen") or not agent_id.strip():
            raise ValueError("Configured agent action and nonempty agent ID required")
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("A positive integer ticket version is required")

        def operation(connection, session):
            config, workflow, _ = self._components(session)
            state = self._state(connection, simulation_id, case_id)
            if state.version != expected_version:
                raise ValueError("Ticket changed; reload before acting")
            if action in ("assign", "reassign"):
                required = "awaiting_agent" if action == "assign" else "assigned"
                if state.status != required or queue not in workflow.queue_ids:
                    raise ValueError("Invalid assignment state or queue")
                if action == "reassign" and queue == state.assigned_queue:
                    raise ValueError("Reassignment must change queue")
                kind = "ticket_assigned" if action == "assign" else "ticket_reassigned"
                payload = {
                    "queue": queue,
                    "previous_queue": state.assigned_queue,
                    "agent_id": agent_id,
                    "service_due": session["clock"] + config.service_seconds,
                }
            else:
                if queue is not None or state.status != (
                    "assigned" if action == "resolve" else "resolved"
                ):
                    raise ValueError("Invalid resolution/reopen transition")
                kind = "ticket_resolved" if action == "resolve" else "ticket_reopened"
                payload = {
                    "queue": state.assigned_queue,
                    "agent_id": agent_id,
                    "basis": "explicit_agent_action",
                }
            return self._emit(
                connection, session, state, case_id, kind, payload, command_id, "agent"
            ).model_dump()

        request = {
            "operation": "agent",
            "case_id": case_id,
            "action": action,
            "queue": queue,
            "agent_id": agent_id,
            "expected_version": expected_version,
        }
        return self._command(simulation_id, command_id, request, operation)

    def advance(
        self, simulation_id: str, seconds: int, command_id: str, expected_clock: int | None = None
    ):
        if type(seconds) is not int or not 0 <= seconds <= 10_000_000:
            raise ValueError("Clock advance must be an integer between 0 and 10000000")
        if expected_clock is not None and (type(expected_clock) is not int or expected_clock < 0):
            raise ValueError("Expected clock must be a nonnegative integer")

        def operation(connection, session):
            if expected_clock is not None and session["clock"] != expected_clock:
                raise ValueError("Simulation clock changed; reload before advancing")
            target = session["clock"] + seconds
            states = [
                TicketState.model_validate_json(row[0])
                for row in connection.execute(
                    "SELECT state_json FROM tickets WHERE simulation_id=?", (simulation_id,)
                )
            ]
            due = sorted(
                (
                    state.service_due if state.status == "assigned" else state.reply_deadline,
                    state.initial_ticket.case_id,
                    state,
                )
                for state in states
                if (state.status == "assigned" and state.service_due <= target)
                or (state.status == "awaiting_customer" and state.reply_deadline <= target)
            )
            for clock, case_id, state in due:
                session["clock"] = clock
                if state.status == "assigned":
                    self._emit(
                        connection,
                        session,
                        state,
                        case_id,
                        "ticket_resolved",
                        {"queue": state.assigned_queue, "basis": "simulated_service_completed"},
                        command_id,
                        "system",
                    )
                else:
                    self._emit(
                        connection,
                        session,
                        state,
                        case_id,
                        "question_timed_out",
                        {"question_id": state.pending_question_id},
                        command_id,
                        "system",
                    )
            connection.execute(
                "UPDATE simulations SET clock=? WHERE simulation_id=?", (target, simulation_id)
            )
            return {"clock": target, "timers_processed": len(due)}

        return self._command(
            simulation_id,
            command_id,
            {"operation": "advance", "seconds": seconds, "expected_clock": expected_clock},
            operation,
        )

    def sessions(self):
        if not self.path.exists():
            return []
        with self.connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT simulation_id,name,created_at,clock "
                    "FROM simulations ORDER BY created_at DESC"
                )
            ]

    def tickets(self, simulation_id):
        with self.connection() as connection:
            self._session(connection, simulation_id)
            return [
                TicketState.model_validate_json(row[0])
                for row in connection.execute(
                    "SELECT state_json FROM tickets WHERE simulation_id=? ORDER BY case_id",
                    (simulation_id,),
                )
            ]

    def ticket(self, simulation_id, case_id):
        with self.connection() as connection:
            return self._state(connection, simulation_id, case_id)

    def snapshot(self, simulation_id):
        with self.connection() as connection:
            session = self._session(connection, simulation_id)
            return json.loads(session["snapshot_json"])

    def events(self, simulation_id, after=0, limit=500):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError("Invalid event cursor or limit")
        with self.connection() as connection:
            self._session(connection, simulation_id)
            rows = connection.execute(
                """SELECT * FROM events WHERE simulation_id=? AND event_id>?
                ORDER BY event_id LIMIT ?""",
                (simulation_id, after, limit),
            ).fetchall()
            return [
                PublicEvent(
                    **{key: value for key, value in dict(row).items() if key != "payload_json"},
                    payload=json.loads(row["payload_json"]),
                )
                for row in rows
            ]

    def capture(self, simulation_id):
        """Read events, state projections, and clock from one consistent transaction."""
        with self.connection() as connection:
            session = self._session(connection, simulation_id)
            states = [
                TicketState.model_validate_json(row[0])
                for row in connection.execute(
                    "SELECT state_json FROM tickets WHERE simulation_id=? ORDER BY case_id",
                    (simulation_id,),
                )
            ]
            rows = connection.execute(
                "SELECT * FROM events WHERE simulation_id=? ORDER BY event_id", (simulation_id,)
            ).fetchall()
            events = [
                PublicEvent(
                    **{key: value for key, value in dict(row).items() if key != "payload_json"},
                    payload=json.loads(row["payload_json"]),
                )
                for row in rows
            ]
            return session, states, events

    def verify(self, simulation_id):
        session, states, events = self.capture(simulation_id)
        return verify_projection(simulation_id, session["clock"], states, events)


def verify_projection(simulation_id, clock, stored_states, events):
    states, last_id, last_clock = {}, 0, 0
    for event in events:
        if (
            event.simulation_id != simulation_id
            or event.clock < last_clock
            or event.event_id <= last_id
        ):
            raise ValueError("Event identity/order/time mismatch")
        states[event.case_id] = apply_event(states.get(event.case_id), event)
        last_clock, last_id = event.clock, event.event_id
    stored = {state.initial_ticket.case_id: state for state in stored_states}
    if last_clock > clock or {key: state.model_dump() for key, state in states.items()} != {
        key: state.model_dump() for key, state in stored.items()
    }:
        raise ValueError("Ticket projection differs from event replay")
    return {
        "simulation_id": simulation_id,
        "events": len(events),
        "tickets": len(states),
        "last_event_id": last_id,
        "verified": True,
    }
