"""Persistent helpdesk transitions, timing, commands, privacy, and replay verification."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rightdesk.cli import app
from rightdesk.schemas import TicketInput
from rightdesk.simulation.engine import Simulator
from rightdesk.simulation.scenarios import export_simulation, replay_development, run_demo

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/simulator/retail_v1.yaml"


@pytest.fixture
def simulation(tmp_path):
    simulator = Simulator(tmp_path / "simulator.sqlite3")
    simulator.initialize()
    return simulator, simulator.create(CONFIG, "test")


def customer(identifier="one", text="I need my password reset.", language="en"):
    return TicketInput(
        case_id=identifier, conversation_id=identifier, text=text, language=language, channel="chat"
    )


def test_direct_route_service_timer_and_persistence(simulation):
    engine, key = simulation
    state = engine.intake(key, customer(), "create")
    assert state["status"] == "assigned" and state["assigned_queue"] == "account"
    assert state["service_due"] == 45
    engine.advance(key, 44, "clock1")
    assert engine.ticket(key, "one").status == "assigned"
    engine.advance(key, 1, "clock2")
    assert Simulator(engine.path).ticket(key, "one").status == "resolved"
    assert engine.verify(key)["verified"]


def test_clarification_reply_reroutes_using_only_arrived_text(simulation):
    engine, key = simulation
    original = customer(text="I have a charge and refund issue.")
    state = engine.intake(key, original, "create")
    assert state["pending_question_id"] == "charge_or_refund"
    assert len(state["messages"]) == 1
    before = json.dumps([event.model_dump() for event in engine.events(key)])
    secret = "I am waiting for refund of the original payment."
    assert secret not in before
    assert "expected_queue" not in before and "eventual_owner" not in before
    engine.advance(key, 15, "clock")
    updated = engine.reply(key, "one", secret, state["version"], "reply")
    assert updated["assigned_queue"] == "returns_refunds"
    assert updated["asked_question_ids"] == ["charge_or_refund"]
    assert len(updated["messages"]) == 2
    engine.advance(key, 45, "finish")
    assert engine.ticket(key, "one").status == "resolved"
    assert engine.verify(key)["verified"]


def test_timeout_late_reply_and_unhelpful_answer_never_reask(simulation):
    engine, key = simulation
    engine.intake(key, customer(text="charge refund"), "create")
    engine.advance(key, 120, "timeout")
    waiting = engine.ticket(key, "one")
    assert waiting.status == "awaiting_agent"
    late = engine.reply(key, "one", "a new charge", waiting.version, "late")
    assert late["status"] == "awaiting_agent"
    events = engine.events(key)
    assert any(event.kind == "question_timed_out" for event in events)
    assert (
        next(event for event in events if event.kind == "customer_replied").payload["late"] is True
    )
    assert len([event for event in events if event.kind == "question_sent"]) == 1
    engine.intake(key, customer("two", "charge refund"), "create2")
    second = engine.ticket(key, "two")
    result = engine.reply(key, "two", "I do not know", second.version, "unhelpful")
    assert result["status"] == "awaiting_agent" and len(result["asked_question_ids"]) == 1


def test_agent_assign_reassign_resolve_reopen_and_invalid_transitions(simulation):
    engine, key = simulation
    state = engine.intake(key, customer(text="Hello"), "create")
    state = engine.agent(key, "one", "assign", "account", "a", state["version"], "assign")
    state = engine.agent(key, "one", "reassign", "technical_support", "a", state["version"], "move")
    assert state["assigned_queue"] == "technical_support"
    state = engine.agent(key, "one", "resolve", None, "a", state["version"], "resolve")
    state = engine.agent(key, "one", "reopen", None, "a", state["version"], "reopen")
    assert state["status"] == "awaiting_agent" and state["assigned_queue"] is None
    with pytest.raises(ValueError, match="transition"):
        engine.agent(key, "one", "resolve", None, "a", state["version"], "bad")
    assert engine.verify(key)["verified"]


def test_idempotency_conflicting_retries_stale_versions_and_append_only_events(simulation):
    engine, key = simulation
    ticket = customer()
    first = engine.intake(key, ticket, "same")
    assert engine.intake(key, ticket, "same") == first
    assert len(engine.events(key)) == 3
    with pytest.raises(ValueError, match="different request"):
        engine.intake(key, customer(text="refund"), "same")
    with pytest.raises(ValueError, match="already exists"):
        engine.intake(key, ticket, "other")
    with pytest.raises(ValueError, match="changed"):
        engine.agent(key, "one", "resolve", None, "agent", 1, "stale")
    engine.advance(key, 1, "clock")
    with pytest.raises(ValueError, match="clock changed"):
        engine.advance(key, 1, "staleclock", 0)
    with engine.connection(True) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM events")
    assert engine.verify(key)["events"] == 3


def test_concurrent_duplicate_delivery_and_cursor_stream(simulation):
    engine, key = simulation
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: engine.intake(key, customer(), "same"), range(2)))
    assert results[0] == results[1]
    first = engine.events(key, limit=2)
    rest = engine.events(key, after=first[-1].event_id)
    assert len(first) + len(rest) == 3
    assert len({event.event_id for event in first + rest}) == 3
    assert engine.events(key, after=rest[-1].event_id) == []


def test_projection_corruption_is_detected(simulation):
    engine, key = simulation
    engine.intake(key, customer(), "create")
    with engine.connection(True) as connection:
        state = engine.ticket(key, "one").model_dump()
        state["assigned_queue"] = "technical_support"
        connection.execute(
            "UPDATE tickets SET state_json=? WHERE simulation_id=?", (json.dumps(state), key)
        )
    with pytest.raises(ValueError, match="projection differs"):
        engine.verify(key)


def test_synthetic_scenario_exercises_lifecycle_and_exports(tmp_path):
    engine = Simulator(tmp_path / "demo.sqlite3")
    report = run_demo(engine, CONFIG, ROOT / "configs/simulator/demo_v1.yaml", tmp_path / "demo")
    run = json.loads((report.parent / "run.json").read_text())
    assert run["tickets"] == 14
    assert run["state_counts"] == {"resolved": 10, "awaiting_agent": 4}
    assert run["questions_sent"] == 8 and run["max_questions_per_ticket"] == 1
    assert run["late_replies"] == 1
    assert all(run["scenario_checks"].values())
    assert not run["human_usefulness_measured"]
    with pytest.raises(ValueError, match="exists"):
        export_simulation(engine, run["simulation_id"], report.parent)


def test_development_replay_uses_intake_only(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    from test_workbench import workbench

    bundle = workbench.__wrapped__(tmp_path)
    original = bundle["store"].query

    def guarded(sql, parameters=()):
        assert not any(
            table in sql.lower() for table in ("reference_labels", "reviews", "predictions")
        )
        return original(sql, parameters)

    monkeypatch.setattr(bundle["store"], "query", guarded)
    report = replay_development(
        Simulator(tmp_path / "replay.sqlite3"),
        CONFIG,
        bundle["store"],
        bundle["dataset_id"],
        3,
        tmp_path / "replay",
    )
    run = json.loads((report.parent / "run.json").read_text())
    assert run["tickets"] == 3 and not run["scripted_customer_replies"]
    assert not run["test_evaluated"]


def test_cli_create_and_unrelated_database_protection(tmp_path):
    path = tmp_path / "sim.sqlite3"
    result = CliRunner().invoke(
        app, ["simulate", "create", "--config", str(CONFIG), "--database", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip().startswith("sim-")
    with sqlite3.connect(tmp_path / "other.sqlite3") as connection:
        connection.execute("CREATE TABLE unrelated(x)")
    with pytest.raises(ValueError, match="unrelated"):
        Simulator(tmp_path / "other.sqlite3").initialize()
