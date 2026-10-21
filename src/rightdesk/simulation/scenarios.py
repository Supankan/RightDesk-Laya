"""Explicit scenario driver. Future scripts and expected outcomes never enter the router."""

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from rightdesk.config import _load
from rightdesk.data.importers.abcd import checksum
from rightdesk.schemas import Contract, Identifier, NonEmpty, TicketInput
from rightdesk.simulation.config import load_simulation
from rightdesk.simulation.engine import Simulator, verify_projection
from rightdesk.storage import Store, encode, utc_now


class PlannedReply(Contract):
    after_seconds: Annotated[int, Field(ge=1)]
    text: NonEmpty


class PlannedAgentAction(Contract):
    after_seconds: Annotated[int, Field(ge=1)]
    action: Literal["assign", "reassign", "resolve", "reopen"]
    queue: Identifier | None = None


class Scenario(Contract):
    case_id: NonEmpty
    text: NonEmpty
    language: NonEmpty = "en"
    replies: list[PlannedReply] = Field(default_factory=list)
    agent_actions: list[PlannedAgentAction] = Field(default_factory=list)
    expected_question: Identifier | None = None
    expected_status: Literal["resolved", "awaiting_agent"]
    expected_queue: Identifier | None = None


class ScenarioBundle(Contract):
    version: NonEmpty
    kind: Literal["synthetic"]
    cases: Annotated[list[Scenario], Field(min_length=1)]


def export_simulation(simulator: Simulator, simulation_id: str, output: Path, extra=None):
    if output.exists():
        raise ValueError("Simulation output exists; choose a fresh directory")
    session, states, public_events = simulator.capture(simulation_id)
    verified = verify_projection(simulation_id, session["clock"], states, public_events)
    snapshot = json.loads(session["snapshot_json"])
    events = [event.model_dump() for event in public_events]
    counts = Counter(event["kind"] for event in events)
    report = {
        "schema_version": 1,
        "phase": 12,
        "created_at": utc_now(),
        **verified,
        "snapshot": snapshot,
        "state_counts": dict(Counter(state.status for state in states)),
        "event_counts": dict(counts),
        "questions_sent": counts["question_sent"],
        "late_replies": sum(
            event["kind"] == "customer_replied" and event["payload"]["late"] for event in events
        ),
        "max_questions_per_ticket": max(
            (len(state.asked_question_ids) for state in states), default=0
        ),
        "test_evaluated": False,
        "human_usefulness_measured": False,
        "incumbent": "fixed phrase/answer rules; independent of Laya and saved model predictions",
        "engine_code_sha256": checksum(Path(__file__).with_name("engine.py")),
        "driver_code_sha256": checksum(Path(__file__)),
        "code_sha256": {
            name: checksum(Path(__file__).parents[1] / name)
            for name in (
                "simulation/engine.py",
                "simulation/config.py",
                "simulation/scenarios.py",
                "policy/clarification.py",
                "schemas.py",
            )
        },
        **(extra or {}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".simulation-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "events.jsonl").write_text(
            "".join(encode(event) + "\n" for event in events), encoding="utf-8"
        )
        (staging / "tickets.jsonl").write_text(
            "".join(state.model_dump_json() + "\n" for state in states), encoding="utf-8"
        )
        text = [
            "# RightDesk primary helpdesk simulation",
            "",
            f"Tickets: {len(states)}; public events: {len(events)}; "
            f"questions: {counts['question_sent']}.",
            "Final states: " + encode(report["state_counts"]),
            "Event counts: " + encode(dict(counts)),
            f"Maximum questions per ticket: {report['max_questions_per_ticket']}.",
            "Event replay matches persisted ticket projections. Time and service are simulated.",
            "The primary router receives only arrived customer messages and its own history.",
            "Future scenario replies and expected outcomes stay in the driver "
            "until their scheduled events.",
            "No source eventual-owner labels, human-review labels, final-test inputs, "
            "or shadow model proposals drive the incumbent.",
            "Synthetic lifecycle assertions are software checks, "
            "not human usefulness or routing-accuracy evidence.",
            "",
            "Scenario checks: " + encode(report.get("scenario_checks", {})),
            "Public events are ordered by event_id and can be read incrementally "
            "by an independent observer. "
            "Start an independent worker with `rightdesk shadow run SIMULATION_ID`.",
            "",
        ]
        (staging / "report.md").write_text("\n".join(text), encoding="utf-8")
        report["artifacts"] = {p.name: checksum(p) for p in staging.iterdir() if p.is_file()}
        (staging / "run.json").write_text(encode(report) + "\n", encoding="utf-8")
        staging.rename(output)
    return output / "report.md"


def run_demo(simulator: Simulator, config_path: Path, scenario_path: Path, output: Path):
    if output.exists():
        raise ValueError("Simulation output exists; choose a fresh directory")
    bundle = _load(scenario_path, ScenarioBundle)
    if len({case.case_id for case in bundle.cases}) != len(bundle.cases):
        raise ValueError("Scenario case IDs must be unique")
    simulator.initialize()
    simulation_id = simulator.create(
        config_path,
        bundle.version,
        {
            "kind": "synthetic_scenario",
            "version": bundle.version,
            "sha256": checksum(scenario_path),
        },
    )
    config = simulator.snapshot(simulation_id)["config"]
    plan = []
    for index, case in enumerate(bundle.cases):
        at = index * config["arrival_gap_seconds"]
        plan.append((at, 0, case.case_id, 0, case))
        plan.extend(
            (at + reply.after_seconds, 1, case.case_id, i, reply)
            for i, reply in enumerate(case.replies)
        )
        plan.extend(
            (at + action.after_seconds, 2, case.case_id, i, action)
            for i, action in enumerate(case.agent_actions)
        )
    clock = 0
    for step, (at, kind, case_id, index, payload) in enumerate(
        sorted(plan, key=lambda item: item[:4])
    ):
        simulator.advance(simulation_id, at - clock, f"demo-clock:{step}")
        clock = at
        if kind == 0:
            ticket = TicketInput(
                case_id=case_id,
                conversation_id=case_id,
                text=payload.text,
                language=payload.language,
                channel="chat",
            )
            simulator.intake(simulation_id, ticket, "demo-intake:" + case_id)
        elif kind == 1:
            state = simulator.ticket(simulation_id, case_id)
            simulator.reply(
                simulation_id, case_id, payload.text, state.version, f"demo-reply:{case_id}:{index}"
            )
        else:
            state = simulator.ticket(simulation_id, case_id)
            simulator.agent(
                simulation_id,
                case_id,
                payload.action,
                payload.queue,
                "scripted-agent",
                state.version,
                f"demo-agent:{case_id}:{index}",
            )
    simulator.advance(
        simulation_id, config["customer_timeout_seconds"] + config["service_seconds"], "demo-drain"
    )
    checks = {}
    for case in bundle.cases:
        state = simulator.ticket(simulation_id, case.case_id)
        expected_questions = [case.expected_question] if case.expected_question else []
        checks[case.case_id] = (
            state.status == case.expected_status
            and state.assigned_queue == case.expected_queue
            and state.asked_question_ids == expected_questions
        )
    if not all(checks.values()):
        raise ValueError("Synthetic lifecycle expectations failed: " + encode(checks))
    return export_simulation(
        simulator,
        simulation_id,
        output,
        {
            "source_kind": "synthetic",
            "scenario_sha256": checksum(scenario_path),
            "scenario_checks": checks,
        },
    )


def replay_development(
    simulator: Simulator, config_path: Path, store: Store, dataset_id: str, limit: int, output: Path
):
    if output.exists() or type(limit) is not int or not 1 <= limit <= 5000:
        raise ValueError("Choose a fresh replay output and a limit between 1 and 5000")
    dataset = store.dataset(dataset_id)
    _, workflow, _, _ = load_simulation(config_path)
    source_queues = {q["id"]: q["description"] for q in dataset["workflow"]["queues"]}
    if dataset["workflow"]["workflow_id"] != workflow.workflow_id or source_queues != {
        q.id: q.description for q in workflow.queues
    }:
        raise ValueError("Replay dataset and simulator queue definitions differ")
    # This query intentionally excludes reference_labels, reviews, predictions, and final test.
    rows = store.query(
        "SELECT ticket_json FROM cases WHERE dataset_id=? AND partition_role=?",
        (dataset_id, "development"),
    )
    if not rows:
        raise ValueError("No development tickets to replay")
    simulator.initialize()
    simulation_id = simulator.create(
        config_path,
        "ABCD development replay",
        {"kind": "abcd_development", "dataset_id": dataset_id},
    )
    config = simulator.snapshot(simulation_id)["config"]
    tickets = [TicketInput.model_validate_json(row["ticket_json"]) for row in rows]
    tickets.sort(
        key=lambda ticket: hashlib.sha256(f"{config['seed']}:{ticket.case_id}".encode()).hexdigest()
    )
    for index, ticket in enumerate(tickets[:limit]):
        if index:
            simulator.advance(simulation_id, config["arrival_gap_seconds"], f"replay-clock:{index}")
        simulator.intake(simulation_id, ticket, "replay-intake:" + ticket.case_id)
    simulator.advance(
        simulation_id,
        config["customer_timeout_seconds"] + config["service_seconds"],
        "replay-drain",
    )
    return export_simulation(
        simulator,
        simulation_id,
        output,
        {
            "source_kind": "abcd_development",
            "dataset_id": dataset_id,
            "scripted_customer_replies": False,
        },
    )
