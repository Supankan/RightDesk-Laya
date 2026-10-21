"""Explicit primary simulation commands and offline clarification evaluation."""

from pathlib import Path
from uuid import uuid4

import typer

from rightdesk.evaluation.clarification import evaluate_clarification
from rightdesk.schemas import TicketInput
from rightdesk.simulation.engine import Simulator
from rightdesk.simulation.scenarios import export_simulation, replay_development, run_demo
from rightdesk.storage import Store
from rightdesk.workbench_cli import DEFAULT_DATABASE, command_errors, show

SIM_DATABASE = Path("artifacts/simulator/rightdesk.sqlite3")
SIM_CONFIG = Path("configs/simulator/retail_v1.yaml")
sim_app = typer.Typer(no_args_is_help=True, help="Operate the primary simulated helpdesk.")
clarification_app = typer.Typer(no_args_is_help=True, help="Evaluate bounded question proposals.")


@clarification_app.command("evaluate")
def evaluate_command(
    run_id: str,
    model_id: str,
    output: Path,
    catalog: Path = Path("configs/clarification/retail_v1.yaml"),
    database: Path = DEFAULT_DATABASE,
):
    with command_errors():
        typer.echo(evaluate_clarification(Store(database), run_id, model_id, catalog, output))


@sim_app.command("init")
def init_command(database: Path = SIM_DATABASE):
    with command_errors():
        typer.echo(Simulator(database).initialize())


@sim_app.command("create")
def create_command(
    name: str = "Manual helpdesk", config: Path = SIM_CONFIG, database: Path = SIM_DATABASE
):
    with command_errors():
        simulator = Simulator(database)
        simulator.initialize()
        typer.echo(simulator.create(config, name))


@sim_app.command("list")
def list_command(database: Path = SIM_DATABASE):
    with command_errors():
        show(Simulator(database).sessions())


@sim_app.command("tickets")
def tickets_command(simulation_id: str, database: Path = SIM_DATABASE):
    with command_errors():
        show([state.model_dump() for state in Simulator(database).tickets(simulation_id)])


@sim_app.command("intake")
def intake_command(
    simulation_id: str, ticket: Path, command_id: str | None = None, database: Path = SIM_DATABASE
):
    with command_errors():
        request = TicketInput.model_validate_json(ticket.read_text(encoding="utf-8"))
        show(Simulator(database).intake(simulation_id, request, command_id or uuid4().hex))


@sim_app.command("reply")
def reply_command(
    simulation_id: str,
    case_id: str,
    text: str,
    version: int,
    command_id: str | None = None,
    database: Path = SIM_DATABASE,
):
    with command_errors():
        show(
            Simulator(database).reply(
                simulation_id, case_id, text, version, command_id or uuid4().hex
            )
        )


@sim_app.command("agent")
def agent_command(
    simulation_id: str,
    case_id: str,
    action: str,
    version: int,
    queue: str | None = None,
    agent_id: str = "operator",
    command_id: str | None = None,
    database: Path = SIM_DATABASE,
):
    with command_errors():
        show(
            Simulator(database).agent(
                simulation_id, case_id, action, queue, agent_id, version, command_id or uuid4().hex
            )
        )


@sim_app.command("advance")
def advance_command(
    simulation_id: str,
    seconds: int,
    expected_clock: int | None = None,
    command_id: str | None = None,
    database: Path = SIM_DATABASE,
):
    with command_errors():
        show(
            Simulator(database).advance(
                simulation_id, seconds, command_id or uuid4().hex, expected_clock
            )
        )


@sim_app.command("events")
def events_command(
    simulation_id: str, after: int = 0, limit: int = 500, database: Path = SIM_DATABASE
):
    with command_errors():
        events = Simulator(database).events(simulation_id, after, limit)
        show(
            {
                "events": [event.model_dump() for event in events],
                "next_cursor": events[-1].event_id if events else after,
            }
        )


@sim_app.command("verify")
def verify_command(simulation_id: str, database: Path = SIM_DATABASE):
    with command_errors():
        show(Simulator(database).verify(simulation_id))


@sim_app.command("export")
def export_command(simulation_id: str, output: Path, database: Path = SIM_DATABASE):
    with command_errors():
        typer.echo(export_simulation(Simulator(database), simulation_id, output))


@sim_app.command("demo")
def demo_command(
    output: Path = Path("artifacts/simulator/demo-v1"),
    config: Path = SIM_CONFIG,
    scenarios: Path = Path("configs/simulator/demo_v1.yaml"),
    database: Path = SIM_DATABASE,
):
    with command_errors():
        typer.echo(run_demo(Simulator(database), config, scenarios, output))


@sim_app.command("replay")
def replay_command(
    dataset_id: str,
    output: Path = Path("artifacts/simulator/development-v1"),
    limit: int = 50,
    config: Path = SIM_CONFIG,
    database: Path = SIM_DATABASE,
    catalog_database: Path = DEFAULT_DATABASE,
):
    with command_errors():
        typer.echo(
            replay_development(
                Simulator(database), config, Store(catalog_database), dataset_id, limit, output
            )
        )
