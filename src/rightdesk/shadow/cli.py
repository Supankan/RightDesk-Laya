"""Explicit model work runs in a separate CLI process, never on dashboard refresh."""

import time
from pathlib import Path

import typer

from rightdesk.shadow.engine import PublicStream, ShadowStore, ShadowWorker, export_shadow
from rightdesk.workbench_cli import command_errors, show

shadow_app = typer.Typer(
    no_args_is_help=True, help="Observe the primary helpdesk without writing to it."
)
final_app = typer.Typer(no_args_is_help=True, help="Freeze and run the final held-out evaluation.")
SHADOW_DB = Path("artifacts/shadow/rightdesk.sqlite3")
PRIMARY_DB = Path("artifacts/simulator/rightdesk.sqlite3")
SOURCE = Path("artifacts/laya-development-v1")
CALIBRATION = Path("artifacts/calibration-v1")
ADAPTER = Path("configs/models/laya_v1.yaml")


@shadow_app.command("run")
def run_command(
    simulation_id: str,
    database: Path = SHADOW_DB,
    primary: Path = PRIMARY_DB,
    source: Path = SOURCE,
    calibration: Path = CALIBRATION,
    adapter: Path = ADAPTER,
    models: str = "majority,rules,tfidf_logistic,laya",
    name: str = "shadow-v1",
    max_events: int = 10000,
    follow: bool = False,
    poll_seconds: float = 2,
):
    """Drain the public stream once, or follow new events until Ctrl+C. Restart is idempotent."""
    with command_errors():
        if not 0.1 <= poll_seconds <= 30:
            raise ValueError("poll-seconds must be 0.1–30")
        stream, store = PublicStream(primary, simulation_id), ShadowStore(database)
        if stream.path == store.path:
            raise ValueError("Shadow and primary databases must be different")
        from rightdesk.shadow.models import FrozenPredictor

        predictor = FrozenPredictor(
            stream.snapshot(),
            source,
            calibration,
            adapter,
            tuple(part.strip() for part in models.split(",")),
        )
        store.initialize()
        worker = ShadowWorker(stream, store, predictor, name)
        while True:
            show(worker.run(max_events))
            if not follow:
                break
            time.sleep(poll_seconds)


@shadow_app.command("list")
def list_command(database: Path = SHADOW_DB):
    with command_errors():
        show(ShadowStore(database).observers())


@shadow_app.command("export")
def export_command(observer_id: str, output: Path, database: Path = SHADOW_DB):
    with command_errors():
        typer.echo(export_shadow(ShadowStore(database), observer_id, output))


@final_app.command("freeze")
def freeze_command(
    output: Path,
    source: Path = SOURCE,
    calibration: Path = CALIBRATION,
    adapter: Path = ADAPTER,
    manifest: Path = Path("data/manifests/abcd-partitions-v1.json"),
):
    with command_errors():
        from rightdesk.evaluation.final import freeze_protocol

        typer.echo(freeze_protocol(source, calibration, manifest, adapter, output))


@final_app.command("run")
def final_command(protocol: Path, output: Path):
    with command_errors():
        from rightdesk.evaluation.final import final_evaluate

        typer.echo(final_evaluate(protocol, output))
