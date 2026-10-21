"""Experiment catalog and human-review commands; shared services also power the UI."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import typer

from rightdesk.exports import export_run
from rightdesk.review import (
    QueueRequest,
    ReviewSubmission,
    agreement,
    create_queue,
    export_reviews,
    list_queues,
    submit_review,
)
from rightdesk.services import import_dataset, register_run, reviewed_evaluation, verify_run
from rightdesk.storage import Store

DEFAULT_DATABASE = Path("artifacts/workbench/rightdesk.sqlite3")
store_app = typer.Typer(no_args_is_help=True, help="Register and export completed experiments.")
review_app = typer.Typer(
    no_args_is_help=True, help="Create blind queues and record human judgments."
)


@contextmanager
def command_errors():
    try:
        yield
    except (ValueError, KeyError, TypeError, OSError, ImportError, sqlite3.Error) as exc:
        typer.echo(f"Workbench operation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def show(value):
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


@store_app.command("init")
def initialize(database: Path = DEFAULT_DATABASE):
    with command_errors():
        typer.echo(f"Initialized: {Store(database).initialize()}")


@store_app.command("import-dataset")
def dataset_command(
    manifest: Path = Path("data/manifests/abcd-partitions-v1.json"),
    workflow: Path = Path("configs/workflows/retail_support.yaml"),
    database: Path = DEFAULT_DATABASE,
):
    with command_errors():
        typer.echo(import_dataset(Store(database), manifest, workflow))


@store_app.command("register-run")
def register_command(directory: Path, database: Path = DEFAULT_DATABASE):
    with command_errors():
        typer.echo(register_run(Store(database), directory))


@store_app.command("list-runs")
def runs_command(database: Path = DEFAULT_DATABASE):
    with command_errors():
        show(Store(database).list_runs())


@store_app.command("verify-run")
def verify_command(run_id: str, database: Path = DEFAULT_DATABASE):
    with command_errors():
        show(verify_run(Store(database), run_id))


@store_app.command("export-run")
def export_command(run_id: str, output: Path, database: Path = DEFAULT_DATABASE):
    with command_errors():
        typer.echo(export_run(Store(database), run_id, output))


@store_app.command("bootstrap")
def bootstrap_command(root: Path = Path("."), database: Path = DEFAULT_DATABASE):
    """Index existing local development/calibration runs and prepare a 50-case blind queue."""
    with command_errors():
        store = Store(database)
        store.initialize()
        dataset_id = import_dataset(
            store,
            root / "data/manifests/abcd-partitions-v1.json",
            root / "configs/workflows/retail_support.yaml",
        )
        registered = []
        for name in ("baseline-development-v1", "laya-development-v1", "calibration-v1"):
            directory = root / "artifacts" / name
            if (directory / "run.json").exists():
                registered.append(register_run(store, directory))
        queue_id = create_queue(
            store, QueueRequest(title="Development intake review", dataset_id=dataset_id, limit=50)
        )
        show({"dataset_id": dataset_id, "run_ids": registered, "review_queue": queue_id})


@review_app.command("create-queue")
def queue_command(
    dataset_id: str,
    title: str = "Intake review",
    strategy: str = "random",
    partition: str = "development",
    run_id: str | None = None,
    model_id: str | None = None,
    comparison_model: str | None = None,
    seed: int = 42,
    limit: int = 50,
    database: Path = DEFAULT_DATABASE,
):
    with command_errors():
        request = QueueRequest(
            dataset_id=dataset_id,
            title=title,
            strategy=strategy,
            partition=partition,
            run_id=run_id,
            model_id=model_id,
            comparison_model=comparison_model,
            seed=seed,
            limit=limit,
        )
        typer.echo(create_queue(Store(database), request))


@review_app.command("list-queues")
def queues_command(database: Path = DEFAULT_DATABASE):
    with command_errors():
        show(list_queues(Store(database)))


@review_app.command("submit")
def submit_command(submission: Path, database: Path = DEFAULT_DATABASE):
    """Save a human-authored ReviewSubmission JSON file with an expected revision."""
    with command_errors():
        request = ReviewSubmission.model_validate_json(submission.read_text(encoding="utf-8"))
        show({"revision": submit_review(Store(database), request)})


@review_app.command("export")
def reviews_command(
    dataset_id: str, output: Path, history: bool = False, database: Path = DEFAULT_DATABASE
):
    with command_errors():
        contents = export_reviews(Store(database), dataset_id, history)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(contents)
        typer.echo(f"Exported reviews: {output}")


@review_app.command("agreement")
def agreement_command(dataset_id: str, database: Path = DEFAULT_DATABASE):
    with command_errors():
        show(agreement(Store(database), dataset_id))


@review_app.command("evaluate")
def evaluate_command(run_id: str, model_id: str, database: Path = DEFAULT_DATABASE):
    with command_errors():
        show(reviewed_evaluation(Store(database), run_id, model_id))
