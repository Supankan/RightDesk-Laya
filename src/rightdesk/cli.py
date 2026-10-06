"""Configuration checks and dataset preparation; experiment execution comes later."""

from pathlib import Path
from typing import Annotated

import typer

from rightdesk.config import (
    ConfigurationError,
    load_experiment,
    load_workflow,
    resolve_config_path,
)
from rightdesk.data.importers.abcd import REVISION, IngestionError, download_abcd, import_abcd
from rightdesk.data.splits import prepare_partitions

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True, help="Download and prepare research datasets.")
app.add_typer(data_app, name="data")


@data_app.command("download-abcd")
def download_abcd_command(
    data_dir: Path = Path("data"),
    revision: str = REVISION,
) -> None:
    """Explicitly download the pinned ABCD archive and license, or verify cached files."""
    try:
        manifest = download_abcd(data_dir, revision)
        typer.echo(f"Verified source manifest: {manifest}")
    except (IngestionError, OSError, EOFError) as exc:
        typer.echo(f"ABCD download failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@data_app.command("import-abcd")
def import_abcd_command(
    source_manifest: Path = Path(f"data/manifests/abcd-source-{REVISION}.json"),
    output_dir: Path = Path("data/processed/abcd-intake-v1"),
    manifest: Path = Path("data/manifests/abcd-intake-v1.json"),
) -> None:
    """Prepare intake records and separate source references without network access."""
    try:
        result = import_abcd(source_manifest, output_dir, manifest)
        typer.echo(f"Prepared ABCD intake manifest: {result}")
    except (IngestionError, OSError) as exc:
        typer.echo(f"ABCD import failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.callback()
def main() -> None:
    """RightDesk support-intake evaluation workbench."""


@app.command("download-laya")
def download_laya_command(config: Path = Path("configs/models/laya_v1.yaml")) -> None:
    """Explicitly download and checksum the pinned English checkpoint."""
    try:
        from rightdesk.models.laya_model import download_laya

        typer.echo(f"Verified Laya checkpoint: {download_laya(config)}")
    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"Laya download failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("benchmark-laya")
def benchmark_laya_command(
    config: Path = Path("configs/experiments/baseline.yaml"),
    adapter: Path = Path("configs/models/laya_v1.yaml"),
    baseline: Path = Path("artifacts/baseline-development-v1"),
    output: Path = Path("artifacts/laya-development-v1"),
) -> None:
    """Compare pinned local Laya with an existing baseline run on identical cases."""
    try:
        from rightdesk.evaluation.laya_benchmark import benchmark_laya

        result = benchmark_laya(config, adapter, baseline, output, typer.echo)
        typer.echo(f"Laya comparison: {result}")
    except (ImportError, ValueError, OSError, RuntimeError, KeyError) as exc:
        typer.echo(f"Laya benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("benchmark-baselines")
def benchmark_baselines_command(
    config: Path = Path("configs/experiments/baseline.yaml"),
    output: Path = Path("artifacts/baseline-development-v1"),
    split: str = "development",
    bootstrap_samples: int = 500,
) -> None:
    """Fit baselines on train and write a development report (requires baseline extra)."""
    try:
        from rightdesk.evaluation.benchmark import benchmark_baselines

        result = benchmark_baselines(config, output, split, bootstrap_samples)
        typer.echo(f"Baseline report: {result}")
    except ImportError as exc:
        typer.echo(f"Install baseline dependencies: uv sync --extra baseline. {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except (ValueError, OSError, KeyError, TypeError) as exc:
        typer.echo(f"Baseline benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@data_app.command("prepare-abcd")
def prepare_abcd_command(config: Path = Path("configs/data/abcd_partitions_v1.yaml")) -> None:
    """Map source labels and prepare isolated train/development/calibration/test data."""
    try:
        result = prepare_partitions(config)
        typer.echo(f"Prepared ABCD partitions: {result}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        typer.echo(f"ABCD partition preparation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def validate_config(
    path: Path,
    experiment: Annotated[
        bool, typer.Option(help="Validate an experiment and its workflow.")
    ] = False,
) -> None:
    """Check configuration contracts without downloading data or running models."""
    try:
        if experiment:
            config = load_experiment(path)
            workflow = load_workflow(resolve_config_path(path, config.workflow))
            typer.echo(f"Valid experiment template: {config.experiment_id}")
            typer.echo("This command checks configuration structure, not experiment readiness.")
        else:
            workflow = load_workflow(path)
        typer.echo(
            f"Valid workflow: {workflow.workflow_id} v{workflow.version}; "
            f"{len(workflow.queues)} queues; mode={workflow.execution_mode}"
        )
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
