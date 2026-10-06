from pathlib import Path

import pytest
from typer.testing import CliRunner

from rightdesk.cli import app
from rightdesk.config import ConfigurationError, load_experiment, load_workflow, resolve_config_path

ROOT = Path(__file__).resolve().parents[1]


def test_templates_and_cli_from_other_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = ROOT / "configs/experiments/baseline.yaml"
    experiment = load_experiment(path)
    workflow = load_workflow(resolve_config_path(path, experiment.workflow))
    assert len(workflow.queues) == 6
    assert workflow.policy.routing_threshold is None
    assert workflow.policy.clarification_enabled is False
    assert {model.id for model in experiment.models} == {
        "majority",
        "rules",
        "tfidf_logistic",
        "laya",
    }
    assert resolve_config_path(path, experiment.output_directory) == ROOT / "artifacts"
    result = CliRunner().invoke(app, ["validate-config", str(path), "--experiment"])
    assert result.exit_code == 0, result.output
    assert "Valid experiment template" in result.output


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "[]",
        "workflow_id: [",
        "workflow_id: a\nworkflow_id: b",
        "policy:\n  version: '1'\n  version: '2'",
        "!!python/object:builtins.object {}",
        "? [a, b]\n: value",
        "workflow_id: a\nunknown_field: 1",
    ],
)
def test_bad_yaml_has_source_and_cli_failure(tmp_path, contents):
    path = tmp_path / "bad.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="bad.yaml"):
        load_workflow(path)
    result = CliRunner().invoke(app, ["validate-config", str(path)])
    assert result.exit_code == 1
    assert "bad.yaml" in result.output


def test_missing_config(tmp_path):
    with pytest.raises(ConfigurationError, match="missing.yaml"):
        load_workflow(tmp_path / "missing.yaml")
