import json

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("laya")

from test_benchmark import benchmark_config as source_benchmark_config
from test_laya import ROOT, FakeAgent

from rightdesk.evaluation.benchmark import benchmark_baselines
from rightdesk.evaluation.laya_benchmark import benchmark_laya
from rightdesk.models.laya_model import LayaError, LayaModel

benchmark_config = source_benchmark_config


def fake_load(self):
    self.agent, self.fitted = FakeAgent(), True
    self.metadata = {"test_double": True}


def test_laya_comparison_matches_saved_baseline_cases(benchmark_config, tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    benchmark_baselines(benchmark_config, baseline, bootstrap_samples=20)
    monkeypatch.setattr(LayaModel, "load", fake_load)
    report = benchmark_laya(
        benchmark_config,
        ROOT / "configs/models/laya_v1.yaml",
        baseline,
        tmp_path / "comparison",
        lambda message: None,
    )
    run = json.loads((report.parent / "run.json").read_text())
    assert run["models_run"][-1] == "laya"
    assert run["metrics"]["laya"]["cases"] == 2
    assert run["metrics"]["laya"]["failed_cases"] == 0
    assert (report.parent / "laya/raw_responses.jsonl").exists()
    assert "not directly comparable" in run["latency_method"]


def test_inference_failure_is_not_a_completed_report(benchmark_config, tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    benchmark_baselines(benchmark_config, baseline, bootstrap_samples=20)
    monkeypatch.setattr(LayaModel, "load", fake_load)

    def fail(self, tickets):
        raise LayaError("injected model failure")

    monkeypatch.setattr(LayaModel, "predict_batch", fail)
    with pytest.raises(LayaError, match="injected"):
        benchmark_laya(
            benchmark_config,
            ROOT / "configs/models/laya_v1.yaml",
            baseline,
            tmp_path / "failed",
            lambda message: None,
        )
    failed = json.loads((tmp_path / "failed/failed_run.json").read_text())
    assert failed["completed_cases"] == 0
    assert not (tmp_path / "failed/run.json").exists()
