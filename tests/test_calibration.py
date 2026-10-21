"""Calibration isolation, stable probability transforms, and honest threshold behavior."""

from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np

from rightdesk.config import load_workflow
from rightdesk.evaluation.calibration import (
    CalibrationConfig,
    TemperatureCalibrator,
    assign_roles,
    routed_mask,
    routing_summary,
    select_threshold,
)
from rightdesk.schemas import ModelPrediction, TicketInput

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workflow():
    return load_workflow(ROOT / "configs/workflows/retail_support.yaml")


def prediction(workflow, identifier, score=0.9, revision="test-v1"):
    return ModelPrediction.for_workflow(
        {
            "case_id": str(identifier),
            "workflow_id": workflow.workflow_id,
            "workflow_version": workflow.version,
            "model_revision": revision,
            "queue_probabilities": {
                "account": score,
                **{q: (1 - score) / 5 for q in sorted(workflow.queue_ids - {"account"})},
            },
        },
        workflow,
    )


def test_role_allocation_keeps_duplicates_together_and_is_order_independent():
    tickets = [
        TicketInput(case_id=str(i), conversation_id=str(i), text=f"message {i}", channel="chat")
        for i in range(100)
    ]
    tickets += [
        TicketInput(
            case_id="duplicate", conversation_id="dup", text=" MESSAGE   0 ", channel="chat"
        )
    ]
    config = CalibrationConfig(version="v1")
    roles = assign_roles(tickets, config)
    assert roles[0] == roles[-1]
    assert set(roles) == {"fit", "selection", "validation"}
    assert assign_roles(list(reversed(tickets)), config) == list(reversed(roles))
    with pytest.raises(ValueError, match="Duplicate"):
        assign_roles([tickets[0], tickets[0]], config)


def test_temperature_improves_fit_nll_preserves_choices_and_roundtrips(workflow):
    rows = [prediction(workflow, i, 0.99) for i in range(20)]
    targets = ["account"] * 10 + ["returns_refunds"] * 10
    calibrator = TemperatureCalibrator.fit(rows, targets, "v1")
    assert calibrator.temperature > 1
    assert calibrator.nll_after < calibrator.nll_before
    loaded = TemperatureCalibrator.model_validate_json(calibrator.model_dump_json())
    result = loaded.transform(rows[0], workflow)
    assert result.selected_queue() == rows[0].selected_queue()
    assert max(result.queue_probabilities.values()) < 0.99
    assert sum(result.queue_probabilities.values()) == pytest.approx(1)
    assert result.calibration_version == "v1"
    assert result.sufficient_information_probability is None
    with pytest.raises(ValueError, match="already calibrated"):
        loaded.transform(result, workflow)
    with pytest.raises(ValueError, match="mismatch"):
        loaded.transform(prediction(workflow, "wrong", revision="other"), workflow)


def test_zeros_and_equal_probability_ties_remain_valid(workflow):
    data = prediction(workflow, "one", 1.0).model_dump()
    exact = ModelPrediction.for_workflow(data, workflow)
    calibrator = TemperatureCalibrator.fit([exact], ["account"], "v1")
    result = calibrator.transform(exact, workflow)
    assert np.isfinite(list(result.queue_probabilities.values())).all()
    tied = prediction(workflow, "tie", 1 / 6)
    tied_data = tied.model_dump()
    tied_data["queue_probabilities"] = dict.fromkeys(workflow.queue_ids, 1 / 6)
    tied_data["predicted_queue"] = "technical_support"
    tied = ModelPrediction.for_workflow(tied_data, workflow)
    assert calibrator.transform(tied, workflow).selected_queue() == "technical_support"


def test_threshold_uses_max_coverage_without_splitting_ties(workflow):
    rows = [prediction(workflow, i, score) for i, score in enumerate([0.99, 0.99, 0.8, 0.5])]
    config = CalibrationConfig(version="v1", min_routed_groups=2, target_error=0.3)
    targets = ["account", "account", "returns_refunds", "returns_refunds"]
    selection = select_threshold(rows, targets, ["a", "b", "c", "d"], config)
    assert selection["threshold"] == 0.99
    assert selection["selection"]["auto_count"] == 2
    assert routed_mask(rows, selection["threshold"]).tolist() == [True, True, False, False]
    assert routed_mask(rows, None).tolist() == [False] * 4


def test_no_eligible_threshold_and_sparse_zero_errors_do_not_claim_safety(workflow):
    rows = [prediction(workflow, i) for i in range(3)]
    config = CalibrationConfig(version="v1", min_routed_groups=2)
    assert select_threshold(rows, ["returns_refunds"] * 3, list("abc"), config)["threshold"] is None
    sparse = routing_summary(rows, ["account"] * 3, list("abc"), [True] * 3)
    assert sparse["routing_error"] == 0
    assert sparse["group_error_upper"] > 0.6
    empty = routing_summary(rows, ["account"] * 3, list("abc"), [False] * 3)
    assert empty["routing_error"] is None
    assert empty["group_error_upper"] is None


def test_duplicate_group_counts_any_error_once(workflow):
    rows = [prediction(workflow, i) for i in range(4)]
    point = routing_summary(
        rows,
        ["account", "returns_refunds", "account", "account"],
        ["duplicate", "duplicate", "other", "third"],
        [True] * 4,
    )
    assert point["error_count"] == 1
    assert point["routed_groups"] == 3
    assert point["groups_with_any_error"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"fit_fraction": 0.8, "selection_fraction": 0.3},
        {"target_error": 0.0},
        {"min_routed_groups": 1},
        {"family_alpha": 1.0},
    ],
)
def test_invalid_calibration_settings(changes):
    with pytest.raises(ValueError):
        CalibrationConfig(version="v1", **changes)
