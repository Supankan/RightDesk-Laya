import json

import pytest

pytest.importorskip("numpy")

from rightdesk.evaluation.metrics import evaluate
from rightdesk.schemas import ModelPrediction, WorkflowConfig


def predictions(probabilities):
    workflow = WorkflowConfig.model_validate(
        {
            "workflow_id": "tiny",
            "version": "1",
            "queues": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
            "policy": {"version": "1"},
        }
    )
    return [
        ModelPrediction.for_workflow(
            {
                "case_id": str(i),
                "workflow_id": "tiny",
                "workflow_version": "1",
                "model_revision": "fixture",
                **(
                    {"queue_probabilities": {"a": p, "b": 1 - p}}
                    if p is not None
                    else {"abstention_reason": "no_rule_match"}
                ),
            },
            workflow,
        )
        for i, p in enumerate(probabilities)
    ]


def test_metrics_match_hand_calculation():
    result = evaluate(
        predictions([0.8, 0.4, 0.3, 0.5]),
        ["a", "a", "b", "b"],
        ["1", "2", "3", "4"],
        ["a", "b"],
        [1, 2, 3, 4],
    )
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == 0.5
    assert result["confusion_matrix"]["counts"] == [[1, 1, 0], [1, 1, 0]]
    assert result["brier_score"] == pytest.approx(0.37)
    assert result["calibration_error"] == pytest.approx(0.4)
    curve = result["risk_coverage_curve"]
    assert [row["auto_count"] for row in curve] == [0, 1, 2, 3, 4]
    assert curve[2]["routing_error"] == 0
    assert curve[3]["routing_error"] == pytest.approx(1 / 3)
    assert curve[0]["routing_error"] is None
    json.dumps(result, allow_nan=False)


def test_ties_are_not_split_and_cluster_uncertainty_is_reproducible():
    args = (
        predictions([0.8] * 20),
        ["a"] * 10 + ["b"] * 10,
        ["g1"] * 10 + ["g2"] * 10,
        ["a", "b"],
        [0.1] * 20,
    )
    result = evaluate(*args)
    assert result == evaluate(*args)
    assert len(result["risk_coverage_curve"]) == 2
    assert result["accuracy_ci95"] == [0.0, 1.0]
    assert result["duplicate_groups"] == 2


def test_all_abstentions_have_unknown_risk_and_probability_metrics():
    result = evaluate(predictions([None, None]), ["a", "b"], ["1", "2"], ["a", "b"], [0, 0])
    assert result["operating_point"]["automation_coverage"] == 0
    assert result["operating_point"]["routing_error"] is None
    assert result["brier_score"] is None
    assert result["macro_f1"] == 0
    assert result["confusion_matrix"]["counts"] == [[0, 0, 1], [0, 0, 1]]
    json.dumps(result, allow_nan=False)


def test_one_group_has_no_uncertainty_estimate():
    result = evaluate(predictions([0.8, 0.8]), ["a", "a"], ["same", "same"], ["a", "b"], [0, 0])
    assert result["accuracy_ci95"] is None


def test_confidence_one_is_in_last_bin():
    result = evaluate(predictions([1.0, 0.0]), ["a", "b"], ["1", "2"], ["a", "b"], [0, 0])
    assert result["reliability_bins"][-1]["count"] == 2
    assert result["brier_score"] == 0


def test_mismatched_arrays_rejected():
    with pytest.raises(ValueError, match="equal nonzero"):
        evaluate(predictions([0.8]), [], ["1"], ["a", "b"], [0])
