"""Boundary checks that protect labels, policy actions, and probability evaluation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rightdesk.config import load_workflow
from rightdesk.schemas import (
    ModelPrediction,
    PolicyDecision,
    ReviewedLabel,
    TicketInput,
    WorkflowConfig,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workflow():
    return load_workflow(ROOT / "configs/workflows/retail_support.yaml")


def identity(workflow):
    return dict(
        case_id="case-1", workflow_id=workflow.workflow_id, workflow_version=workflow.version
    )


def prediction(workflow):
    return {
        **identity(workflow),
        "queue_probabilities": {queue: 1 / 6 for queue in workflow.queue_ids},
        "model_revision": "baseline-v1",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"text": "  \n"},
        {"case_id": ""},
        {"channel": "phone"},
        {"text": 123},
        {"eventual_owner_queue": "account"},
    ],
)
def test_ticket_rejects_bad_inputs_and_label_leakage(change):
    with pytest.raises(ValidationError):
        TicketInput.model_validate(
            {
                "case_id": "1",
                "conversation_id": "c1",
                "text": "Where is my order?",
                "channel": "chat",
                **change,
            }
        )


def test_ticket_preserves_text_and_unknown_language():
    ticket = TicketInput(case_id="1", conversation_id="c1", text="  Help?\n", channel="chat")
    assert ticket.text == "  Help?\n"
    assert ticket.language is None


def test_prediction_unknowns_and_round_trip(workflow):
    result = ModelPrediction.for_workflow(prediction(workflow), workflow)
    assert result.sufficient_information_probability is None
    assert result.calibration_version is None
    assert (
        ModelPrediction.model_validate_json(
            result.model_dump_json(),
            context={"workflow": workflow},
        )
        == result
    )


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), "0.5", True])
def test_invalid_probabilities(workflow, value):
    data = prediction(workflow)
    data["queue_probabilities"]["account"] = value
    with pytest.raises(ValidationError):
        ModelPrediction.for_workflow(data, workflow)


@pytest.mark.parametrize(
    "change",
    [
        {"queue_probabilities": {"account": 1.0}},
        {"suggested_question_id": "unknown"},
        {"workflow_version": "wrong"},
        {"workflow_id": "other_workflow"},
        {"sufficient_information_probability": -1},
    ],
)
def test_prediction_workflow_references(workflow, change):
    with pytest.raises(ValidationError):
        ModelPrediction.for_workflow({**prediction(workflow), **change}, workflow)


def test_probability_keys_and_sum(workflow):
    for probabilities in (
        {queue: 0.1 for queue in workflow.queue_ids},
        {**prediction(workflow)["queue_probabilities"], "extra_queue": 0.0},
    ):
        with pytest.raises(ValidationError):
            ModelPrediction.for_workflow(
                {**prediction(workflow), "queue_probabilities": probabilities}, workflow
            )


def test_workflow_context_is_required(workflow):
    with pytest.raises(ValidationError, match="workflow context is required"):
        ModelPrediction.model_validate(prediction(workflow))


@pytest.mark.parametrize("section", ["queues", "questions"])
def test_duplicate_definitions(workflow, section):
    data = workflow.model_dump()
    data[section].append(data[section][0])
    with pytest.raises(ValidationError, match="unique"):
        WorkflowConfig.model_validate(data)


def test_policy_settings(workflow):
    for changes in (
        {"max_questions": -1},
        {"routing_threshold": 1.1},
        {"clarification_enabled": True, "max_questions": 0},
        {"clarification_enabled": "false"},
    ):
        data = workflow.model_dump()
        data["policy"].update(changes)
        with pytest.raises(ValidationError):
            WorkflowConfig.model_validate(data)


@pytest.mark.parametrize(
    "action,queue,question,reason",
    [
        ("route", "account", None, "explicit_rule"),
        ("review", None, None, "low_confidence"),
        ("clarify", None, "charge_or_refund", "missing_information"),
    ],
)
def test_valid_actions(workflow, action, queue, question, reason):
    data = workflow.model_dump()
    data["policy"]["clarification_enabled"] = True
    workflow = WorkflowConfig.model_validate(data)
    result = PolicyDecision.for_workflow(
        {
            **identity(workflow),
            "action": action,
            "target_queue": queue,
            "question_id": question,
            "reason_code": reason,
            "policy_version": "1",
        },
        workflow,
    )
    assert result.execution_mode == "shadow"


@pytest.mark.parametrize(
    "change",
    [
        {"target_queue": None},
        {"target_queue": "unknown"},
        {"question_id": "charge_or_refund"},
        {"action": "review"},
        {"reason_code": "low_confidence"},
        {"policy_version": "old"},
        {"execution_mode": "live"},
        {
            "action": "clarify",
            "target_queue": None,
            "question_id": "charge_or_refund",
            "reason_code": "missing_information",
        },
    ],
)
def test_invalid_policy_actions(workflow, change):
    with pytest.raises(ValidationError):
        PolicyDecision.for_workflow(
            {
                **identity(workflow),
                "action": "route",
                "target_queue": "account",
                "reason_code": "explicit_rule",
                "policy_version": "1",
                **change,
            },
            workflow,
        )


def test_labels_do_not_infer_intake_sufficiency(workflow):
    unreviewed = ReviewedLabel.for_workflow(
        {
            **identity(workflow),
            "eventual_owner_queue": "account",
        },
        workflow,
    )
    assert unreviewed.sufficient_information_at_intake is None
    assert unreviewed.acceptable_queues_at_intake is None
    reviewed = ReviewedLabel.for_workflow(
        {
            **identity(workflow),
            "review_status": "reviewed",
            "reviewer_id": "human-1",
            "useful_question_ids": [],
            "sufficient_information_at_intake": False,
        },
        workflow,
    )
    assert reviewed.useful_question_ids == []
    assert reviewed.sufficient_information_at_intake is False
    assert reviewed.acceptable_queues_at_intake is None


@pytest.mark.parametrize(
    "change",
    [
        {"eventual_owner_queue": "unknown"},
        {"useful_question_ids": []},
        {"review_status": "reviewed"},
        {"reviewer_id": "human"},
        {"review_status": "reviewed", "reviewer_id": "human", "useful_question_ids": ["unknown"]},
        {
            "review_status": "reviewed",
            "reviewer_id": "human",
            "acceptable_queues_at_intake": ["account", "account"],
        },
    ],
)
def test_invalid_review_labels(workflow, change):
    with pytest.raises(ValidationError):
        ReviewedLabel.for_workflow({**identity(workflow), **change}, workflow)
