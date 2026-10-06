import json
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from pydantic import ValidationError

from rightdesk.config import load_workflow
from rightdesk.models.majority import MajorityModel
from rightdesk.models.rules import RulesModel
from rightdesk.models.sklearn_model import TfidfLogisticModel
from rightdesk.schemas import ModelPrediction, TicketInput

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workflow():
    return load_workflow(ROOT / "configs/workflows/retail_support.yaml")


def ticket(identifier, text):
    return TicketInput(
        case_id=str(identifier),
        conversation_id=str(identifier),
        text=text,
        language="en",
        channel="chat",
    )


def training():
    return (
        [ticket(1, "forgot password"), ticket(2, "reset username"), ticket(3, "refund item")],
        ["account", "account", "returns_refunds"],
    )


def test_majority_uses_training_priors_and_keeps_unknowns(workflow):
    model = MajorityModel(workflow)
    model.fit(*training())
    prediction = model.predict_one(ticket(4, "refund my order"))
    assert prediction.selected_queue() == "account"
    assert prediction.queue_probabilities["account"] == pytest.approx(2 / 3)
    assert set(prediction.queue_probabilities) == workflow.queue_ids
    assert prediction.sufficient_information_probability is None
    assert prediction.calibration_version is None


@pytest.mark.parametrize(
    "text,queue,reason",
    [
        ("Please reset my PASSWORD", "account", None),
        ("refund", "returns_refunds", None),
        ("Hi", None, "no_rule_match"),
        ("shipping refund", None, "conflicting_rules"),
        ("billboard", None, "no_rule_match"),
    ],
)
def test_rules_abstain_without_inventing_probabilities(workflow, text, queue, reason):
    model = RulesModel(workflow, ROOT / "configs/models/rules_v1.yaml")
    model.fit(*training())
    result = model.predict_one(ticket(5, text))
    assert result.predicted_queue == queue
    assert result.abstention_reason == reason
    assert result.queue_probabilities is None
    assert result.sufficient_information_probability is None


def test_tfidf_fits_only_training_text_and_serializes(workflow, tmp_path):
    model = TfidfLogisticModel(workflow)
    with pytest.raises(ValueError, match="fitted"):
        model.predict_one(ticket(4, "newword"))
    model.fit(*training())
    prediction = model.predict_one(ticket(4, "evaluationonlyword"))
    assert "evaluationonlyword" not in model.pipeline["tfidf"].vocabulary_
    assert set(prediction.queue_probabilities) == workflow.queue_ids
    assert sum(prediction.queue_probabilities.values()) == pytest.approx(1)
    assert prediction.queue_probabilities["technical_support"] == 0
    model.save(tmp_path)
    assert (tmp_path / "pipeline.joblib").exists()
    assert json.loads((tmp_path / "model.json").read_text())["training_cases"] == 3


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"predicted_queue": "unknown"},
        {"predicted_queue": "account", "abstention_reason": "no_rule_match"},
    ],
)
def test_prediction_requires_a_valid_prediction_or_abstention(workflow, values):
    with pytest.raises(ValidationError):
        ModelPrediction.for_workflow(
            {
                "case_id": "x",
                "workflow_id": workflow.workflow_id,
                "workflow_version": workflow.version,
                "model_revision": "fixture",
                **values,
            },
            workflow,
        )


def test_bad_training_labels_fail(workflow):
    model = TfidfLogisticModel(workflow)
    tickets, _ = training()
    with pytest.raises(ValueError, match="unknown"):
        model.fit(tickets, ["unknown"] * 3)
    with pytest.raises(ValueError, match="two observed"):
        model.fit(tickets, ["account"] * 3)
