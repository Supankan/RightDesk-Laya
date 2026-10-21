"""Questions require a supported ambiguity, one-question budget, and honest labels."""

import json
from pathlib import Path

import pytest

from rightdesk.config import _load, load_workflow
from rightdesk.evaluation.clarification import evaluate_clarification, score_usefulness
from rightdesk.policy.clarification import ClarificationConfig, ClarificationSelector, IntakePolicy
from rightdesk.policy.routing import RoutingSettings
from rightdesk.schemas import ModelPrediction, TicketInput, WorkflowConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workflow():
    original = load_workflow(ROOT / "configs/workflows/retail_support.yaml").model_dump()
    original["policy"].update(version="questions-test", clarification_enabled=True)
    return WorkflowConfig.model_validate(original)


def ticket(case="one", language="en"):
    return TicketInput(
        case_id=case,
        conversation_id=case,
        text="Please explain the payment issue.",
        language=language,
        channel="chat",
    )


def prediction(workflow, probabilities=None):
    values = dict.fromkeys(workflow.queue_ids, 0.035)
    values.update(billing_subscriptions=0.44, returns_refunds=0.42)
    return ModelPrediction.for_workflow(
        {
            "case_id": "one",
            "workflow_id": workflow.workflow_id,
            "workflow_version": workflow.version,
            "model_revision": "test",
            "queue_probabilities": values if probabilities is None else probabilities,
        },
        workflow,
    )


def selector(workflow):
    return ClarificationSelector.from_path(workflow, ROOT / "configs/clarification/retail_v1.yaml")


def test_question_selection_is_bounded_and_does_not_invent_sufficiency(workflow):
    model = prediction(workflow)
    selected = selector(workflow).select(ticket(), model, [])
    assert selected.question_id == "charge_or_refund"
    assert model.sufficient_information_probability is None
    assert selector(workflow).select(ticket(), model, ["charge_or_refund"]).question_id is None
    assert selector(workflow).select(ticket(language="fr"), model, []).basis == "unsupported_input"
    assert selector(workflow).select(ticket(), None, []).basis == "model_unavailable"


def test_diffuse_confident_ties_and_unconfigured_pairs_do_not_ask(workflow):
    selected = selector(workflow)
    diffuse = prediction(workflow, dict.fromkeys(workflow.queue_ids, 1 / 6))
    assert selected.select(ticket(), diffuse, []).question_id is None
    confident = prediction(
        workflow, {q: 0.9 if q == "billing_subscriptions" else 0.02 for q in workflow.queue_ids}
    )
    assert selected.select(ticket(), confident, []).basis == "confident_queue_choice"
    values = {q: 0.01 for q in workflow.queue_ids}
    values.update(account=0.5, technical_support=0.46)
    assert selected.select(ticket(), prediction(workflow, values), []).question_id is None
    triple = {q: 0.01 for q in workflow.queue_ids}
    triple.update(billing_subscriptions=0.37, returns_refunds=0.3, account=0.3)
    assert selected.select(ticket(), prediction(workflow, triple), []).basis == "multi_queue_tie"


def test_gate_is_preserved_while_clarification_is_a_separate_proposal(workflow):
    policy = IntakePolicy(
        workflow,
        RoutingSettings(model_revision="test", enabled=False),
        _load(ROOT / "configs/clarification/retail_v1.yaml", ClarificationConfig),
    )
    assert policy.decide(ticket(), prediction(workflow), []).action == "clarify"
    assert policy.decide(ticket(), prediction(workflow), ["charge_or_refund"]).action == "review"
    assert policy.decide(ticket(), None, []).reason_code == "model_error"
    assert policy.decide(ticket(), prediction(workflow), []).execution_mode == "shadow"
    with pytest.raises(ValueError, match="history"):
        policy.decide(ticket(), prediction(workflow), ["unknown"])


def test_rule_configuration_and_case_identity_are_checked(workflow):
    data = _load(ROOT / "configs/clarification/retail_v1.yaml", ClarificationConfig).model_dump()
    data["rules"][0]["candidate_queues"] = ["account", "account"]
    with pytest.raises(ValueError, match="distinct"):
        ClarificationSelector(workflow, ClarificationConfig.model_validate(data))
    with pytest.raises(ValueError, match="mismatch"):
        selector(workflow).select(ticket("different"), prediction(workflow), [])


def test_unknown_usefulness_is_not_counted_as_negative_and_reviewers_stay_individual():
    proposals = {"one": "charge_or_refund", "two": None, "three": "charge_or_refund"}
    labels = [
        {
            "case_id": "one",
            "useful_question_ids": ["charge_or_refund"],
            "sufficient_information_at_intake": False,
        },
        {"case_id": "one", "useful_question_ids": None, "sufficient_information_at_intake": None},
        {
            "case_id": "two",
            "useful_question_ids": ["charge_or_refund"],
            "sufficient_information_at_intake": False,
        },
        {"case_id": "three", "useful_question_ids": [], "sufficient_information_at_intake": True},
    ]
    score = score_usefulness(proposals, labels)
    assert score["individual_judgments"] == 4
    assert score["known_usefulness_judgments"] == 3
    assert score["useful_question_precision"] == 0.5
    assert score["useful_question_recall"] == 0.5
    assert score["ask_rate_when_already_sufficient"] == 1
    assert score_usefulness(proposals, [])["useful_question_precision"] is None


def test_offline_report_with_no_reviews_has_null_usefulness(tmp_path):
    pytest.importorskip("numpy")
    from test_workbench import workbench as fixture_function

    # Reuse the synthetic catalog factory without invoking any model SDK.
    bundle = fixture_function.__wrapped__(tmp_path)
    report = evaluate_clarification(
        bundle["store"],
        bundle["run_id"],
        "laya",
        ROOT / "configs/clarification/retail_v1.yaml",
        tmp_path / "questions",
    )
    run = json.loads((report.parent / "run.json").read_text())
    assert run["human_usefulness"]["known_usefulness_judgments"] == 0
    assert run["human_usefulness"]["useful_question_precision"] is None
    assert not run["routing_gate_enabled"] and not run["test_evaluated"]
    assert (report.parent / "review_snapshot.jsonl").read_text() == ""
