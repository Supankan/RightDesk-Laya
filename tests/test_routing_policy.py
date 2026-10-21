"""Application policy enforces artifact identity and reviews unsupported or failed inputs."""

import pytest
from test_calibration import prediction
from test_calibration import workflow as source_workflow

from rightdesk.policy.routing import RoutingPolicy, RoutingSettings
from rightdesk.schemas import ModelPrediction, TicketInput, WorkflowConfig

workflow = source_workflow


def ticket(language="en"):
    return TicketInput(
        case_id="one",
        conversation_id="one",
        text="account access",
        language=language,
        channel="chat",
    )


def policy(workflow, enabled=True, threshold=0.9, native=False):
    data = workflow.model_dump()
    data["policy"].update(version="policy-v1", routing_threshold=threshold)
    bound = WorkflowConfig.model_validate(data)
    return RoutingPolicy(
        bound, RoutingSettings(model_revision="test-v1", enabled=enabled, native_rules=native)
    )


def test_threshold_boundary_and_review_below_it(workflow):
    configured = policy(workflow)
    assert configured.decide(ticket(), prediction(workflow, "one", 0.9)).action == "route"
    result = configured.decide(ticket(), prediction(workflow, "one", 0.899))
    assert result.action == "review"
    assert result.target_queue is None
    assert result.execution_mode == "shadow"
    assert result.policy_version == "policy-v1"


def test_disabled_missing_and_unsupported_inputs_review(workflow):
    assert (
        policy(workflow, enabled=False).decide(ticket(), prediction(workflow, "one")).action
        == "review"
    )
    assert policy(workflow).decide(ticket(), None).reason_code == "model_error"
    assert policy(workflow).decide(ticket("fr"), None).reason_code == "unsupported_input"


def test_rejects_case_revision_and_calibration_mismatches(workflow):
    configured = policy(workflow)
    for row in [prediction(workflow, "other"), prediction(workflow, "one", revision="other")]:
        with pytest.raises(ValueError, match="mismatch"):
            configured.decide(ticket(), row)
    data = prediction(workflow, "one").model_dump()
    data["calibration_version"] = "wrong"
    with pytest.raises(ValueError, match="mismatch"):
        configured.decide(ticket(), ModelPrediction.for_workflow(data, workflow))


def test_native_rules_only_route_explicit_matches(workflow):
    data = prediction(workflow, "one").model_dump()
    data.update(queue_probabilities=None, predicted_queue="account")
    configured = policy(workflow, native=True, threshold=None)
    assert (
        configured.decide(ticket(), ModelPrediction.for_workflow(data, workflow)).reason_code
        == "explicit_rule"
    )
    data.update(predicted_queue=None, abstention_reason="no_rule_match")
    assert (
        configured.decide(ticket(), ModelPrediction.for_workflow(data, workflow)).action == "review"
    )
    with pytest.raises(ValueError, match="rule prediction"):
        configured.decide(ticket(), prediction(workflow, "one"))


def test_requires_threshold_for_enabled_probability_policy(workflow):
    with pytest.raises(ValueError, match="requires a threshold"):
        policy(workflow, threshold=None)
