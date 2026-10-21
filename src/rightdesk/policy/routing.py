"""Deterministic route/review proposals; model confidence never executes an action."""

from rightdesk.schemas import (
    Contract,
    ModelPrediction,
    NonEmpty,
    PolicyDecision,
    TicketInput,
    WorkflowConfig,
)


class RoutingSettings(Contract):
    model_revision: NonEmpty
    calibration_version: NonEmpty | None = None
    enabled: bool = False
    native_rules: bool = False


class RoutingPolicy:
    def __init__(self, workflow: WorkflowConfig, settings: RoutingSettings):
        self.workflow = WorkflowConfig.model_validate(workflow)
        self.settings = RoutingSettings.model_validate(settings)
        if workflow.policy.clarification_enabled:
            raise ValueError("Phase 7 routing policy does not implement clarification")
        if workflow.policy.sufficient_information_threshold is not None:
            raise ValueError("Sufficiency threshold requires a separately validated model")
        if workflow.policy.calibration_version != settings.calibration_version:
            raise ValueError("Policy/calibration version mismatch")
        if (
            settings.enabled
            and not settings.native_rules
            and workflow.policy.routing_threshold is None
        ):
            raise ValueError("Enabled probability policy requires a threshold")
        if settings.native_rules and (
            settings.calibration_version is not None
            or workflow.policy.routing_threshold is not None
        ):
            raise ValueError("Native rules cannot use probability calibration/thresholds")

    def decide(self, ticket: TicketInput, prediction: ModelPrediction | None) -> PolicyDecision:
        ticket = TicketInput.model_validate(ticket)
        action, reason, target = "review", "low_confidence", None
        if ticket.language not in (None, "en"):
            reason = "unsupported_input"
        elif prediction is None:
            reason = "model_error"
        else:
            prediction = ModelPrediction.for_workflow(prediction, self.workflow)
            if prediction.case_id != ticket.case_id:
                raise ValueError("Ticket/prediction case mismatch")
            if (prediction.model_revision, prediction.calibration_version) != (
                self.settings.model_revision,
                self.settings.calibration_version,
            ):
                raise ValueError("Policy model revision/calibration mismatch")
            if self.settings.enabled:
                if self.settings.native_rules:
                    if prediction.queue_probabilities is not None:
                        raise ValueError("Native rule policy requires a rule prediction")
                    if prediction.selected_queue() is not None:
                        action, reason, target = (
                            "route",
                            "explicit_rule",
                            prediction.selected_queue(),
                        )
                elif prediction.queue_probabilities is None:
                    raise ValueError("Probability policy requires probabilities")
                elif (
                    max(prediction.queue_probabilities.values())
                    >= self.workflow.policy.routing_threshold
                ):
                    action, reason, target = (
                        "route",
                        "meets_routing_threshold",
                        prediction.selected_queue(),
                    )
        return PolicyDecision.for_workflow(
            {
                "case_id": ticket.case_id,
                "workflow_id": self.workflow.workflow_id,
                "workflow_version": self.workflow.version,
                "policy_version": self.workflow.policy.version,
                "action": action,
                "reason_code": reason,
                "target_queue": target,
            },
            self.workflow,
        )
