"""Validated intake contracts; workflow-bound records require validation context."""

from math import fsum, isclose
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    model_validator,
)

NonEmpty = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Contract(BaseModel):
    """Reject unknown fields and implicit coercion at application boundaries."""

    model_config = ConfigDict(strict=True, extra="forbid", revalidate_instances="always")


class TicketInput(Contract):
    case_id: NonEmpty
    conversation_id: NonEmpty
    text: NonEmpty
    language: NonEmpty | None = None
    channel: Literal["chat", "email", "form"]


class QueueDefinition(Contract):
    id: Identifier
    description: NonEmpty


class QuestionDefinition(Contract):
    id: Identifier
    text: NonEmpty
    dimension: Identifier


class PolicyConfig(Contract):
    version: NonEmpty
    max_questions: Annotated[int, Field(ge=0, le=1)] = 1
    routing_threshold: Probability | None = None
    sufficient_information_threshold: Probability | None = None
    calibration_version: NonEmpty | None = None
    clarification_enabled: bool = False

    @model_validator(mode="after")
    def check_settings(self) -> Self:
        if self.clarification_enabled and self.max_questions == 0:
            raise ValueError("clarification requires a nonzero question budget")
        return self


class WorkflowConfig(Contract):
    workflow_id: Identifier
    version: NonEmpty
    execution_mode: Literal["shadow"] = "shadow"
    queues: Annotated[list[QueueDefinition], Field(min_length=2, max_length=20)]
    questions: list[QuestionDefinition] = Field(default_factory=list)
    policy: PolicyConfig

    @property
    def queue_ids(self) -> set[str]:
        return {queue.id for queue in self.queues}

    @property
    def question_ids(self) -> set[str]:
        return {question.id for question in self.questions}

    @model_validator(mode="after")
    def check_definitions(self) -> Self:
        if len(self.queue_ids) != len(self.queues):
            raise ValueError("queue IDs must be unique")
        if len(self.question_ids) != len(self.questions):
            raise ValueError("question IDs must be unique")
        if self.policy.clarification_enabled and not self.questions:
            raise ValueError("clarification requires configured questions")
        return self


class WorkflowRecord(Contract):
    case_id: NonEmpty
    workflow_id: Identifier
    workflow_version: NonEmpty

    @classmethod
    def for_workflow(cls, data: Any, workflow: WorkflowConfig) -> Self:
        """Validate a record against a freshly checked workflow configuration."""
        workflow = WorkflowConfig.model_validate(workflow)
        return cls.model_validate(data, context={"workflow": workflow})

    @model_validator(mode="after")
    def check_workflow(self, info: ValidationInfo) -> Self:
        workflow = self.workflow_context(info)
        if (self.workflow_id, self.workflow_version) != (workflow.workflow_id, workflow.version):
            raise ValueError("record workflow ID/version does not match the supplied workflow")
        return self

    @staticmethod
    def workflow_context(info: ValidationInfo) -> WorkflowConfig:
        workflow = (info.context or {}).get("workflow")
        if not isinstance(workflow, WorkflowConfig):
            raise ValueError("workflow context is required; use for_workflow(data, workflow)")
        return workflow


class ModelPrediction(WorkflowRecord):
    queue_probabilities: dict[Identifier, Probability] | None = None
    predicted_queue: Identifier | None = None
    abstention_reason: Literal["no_rule_match", "conflicting_rules"] | None = None
    sufficient_information_probability: Probability | None = None
    suggested_question_id: Identifier | None = None
    model_revision: NonEmpty
    calibration_version: NonEmpty | None = None

    @model_validator(mode="after")
    def check_prediction(self, info: ValidationInfo) -> Self:
        workflow = self.workflow_context(info)
        if self.predicted_queue is not None and self.predicted_queue not in workflow.queue_ids:
            raise ValueError("predicted queue is not configured")
        if self.abstention_reason is not None and (
            self.predicted_queue is not None or self.queue_probabilities is not None
        ):
            raise ValueError("abstention cannot include a queue or probability distribution")
        if self.queue_probabilities is None:
            if self.predicted_queue is None and self.abstention_reason is None:
                raise ValueError(
                    "provide probabilities, a predicted queue, or an abstention reason"
                )
        elif set(self.queue_probabilities) != workflow.queue_ids:
            raise ValueError("queue probabilities must cover exactly the configured queues")
        else:
            if not isclose(fsum(self.queue_probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-6):
                raise ValueError("queue probabilities must sum to 1 within 1e-6")
            if self.predicted_queue is not None and (
                self.queue_probabilities[self.predicted_queue]
                != max(self.queue_probabilities.values())
            ):
                raise ValueError("predicted queue must maximize queue probability")
        if (
            self.suggested_question_id is not None
            and self.suggested_question_id not in workflow.question_ids
        ):
            raise ValueError("suggested question is not configured")
        return self

    def selected_queue(self) -> str | None:
        if self.predicted_queue is not None:
            return self.predicted_queue
        if self.queue_probabilities is None:
            return None
        return min(self.queue_probabilities, key=lambda q: (-self.queue_probabilities[q], q))


ReasonCode = Literal[
    "explicit_rule",
    "meets_routing_threshold",
    "missing_information",
    "low_confidence",
    "unsupported_input",
    "model_error",
]


class PolicyDecision(WorkflowRecord):
    action: Literal["route", "clarify", "review"]
    target_queue: Identifier | None = None
    question_id: Identifier | None = None
    reason_code: ReasonCode
    policy_version: NonEmpty
    execution_mode: Literal["shadow"] = "shadow"

    @model_validator(mode="after")
    def check_decision(self, info: ValidationInfo) -> Self:
        workflow = self.workflow_context(info)
        if self.policy_version != workflow.policy.version:
            raise ValueError("policy version does not match the supplied workflow")
        expected = {"route": (True, False), "clarify": (False, True), "review": (False, False)}
        if (self.target_queue is not None, self.question_id is not None) != expected[self.action]:
            raise ValueError("route needs only a queue; clarify only a question; review neither")
        if self.target_queue is not None and self.target_queue not in workflow.queue_ids:
            raise ValueError("target queue is not configured")
        if self.question_id is not None and self.question_id not in workflow.question_ids:
            raise ValueError("question is not configured")
        if self.action == "clarify" and not workflow.policy.clarification_enabled:
            raise ValueError("clarification is disabled in this workflow")
        allowed_reasons = {
            "route": {"explicit_rule", "meets_routing_threshold"},
            "clarify": {"missing_information"},
            "review": {"missing_information", "low_confidence", "unsupported_input", "model_error"},
        }
        if self.reason_code not in allowed_reasons[self.action]:
            raise ValueError("reason code is incompatible with the action")
        return self


class ReviewedLabel(WorkflowRecord):
    eventual_owner_queue: Identifier | None = None
    acceptable_queues_at_intake: list[Identifier] | None = None
    sufficient_information_at_intake: bool | None = None
    useful_question_ids: list[Identifier] | None = None
    review_status: Literal["unreviewed", "reviewed"] = "unreviewed"
    reviewer_id: NonEmpty | None = None

    @model_validator(mode="after")
    def check_labels(self, info: ValidationInfo) -> Self:
        workflow = self.workflow_context(info)
        if (
            self.eventual_owner_queue is not None
            and self.eventual_owner_queue not in workflow.queue_ids
        ):
            raise ValueError("eventual owner queue is not configured")
        for values, allowed in (
            (self.acceptable_queues_at_intake, workflow.queue_ids),
            (self.useful_question_ids, workflow.question_ids),
        ):
            if values is not None and (
                len(values) != len(set(values)) or not set(values) <= allowed
            ):
                raise ValueError("reviewed lists must contain unique configured IDs")
        judgments = (
            self.acceptable_queues_at_intake,
            self.sufficient_information_at_intake,
            self.useful_question_ids,
        )
        if self.review_status == "reviewed":
            if self.reviewer_id is None or all(value is None for value in judgments):
                raise ValueError(
                    "reviewed labels require a reviewer and at least one intake judgment"
                )
        elif self.reviewer_id is not None or any(value is not None for value in judgments):
            raise ValueError("unreviewed labels cannot contain reviewer judgments")
        return self
