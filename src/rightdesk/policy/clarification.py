"""Bounded, predefined clarification proposals from observable queue ambiguity."""

from pathlib import Path
from typing import Annotated

from pydantic import Field

from rightdesk.config import _load
from rightdesk.policy.routing import RoutingPolicy, RoutingSettings
from rightdesk.schemas import (
    Contract,
    Identifier,
    ModelPrediction,
    NonEmpty,
    PolicyDecision,
    Probability,
    TicketInput,
    WorkflowConfig,
)


class ClarificationRule(Contract):
    question_id: Identifier
    candidate_queues: Annotated[list[Identifier], Field(min_length=2, max_length=2)]


class ClarificationConfig(Contract):
    version: NonEmpty
    workflow_id: Identifier
    workflow_version: NonEmpty
    min_pair_mass: Probability = 0.65
    max_probability: Probability = 0.85
    max_margin: Probability = 0.2
    rules: list[ClarificationRule]


class QuestionChoice(Contract):
    question_id: Identifier | None = None
    basis: NonEmpty
    candidate_queues: list[Identifier] = Field(default_factory=list)


class ClarificationSelector:
    def __init__(self, workflow: WorkflowConfig, config: ClarificationConfig):
        self.workflow = WorkflowConfig.model_validate(workflow)
        self.config = ClarificationConfig.model_validate(config)
        if (config.workflow_id, config.workflow_version) != (
            workflow.workflow_id,
            workflow.version,
        ):
            raise ValueError("Clarification workflow mismatch")
        pairs, questions = set(), set()
        for rule in config.rules:
            pair = frozenset(rule.candidate_queues)
            if len(pair) != 2 or not pair <= workflow.queue_ids:
                raise ValueError("Clarification rules require two distinct configured queues")
            if (
                rule.question_id not in workflow.question_ids
                or pair in pairs
                or rule.question_id in questions
            ):
                raise ValueError("Unknown or duplicate clarification rule")
            pairs.add(pair)
            questions.add(rule.question_id)

    @classmethod
    def from_path(cls, workflow: WorkflowConfig, path: Path):
        return cls(workflow, _load(path, ClarificationConfig))

    def _eligibility(self, ticket: TicketInput, asked_question_ids: list[str]):
        TicketInput.model_validate(ticket)
        if (
            len(asked_question_ids) != len(set(asked_question_ids))
            or not set(asked_question_ids) <= self.workflow.question_ids
        ):
            raise ValueError("Invalid question history")
        if not self.workflow.policy.clarification_enabled:
            return "clarification_disabled"
        if ticket.language not in (None, "en"):
            return "unsupported_input"
        if len(asked_question_ids) >= self.workflow.policy.max_questions:
            return "question_budget_exhausted"
        return None

    def from_ambiguity(
        self, ticket: TicketInput, candidate_queues: list[str], asked_question_ids: list[str]
    ) -> QuestionChoice:
        """For an incumbent's explicit conflicting-rule matches; no invented probabilities."""
        if (
            len(candidate_queues) != len(set(candidate_queues))
            or not set(candidate_queues) <= self.workflow.queue_ids
        ):
            raise ValueError("Invalid ambiguity candidates")
        reason = self._eligibility(ticket, asked_question_ids)
        if reason:
            return QuestionChoice(basis=reason)
        pair = set(candidate_queues)
        for rule in self.config.rules:
            if pair == set(rule.candidate_queues):
                return QuestionChoice(
                    question_id=rule.question_id,
                    basis="configured_queue_pair",
                    candidate_queues=sorted(pair),
                )
        return QuestionChoice(
            basis="no_bounded_question_for_ambiguity", candidate_queues=sorted(pair)
        )

    def select(
        self, ticket: TicketInput, prediction: ModelPrediction | None, asked_question_ids: list[str]
    ) -> QuestionChoice:
        reason = self._eligibility(ticket, asked_question_ids)
        if reason:
            return QuestionChoice(basis=reason)
        if prediction is None:
            return QuestionChoice(basis="model_unavailable")
        prediction = ModelPrediction.for_workflow(prediction, self.workflow)
        if prediction.case_id != ticket.case_id:
            raise ValueError("Question selection ticket/prediction mismatch")
        probabilities = prediction.queue_probabilities
        if probabilities is None:
            return QuestionChoice(basis="probability_ambiguity_unavailable")
        ranked = sorted(probabilities, key=lambda key: (-probabilities[key], key))
        first, second = (probabilities[key] for key in ranked[:2])
        if first >= self.config.max_probability:
            return QuestionChoice(basis="confident_queue_choice")
        if first + second < self.config.min_pair_mass or first - second > self.config.max_margin:
            return QuestionChoice(basis="not_a_supported_two_queue_ambiguity")
        if sum(value >= second for value in probabilities.values()) != 2:
            return QuestionChoice(basis="multi_queue_tie")
        return self.from_ambiguity(ticket, ranked[:2], asked_question_ids)


class IntakePolicy:
    """Compose the frozen route gate with a separate experimental question budget."""

    def __init__(
        self, workflow: WorkflowConfig, routing: RoutingSettings, clarification: ClarificationConfig
    ):
        self.workflow = WorkflowConfig.model_validate(workflow)
        self.selector = ClarificationSelector(workflow, clarification)
        data = workflow.model_dump()
        data["policy"]["clarification_enabled"] = False
        self.routing = RoutingPolicy(WorkflowConfig.model_validate(data), routing)

    def decide(
        self, ticket: TicketInput, prediction: ModelPrediction | None, asked_question_ids: list[str]
    ) -> PolicyDecision:
        # Validate history even when the route gate would accept a ticket.
        self.selector._eligibility(ticket, asked_question_ids)
        decision = self.routing.decide(ticket, prediction)
        if decision.action == "route" or decision.reason_code in (
            "model_error",
            "unsupported_input",
        ):
            return PolicyDecision.for_workflow(decision.model_dump(), self.workflow)
        choice = self.selector.select(ticket, prediction, asked_question_ids)
        if choice.question_id is not None:
            data = decision.model_dump()
            data.update(
                action="clarify", question_id=choice.question_id, reason_code="missing_information"
            )
            return PolicyDecision.for_workflow(data, self.workflow)
        return PolicyDecision.for_workflow(decision.model_dump(), self.workflow)
