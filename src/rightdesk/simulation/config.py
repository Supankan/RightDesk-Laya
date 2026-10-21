"""Strict simulation configuration and deterministic incumbent rules."""

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from rightdesk.config import _load, load_workflow, resolve_config_path
from rightdesk.policy.clarification import ClarificationConfig, ClarificationSelector
from rightdesk.schemas import Contract, Identifier, NonEmpty, TicketInput, WorkflowConfig


class SimulationConfig(Contract):
    version: NonEmpty
    workflow: NonEmpty
    incumbent: NonEmpty
    clarification: NonEmpty
    seed: Annotated[int, Field(ge=0)] = 42
    start_at: NonEmpty
    arrival_gap_seconds: Annotated[int, Field(ge=0, le=3600)] = 10
    customer_timeout_seconds: Annotated[int, Field(ge=1, le=86400)] = 120
    service_seconds: Annotated[int, Field(ge=1, le=86400)] = 45

    @field_validator("start_at")
    @classmethod
    def aware_start(cls, value):
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("Simulation start_at must specify UTC")
        return value


class IncumbentConfig(Contract):
    version: NonEmpty
    workflow_id: Identifier
    workflow_version: NonEmpty
    phrases: dict[Identifier, list[NonEmpty]]
    answer_phrases: dict[Identifier, dict[Identifier, list[NonEmpty]]]


class IncumbentDecision(Contract):
    action: Literal["route", "ask", "review"]
    queue: Identifier | None = None
    question_id: Identifier | None = None
    basis: NonEmpty

    @model_validator(mode="after")
    def valid_action(self):
        expected = {"route": (True, False), "ask": (False, True), "review": (False, False)}
        if (self.queue is not None, self.question_id is not None) != expected[self.action]:
            raise ValueError("Incumbent action/queue/question combination is invalid")
        return self


def load_simulation(path: Path):
    config = _load(path, SimulationConfig)
    workflow = load_workflow(resolve_config_path(path, config.workflow))
    incumbent = _load(resolve_config_path(path, config.incumbent), IncumbentConfig)
    clarification = _load(resolve_config_path(path, config.clarification), ClarificationConfig)
    PrimaryRouter(workflow, incumbent, clarification)  # Validate linked identities and rules.
    return config, workflow, incumbent, clarification


def _patterns(phrases):
    return {
        queue: [
            re.compile(
                r"(?<!\w)"
                + r"\s+".join(re.escape(word) for word in phrase.casefold().split())
                + r"(?!\w)"
            )
            for phrase in values
        ]
        for queue, values in phrases.items()
    }


class PrimaryRouter:
    """Fixed incumbent; sees only arrived customer text and its own question history."""

    def __init__(
        self, workflow: WorkflowConfig, config: IncumbentConfig, clarification: ClarificationConfig
    ):
        self.workflow, self.config = workflow, config
        self.selector = ClarificationSelector(workflow, clarification)
        if (config.workflow_id, config.workflow_version) != (
            workflow.workflow_id,
            workflow.version,
        ):
            raise ValueError("Incumbent workflow mismatch")
        if set(config.phrases) != workflow.queue_ids or any(
            not phrases for phrases in config.phrases.values()
        ):
            raise ValueError("Incumbent must define phrases for every queue")
        pairs = {rule.question_id: set(rule.candidate_queues) for rule in clarification.rules}
        if set(config.answer_phrases) != set(pairs):
            raise ValueError("Incumbent must define answers for every bounded question")
        for question, phrases in config.answer_phrases.items():
            if set(phrases) != pairs[question] or any(not values for values in phrases.values()):
                raise ValueError("Answer rules must cover exactly their question queue pair")
        self.patterns = _patterns(config.phrases)
        self.answers = {
            question: _patterns(phrases) for question, phrases in config.answer_phrases.items()
        }

    @staticmethod
    def matches(patterns, text):
        return sorted(
            queue
            for queue, rules in patterns.items()
            if any(rule.search(text.casefold()) for rule in rules)
        )

    def decide(self, ticket: TicketInput, asked: list[str], reply: str | None = None):
        self.selector._eligibility(ticket, asked)
        if ticket.language not in (None, "en"):
            return IncumbentDecision(action="review", basis="unsupported_input")
        if reply is not None:
            if not asked:
                raise ValueError("A clarification reply requires a prior question")
            matches = self.matches(self.answers[asked[-1]], reply)
            if len(matches) == 1:
                return IncumbentDecision(
                    action="route", queue=matches[0], basis="explicit_answer_rule"
                )
            return IncumbentDecision(action="review", basis="unresolved_clarification")
        matches = self.matches(self.patterns, ticket.text)
        if len(matches) == 1:
            return IncumbentDecision(
                action="route", queue=matches[0], basis="explicit_incumbent_rule"
            )
        choice = self.selector.from_ambiguity(ticket, matches, asked)
        if choice.question_id:
            return IncumbentDecision(
                action="ask", question_id=choice.question_id, basis=choice.basis
            )
        return IncumbentDecision(
            action="review", basis=choice.basis if matches else "no_incumbent_rule"
        )
