"""Deterministic phrase matching; unmatched and conflicting queues abstain."""

import re
from pathlib import Path

from pydantic import Field

from rightdesk.config import _load
from rightdesk.data.importers.abcd import checksum
from rightdesk.models.base import RoutingModel
from rightdesk.schemas import (
    Contract,
    Identifier,
    ModelPrediction,
    NonEmpty,
    TicketInput,
    WorkflowConfig,
)


class RulesConfig(Contract):
    version: NonEmpty
    workflow_id: Identifier
    workflow_version: NonEmpty
    phrases: dict[Identifier, list[NonEmpty]] = Field(min_length=1)


class RulesModel(RoutingModel):
    def __init__(self, workflow: WorkflowConfig, config_path: Path):
        super().__init__(workflow)
        config = _load(config_path, RulesConfig)
        if (config.workflow_id, config.workflow_version) != (
            workflow.workflow_id,
            workflow.version,
        ):
            raise ValueError("Rules workflow ID/version mismatch")
        if set(config.phrases) != workflow.queue_ids or any(
            not phrases for phrases in config.phrases.values()
        ):
            raise ValueError(
                "Rules must define nonempty phrase lists for exactly the workflow queues"
            )
        self.patterns = {
            queue: [
                re.compile(
                    r"(?<!\w)"
                    + r"\s+".join(re.escape(part) for part in phrase.casefold().split())
                    + r"(?!\w)"
                )
                for phrase in phrases
            ]
            for queue, phrases in config.phrases.items()
        }
        self.revision = f"rules-{config.version}:{checksum(config_path)}"
        self.metadata = {"rules": config.model_dump(), "probability_semantics": "unavailable"}

    def fit(self, tickets: list[TicketInput], targets: list[str]) -> None:
        self.check_training(tickets, targets)
        self.fitted = True  # Fixed rules; no parameters are estimated from these labels.

    def predict_one(self, ticket: TicketInput) -> ModelPrediction:
        matches = [
            queue
            for queue, patterns in self.patterns.items()
            if any(pattern.search(ticket.text.casefold()) for pattern in patterns)
        ]
        if len(matches) == 1:
            return self.result(ticket, predicted_queue=matches[0])
        return self.result(
            ticket, abstention_reason="conflicting_rules" if matches else "no_rule_match"
        )
