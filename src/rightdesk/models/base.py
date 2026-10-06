"""Shared baseline interface: fitting sees targets; prediction receives tickets only."""

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

from rightdesk.schemas import ModelPrediction, TicketInput, WorkflowConfig


def training_fingerprint(tickets: list[TicketInput], targets: list[str]) -> str:
    rows = sorted(
        (ticket.case_id, ticket.text, target)
        for ticket, target in zip(tickets, targets, strict=True)
    )
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


class RoutingModel(ABC):
    def __init__(self, workflow: WorkflowConfig):
        self.workflow = WorkflowConfig.model_validate(workflow)
        self.queues = sorted(workflow.queue_ids)
        self.fitted = False
        self.metadata: dict = {}
        self.revision = "unfitted"

    def check_training(self, tickets: list[TicketInput], targets: list[str]) -> None:
        if not tickets or len(tickets) != len(targets):
            raise ValueError("Training tickets and targets must have equal nonzero length")
        if len({ticket.case_id for ticket in tickets}) != len(tickets):
            raise ValueError("Duplicate training case IDs")
        if not set(targets) <= self.workflow.queue_ids:
            raise ValueError("Training targets contain unknown queues")

    def result(self, ticket: TicketInput, **values) -> ModelPrediction:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return ModelPrediction.for_workflow(
            {
                "case_id": ticket.case_id,
                "workflow_id": self.workflow.workflow_id,
                "workflow_version": self.workflow.version,
                "model_revision": self.revision,
                **values,
            },
            self.workflow,
        )

    @abstractmethod
    def fit(self, tickets: list[TicketInput], targets: list[str]) -> None: ...

    @abstractmethod
    def predict_one(self, ticket: TicketInput) -> ModelPrediction: ...

    def save(self, directory: Path) -> None:
        (directory / "model.json").write_text(
            json.dumps({"revision": self.revision, **self.metadata}, indent=2) + "\n",
            encoding="utf-8",
        )
