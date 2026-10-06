"""Explicit source-label mappings, independent of classifier predictions."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from rightdesk.config import _load
from rightdesk.schemas import Contract, Identifier, NonEmpty, WorkflowConfig


class FlowMapping(Contract):
    flow: Identifier
    subflows: list[Identifier] = Field(min_length=1)
    target_queue: Identifier
    overrides: dict[Identifier, Identifier] = Field(default_factory=dict)
    rationale: NonEmpty

    @model_validator(mode="after")
    def check_subflows(self) -> Self:
        if len(set(self.subflows)) != len(self.subflows):
            raise ValueError("subflows must be unique within each flow")
        if not set(self.overrides) <= set(self.subflows):
            raise ValueError("override keys must be listed subflows")
        return self


class QueueMapping(Contract):
    mapping_id: Identifier
    version: NonEmpty
    workflow_id: Identifier
    workflow_version: NonEmpty
    status: Literal["provisional", "reviewed"]
    basis: NonEmpty
    flows: list[FlowMapping] = Field(min_length=1)

    @model_validator(mode="after")
    def check_flows(self) -> Self:
        if len({entry.flow for entry in self.flows}) != len(self.flows):
            raise ValueError("flow mappings must be unique")
        return self

    def validate_workflow(self, workflow: WorkflowConfig) -> None:
        if (self.workflow_id, self.workflow_version) != (workflow.workflow_id, workflow.version):
            raise ValueError("mapping workflow ID/version mismatch")
        targets = {entry.target_queue for entry in self.flows}
        targets.update(queue for entry in self.flows for queue in entry.overrides.values())
        if not targets <= workflow.queue_ids:
            raise ValueError(
                f"mapping has unknown target queues: {sorted(targets - workflow.queue_ids)}"
            )

    def lookup(self) -> dict[tuple[str, str], str]:
        return {
            (entry.flow, subflow): entry.overrides.get(subflow, entry.target_queue)
            for entry in self.flows
            for subflow in entry.subflows
        }


def load_mapping(path: Path, workflow: WorkflowConfig) -> QueueMapping:
    mapping = _load(path, QueueMapping)
    mapping.validate_workflow(workflow)
    return mapping
