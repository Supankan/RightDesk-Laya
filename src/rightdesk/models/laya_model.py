"""Pinned local Laya inference; no routing, training, or confidence-policy side effects."""

import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from rightdesk.config import _load, resolve_config_path
from rightdesk.data.importers.abcd import checksum, write_json
from rightdesk.models.base import RoutingModel
from rightdesk.schemas import Contract, ModelPrediction, NonEmpty, TicketInput, WorkflowConfig


class LayaError(RuntimeError):
    """Loading or inference failed; callers must not route using fabricated output."""


class LayaConfig(Contract):
    checkpoint: Literal["convaiinnovations/laya"]
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    local_path: NonEmpty
    device: Literal["cpu"] = "cpu"
    batch_size: Annotated[int, Field(ge=1, le=32)] = 4
    cpu_threads: Annotated[int, Field(ge=1, le=32)] = 4
    max_len: Literal[512] = 512
    head_max_len: Literal[192] = 192


def load_laya_config(path: Path) -> LayaConfig:
    return _load(path, LayaConfig)


def verify_checkpoint(path: Path, config: LayaConfig) -> dict:
    try:
        manifest = json.loads((path / "rightdesk-model.json").read_text(encoding="utf-8"))
        if (manifest["checkpoint"], manifest["revision"]) != (config.checkpoint, config.revision):
            raise ValueError("Checkpoint identity/revision mismatch")
        required = {
            "rl_agent_config.json",
            "model.safetensors",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        }
        if set(manifest["sha256"]) != required:
            raise ValueError("Incomplete checkpoint manifest")
        for name, digest in manifest["sha256"].items():
            if checksum(path / name) != digest:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LayaError(f"Invalid local checkpoint at {path}: {exc}") from exc


def download_laya(config_path: Path) -> Path:
    from huggingface_hub import snapshot_download

    config = load_laya_config(config_path)
    path = resolve_config_path(config_path, config.local_path)
    if (path / "rightdesk-model.json").exists():
        verify_checkpoint(path, config)
        return path
    names = [
        "rl_agent_config.json",
        "model.safetensors",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ]
    snapshot_download(
        config.checkpoint,
        revision=config.revision,
        local_dir=path,
        allow_patterns=[*names, "README.md"],
        token=False,
    )
    write_json(
        path / "rightdesk-model.json",
        {
            "checkpoint": config.checkpoint,
            "revision": config.revision,
            "source": f"https://huggingface.co/{config.checkpoint}/tree/{config.revision}",
            "license": "Apache-2.0 (model card)",
            "sha256": {name: checksum(path / name) for name in names},
        },
    )
    verify_checkpoint(path, config)
    return path


class LayaModel(RoutingModel):
    def __init__(self, workflow: WorkflowConfig, config_path: Path):
        super().__init__(workflow)
        self.config = load_laya_config(config_path)
        self.local_path = resolve_config_path(config_path, self.config.local_path)
        self.agent = None
        self.last_raw: list[dict] = []
        self.questions = {
            "owner_queue": {
                "type": "choice",
                "instructions": "Which support queue should handle this customer's request?",
                "criteria": {queue.id: queue.description for queue in workflow.queues},
            }
        }
        self.revision = f"{self.config.checkpoint}@{self.config.revision}"

    def fit(self, tickets: list[TicketInput], targets: list[str]) -> None:
        # Shared interface compatibility only. No task labels train this checkpoint.
        self.check_training(tickets, targets)
        self.load()

    def load(self) -> None:
        if self.agent is not None:
            return
        manifest = verify_checkpoint(self.local_path, self.config)
        try:
            import torch
            from laya import Agent

            torch.set_num_threads(self.config.cpu_threads)
            self.agent = Agent(
                str(self.local_path),
                device=self.config.device,
                expected_sha256=manifest["sha256"],
                fast=False,
                compile=False,
            )
        except Exception as exc:
            raise LayaError(f"Laya load failed: {type(exc).__name__}: {exc}") from exc
        self.fitted = True
        self.metadata = {
            "config": self.config.model_dump(),
            "questions": self.questions,
            "checkpoint_manifest": manifest,
            "device": str(self.agent.device),
            "training": "zero-shot; no RightDesk training examples or labels used",
            "probability_semantics": "SDK checkpoint temperatures; not calibrated on RightDesk",
            "rounding": "four-decimal SDK probabilities renormalized within k*0.00005 + 1e-6",
            "context": "Reject ticket text over 312 tokens; reserve head_max_len + 8 tokens",
        }

    def _decode(self, ticket: TicketInput, raw: dict) -> ModelPrediction:
        answer = raw["answers"]["owner_queue"]
        values = answer["probabilities"]
        if not isinstance(values, dict) or set(values) != self.workflow.queue_ids:
            raise ValueError("Laya probabilities must cover exactly the configured queues")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in values.values()
        ):
            raise ValueError("Laya returned invalid probabilities")
        total = math.fsum(values.values())
        if total <= 0 or abs(total - 1) > len(values) * 0.00005 + 1e-6:
            raise ValueError("Probability sum exceeds SDK rounding tolerance")
        choice = answer["choice"]
        if choice not in values or values[choice] != max(values.values()):
            raise ValueError("Laya choice is inconsistent with its probabilities")
        return self.result(
            ticket,
            predicted_queue=choice,
            queue_probabilities={key: value / total for key, value in values.items()},
        )

    def predict_batch(self, tickets: list[TicketInput]) -> list[ModelPrediction]:
        if not self.fitted or self.agent is None:
            raise LayaError("Load Laya before prediction")
        if not tickets:
            return []
        self.last_raw = []
        try:
            for ticket in tickets:
                if ticket.language not in (None, "en"):
                    raise ValueError(
                        "English checkpoint only supports English inputs in this adapter"
                    )
                length = len(self.agent.tok.encode(ticket.text, add_special_tokens=False))
                if length > self.config.max_len - self.config.head_max_len - 8:
                    raise ValueError(f"Input exceeds conservative token budget: {ticket.case_id}")
            raw = self.agent.predict_batch(
                [ticket.text for ticket in tickets],
                self.questions,
                batch_size=self.config.batch_size,
                max_len=self.config.max_len,
                head_max_len=self.config.head_max_len,
                lang="en",
            )
            if not isinstance(raw, list) or len(raw) != len(tickets):
                raise ValueError("Laya batch output count does not match input count")
            self.last_raw = raw
            return [
                self._decode(ticket, result) for ticket, result in zip(tickets, raw, strict=True)
            ]
        except Exception as exc:
            raise LayaError(f"Laya inference failed: {type(exc).__name__}: {exc}") from exc

    def predict_one(self, ticket: TicketInput) -> ModelPrediction:
        return self.predict_batch([ticket])[0]
