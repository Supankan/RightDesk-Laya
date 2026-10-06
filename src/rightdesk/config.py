"""Load workflow and experiment YAML without silently accepting duplicate keys."""

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, ValidationError, model_validator
from yaml.constructor import ConstructorError

from rightdesk.schemas import Contract, Identifier, NonEmpty, WorkflowConfig


class ConfigurationError(ValueError):
    """An unreadable or invalid configuration, with its source path."""


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in result:
                    raise ConstructorError(None, None, f"duplicate key: {key}", key_node.start_mark)
                result[key] = self.construct_object(value_node, deep=deep)
            except TypeError as exc:
                raise ConstructorError(
                    None, None, "invalid mapping key", key_node.start_mark
                ) from exc
        return result


class DatasetConfig(Contract):
    source: Literal["abcd"]
    manifest: NonEmpty | None = None
    processed_path: NonEmpty | None = None
    queue_mapping_version: NonEmpty | None = None


class ModelConfig(Contract):
    id: Literal["majority", "rules", "tfidf_logistic", "laya"]
    checkpoint: NonEmpty | None = None
    revision: NonEmpty | None = None
    rules_path: NonEmpty | None = None


class ExperimentConfig(Contract):
    experiment_id: Identifier
    seed: Annotated[int, Field(ge=0)]
    workflow: NonEmpty
    dataset: DatasetConfig
    split_roles: list[Literal["train", "development", "calibration", "test"]]
    models: Annotated[list[ModelConfig], Field(min_length=1)]
    metrics: Annotated[
        list[
            Literal[
                "accuracy",
                "routing_error",
                "automation_coverage",
                "macro_f1",
                "brier_score",
                "calibration_error",
                "latency_p50_ms",
                "latency_p95_ms",
            ]
        ],
        Field(min_length=1),
    ]
    output_directory: NonEmpty

    @model_validator(mode="after")
    def check_experiment(self) -> Self:
        if len(self.split_roles) != 4 or set(self.split_roles) != {
            "train",
            "development",
            "calibration",
            "test",
        }:
            raise ValueError("split_roles must contain each of the four roles exactly once")
        if len({model.id for model in self.models}) != len(self.models):
            raise ValueError("model IDs must be unique")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be unique")
        return self


def _load[T: Contract](path: str | Path, schema: type[T]) -> T:
    path = Path(path)
    try:
        with path.open(encoding="utf-8-sig") as stream:
            data = yaml.load(stream, Loader=UniqueKeyLoader)
        return schema.model_validate(data)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigurationError(f"Invalid configuration {path}: {exc}") from exc


def load_workflow(path: str | Path) -> WorkflowConfig:
    return _load(path, WorkflowConfig)


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load a planning config; nullable dataset/model fields are not execution readiness."""
    return _load(path, ExperimentConfig)


def resolve_config_path(config_path: str | Path, reference: str) -> Path:
    """Resolve file references relative to the containing YAML, never the shell CWD."""
    return (Path(config_path).resolve().parent / reference).resolve()
