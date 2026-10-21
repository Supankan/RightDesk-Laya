"""Group-isolated probability calibration and frozen selective-routing thresholds."""

import hashlib
import json
import math
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import beta

from rightdesk.data.preprocessing import duplicate_key
from rightdesk.schemas import Contract, ModelPrediction, NonEmpty, TicketInput


class CalibrationConfig(Contract):
    version: NonEmpty
    seed: Annotated[int, Field(ge=0)] = 42
    fit_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.5
    selection_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.25
    target_error: Annotated[float, Field(gt=0, lt=1)] = 0.05
    min_routed_groups: Annotated[int, Field(ge=2)] = 30
    family_alpha: Annotated[float, Field(gt=0, lt=1)] = 0.05
    bootstrap_samples: Annotated[int, Field(ge=20)] = 500

    @model_validator(mode="after")
    def check_fractions(self) -> Self:
        if self.fit_fraction + self.selection_fraction >= 1:
            raise ValueError("Reserve a nonzero validation fraction")
        return self


def assign_roles(tickets: list[TicketInput], config: CalibrationConfig) -> list[str]:
    """Stable, label-independent allocation; identical normalized text stays together."""
    if len({ticket.case_id for ticket in tickets}) != len(tickets):
        raise ValueError("Duplicate calibration case IDs")
    roles = []
    for ticket in tickets:
        digest = hashlib.sha256(
            f"rightdesk-calibration-v1:{config.seed}:{duplicate_key(ticket.text)}".encode()
        ).hexdigest()
        position = int(digest, 16) / 2**256
        roles.append(
            "fit"
            if position < config.fit_fraction
            else "selection"
            if position < config.fit_fraction + config.selection_fraction
            else "validation"
        )
    if set(roles) != {"fit", "selection", "validation"}:
        raise ValueError("Calibration allocation must populate all three roles")
    return roles


def probability_matrix(predictions: list[ModelPrediction], queues: list[str]) -> np.ndarray:
    if not predictions or len(set(queues)) != len(queues):
        raise ValueError("Nonempty predictions and unique queues required")
    if any(
        p.queue_probabilities is None or set(p.queue_probabilities) != set(queues)
        for p in predictions
    ):
        raise ValueError("Every prediction must cover the exact probability vocabulary")
    return np.array([[p.queue_probabilities[q] for q in queues] for p in predictions])


class TemperatureCalibrator(Contract):
    version: NonEmpty
    model_revision: NonEmpty
    workflow_id: NonEmpty
    workflow_version: NonEmpty
    queues: Annotated[list[str], Field(min_length=2)]
    temperature: Annotated[float, Field(ge=0.05, le=20, allow_inf_nan=False)]
    fit_cases: Annotated[int, Field(ge=1)]
    fit_fingerprint: NonEmpty
    nll_before: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    nll_after: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    method: Literal["power_temperature_v1"] = "power_temperature_v1"
    zero_floor: Literal[1e-12] = 1e-12

    @model_validator(mode="after")
    def unique_queues(self) -> Self:
        if len(self.queues) != len(set(self.queues)):
            raise ValueError("Calibration queues must be unique")
        return self

    @classmethod
    def fit(cls, predictions: list[ModelPrediction], targets: list[str], version: str):
        if not predictions or len(predictions) != len(targets):
            raise ValueError("Fit predictions and targets must have equal nonzero length")
        if len({p.case_id for p in predictions}) != len(predictions):
            raise ValueError("Duplicate calibration fit IDs")
        first = predictions[0]
        if first.queue_probabilities is None:
            raise ValueError("Cannot calibrate a model without probabilities")
        queues = sorted(first.queue_probabilities)
        if not set(targets) <= set(queues):
            raise ValueError("Unknown calibration target")
        identity = (first.model_revision, first.workflow_id, first.workflow_version)
        if any(
            (p.model_revision, p.workflow_id, p.workflow_version) != identity
            or p.calibration_version is not None
            for p in predictions
        ):
            raise ValueError("Fit requires one uncalibrated model and workflow")
        logs = np.log(np.maximum(probability_matrix(predictions, queues), 1e-12))
        truth = np.array([queues.index(target) for target in targets])

        def loss(log_temperature):
            scaled = logs / math.exp(log_temperature)
            return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(truth)), truth]))

        result = minimize_scalar(loss, bounds=(math.log(0.05), math.log(20)), method="bounded")
        if not result.success or not math.isfinite(result.fun):
            raise ValueError("Temperature optimization failed")
        # Include identity/endpoints so fitted NLL never exceeds identity NLL.
        candidates = [0.0, float(result.x), math.log(0.05), math.log(20)]
        best = min(candidates, key=loss)
        fingerprint = hashlib.sha256(
            json.dumps(
                [(p.model_dump(), target) for p, target in zip(predictions, targets, strict=True)],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return cls(
            version=version,
            model_revision=first.model_revision,
            workflow_id=first.workflow_id,
            workflow_version=first.workflow_version,
            queues=queues,
            temperature=math.exp(best),
            fit_cases=len(predictions),
            fit_fingerprint=fingerprint,
            nll_before=loss(0.0),
            nll_after=loss(best),
        )

    def transform(self, prediction: ModelPrediction, workflow) -> ModelPrediction:
        prediction = ModelPrediction.for_workflow(prediction, workflow)
        if (prediction.model_revision, prediction.workflow_id, prediction.workflow_version) != (
            self.model_revision,
            self.workflow_id,
            self.workflow_version,
        ) or prediction.calibration_version is not None:
            raise ValueError("Calibration model/workflow mismatch or already calibrated")
        values = probability_matrix([prediction], self.queues)[0]
        logs = np.log(np.maximum(values, self.zero_floor)) / self.temperature
        probs = np.exp(logs - logsumexp(logs))
        data = prediction.model_dump()
        data.update(
            queue_probabilities=dict(zip(self.queues, probs.tolist(), strict=True)),
            calibration_version=self.version,
        )
        # Preserve the original tie choice, and reject any numerical argmax change.
        chosen = prediction.selected_queue()
        if data["queue_probabilities"][chosen] != max(data["queue_probabilities"].values()):
            raise ValueError("Temperature transform unexpectedly changed the selected queue")
        data["predicted_queue"] = chosen
        return ModelPrediction.for_workflow(data, workflow)


def routed_mask(
    predictions: list[ModelPrediction], threshold: float | None, native_rules: bool = False
) -> np.ndarray:
    if native_rules:
        return np.array([p.selected_queue() is not None for p in predictions], dtype=bool)
    if threshold is None:
        return np.zeros(len(predictions), dtype=bool)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid routing threshold")
    if any(p.queue_probabilities is None for p in predictions):
        raise ValueError("Threshold routing requires probabilities")
    return np.array([max(p.queue_probabilities.values()) >= threshold for p in predictions])


def routing_summary(predictions, targets, groups, mask, alpha=0.05) -> dict:
    """Ticket risk plus an independent-group, any-error binomial upper bound."""
    n = len(predictions)
    if not n or not (n == len(targets) == len(groups) == len(mask)) or not 0 < alpha < 1:
        raise ValueError("Invalid routing summary inputs")
    errors = [p.selected_queue() != target for p, target in zip(predictions, targets, strict=True)]
    routed_groups = {}
    for group, routed, wrong in zip(groups, mask, errors, strict=True):
        if routed:
            routed_groups[group] = routed_groups.get(group, False) or wrong
    count = int(np.sum(mask))
    wrong = sum(bool(route and error) for route, error in zip(mask, errors, strict=True))
    group_count, bad_groups = len(routed_groups), sum(routed_groups.values())
    upper = (
        None
        if not group_count
        else 1.0
        if bad_groups == group_count
        else float(beta.ppf(1 - alpha, bad_groups + 1, group_count - bad_groups))
    )
    return {
        "cases": n,
        "auto_count": count,
        "review_count": n - count,
        "automation_coverage": count / n,
        "error_count": wrong,
        "routing_error": wrong / count if count else None,
        "routed_groups": group_count,
        "groups_with_any_error": bad_groups,
        "group_error_upper": upper,
        "group_bound_alpha": alpha,
    }


def select_threshold(predictions, targets, groups, config: CalibrationConfig) -> dict:
    """Maximize selection coverage under observed ticket risk; never split tied scores."""
    if not predictions:
        raise ValueError("Empty threshold selection data")
    if any(p.queue_probabilities is None for p in predictions):
        raise ValueError("Rules have no probability threshold")
    scores = sorted({max(p.queue_probabilities.values()) for p in predictions})
    curve = []
    for threshold in scores:
        point = routing_summary(predictions, targets, groups, routed_mask(predictions, threshold))
        point.update(threshold=threshold)
        curve.append(point)
    feasible = [
        point
        for point in curve
        if point["routing_error"] <= config.target_error
        and point["routed_groups"] >= config.min_routed_groups
    ]
    selected = max(feasible, key=lambda p: p["auto_count"]) if feasible else None
    return {
        "threshold": selected["threshold"] if selected else None,
        "selection": selected,
        "curve": curve,
        "reason": "max_coverage_under_observed_target"
        if selected
        else "no_threshold_meets_selection_requirements",
    }
