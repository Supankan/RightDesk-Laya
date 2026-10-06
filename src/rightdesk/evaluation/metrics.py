"""Queue metrics and tie-preserving coverage curves with cluster bootstrap intervals."""

import numpy as np

from rightdesk.schemas import ModelPrediction


def _interval(values: np.ndarray, enabled: bool) -> list[float] | None:
    values = values[np.isfinite(values)]
    return np.quantile(values, [0.025, 0.975]).tolist() if enabled and len(values) else None


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def evaluate(
    predictions: list[ModelPrediction],
    targets: list[str],
    groups: list[str],
    queues: list[str],
    latencies_ms: list[float],
    seed: int = 42,
    bootstrap_samples: int = 500,
) -> dict:
    """Targets are eventual-owner proxies. No confidence threshold is selected here."""
    n = len(predictions)
    if not n or not (len(targets) == len(groups) == len(latencies_ms) == n):
        raise ValueError("Evaluation arrays must have equal nonzero length")
    if len(set(queues)) != len(queues) or not set(targets) <= set(queues):
        raise ValueError("Invalid queue vocabulary or unknown reference labels")
    if len({prediction.case_id for prediction in predictions}) != n:
        raise ValueError("Duplicate prediction IDs")
    if bootstrap_samples < 20:
        raise ValueError("At least 20 bootstrap samples are required")
    if not np.isfinite(latencies_ms).all() or any(value < 0 for value in latencies_ms):
        raise ValueError("Latencies must be finite and nonnegative")
    chosen = [prediction.selected_queue() for prediction in predictions]
    if any(queue is not None and queue not in queues for queue in chosen):
        raise ValueError("Prediction contains an unknown queue")
    routed = np.array([queue is not None for queue in chosen])
    correct = np.array([queue == target for queue, target in zip(chosen, targets, strict=True)])
    _, inverse = np.unique(groups, return_inverse=True)
    group_count = int(inverse.max()) + 1
    rng = np.random.default_rng(seed)
    cluster_weights = rng.multinomial(
        group_count, np.full(group_count, 1 / group_count), size=bootstrap_samples
    )
    weights = cluster_weights[:, inverse]
    total_weights = weights.sum(axis=1)
    routed_weights = weights[:, routed].sum(axis=1)
    wrong_weights = weights[:, routed & ~correct].sum(axis=1)
    ci_enabled = group_count >= 2
    auto_count = int(routed.sum())
    errors = int((routed & ~correct).sum())
    operating = {
        "auto_count": auto_count,
        "review_count": n - auto_count,
        "automation_coverage": auto_count / n,
        "error_count": errors,
        "routing_error": errors / auto_count if auto_count else None,
        "routing_error_ci95": _interval(_ratio(wrong_weights, routed_weights), ci_enabled),
        "coverage_ci95": _interval(routed_weights / total_weights, ci_enabled),
        "bootstrap_valid_replicates": int((routed_weights > 0).sum()),
    }
    # Rows = reference queues; final column = abstention, which counts as a false negative.
    matrix = np.zeros((len(queues), len(queues) + 1), dtype=int)
    for truth, selected in zip(targets, chosen, strict=True):
        matrix[queues.index(truth), queues.index(selected) if selected else len(queues)] += 1
    per_queue = {}
    for index, queue in enumerate(queues):
        tp, support, predicted = (
            int(matrix[index, index]),
            int(matrix[index].sum()),
            int(matrix[:, index].sum()),
        )
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_queue[queue] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted,
        }
    report = {
        "cases": n,
        "duplicate_groups": group_count,
        "accuracy": float(correct.mean()),
        "accuracy_ci95": _interval(weights[:, correct].sum(axis=1) / total_weights, ci_enabled),
        "macro_f1": float(np.mean([entry["f1"] for entry in per_queue.values()])),
        "per_queue": per_queue,
        "operating_point": operating,
        "confusion_matrix": {
            "rows": queues,
            "columns": [*queues, "review"],
            "counts": matrix.tolist(),
        },
        "brier_score": None,
        "calibration_error": None,
        "reliability_bins": [],
        "latency_p50_ms": float(np.quantile(latencies_ms, 0.5)),
        "latency_p95_ms": float(np.quantile(latencies_ms, 0.95)),
        "uncertainty": {
            "method": "percentile cluster bootstrap of normalized-text groups",
            "samples": bootstrap_samples,
            "seed": seed,
            "confidence_level": 0.95,
            "limitations": (
                "Descriptive pointwise intervals, not safety bounds. Zero errors can yield "
                "a degenerate interval; sparse routes and few groups are unreliable."
            ),
        },
    }
    all_probabilities = all(
        prediction.queue_probabilities is not None for prediction in predictions
    )
    if (
        any(prediction.queue_probabilities is not None for prediction in predictions)
        and not all_probabilities
    ):
        raise ValueError(
            "Evaluate homogeneous probability availability; do not drop missing values"
        )
    curve = [
        {
            "threshold": None,
            "auto_count": 0,
            "review_count": n,
            "automation_coverage": 0.0,
            "error_count": 0,
            "routing_error": None,
            "routing_error_ci95": None,
            "bootstrap_valid_replicates": 0,
        }
    ]
    if all_probabilities:
        if any(set(prediction.queue_probabilities) != set(queues) for prediction in predictions):
            raise ValueError("Probability columns differ from the evaluation queues")
        probabilities = np.array(
            [[prediction.queue_probabilities[q] for q in queues] for prediction in predictions]
        )
        confidence = probabilities.max(axis=1)
        one_hot = np.eye(len(queues))[[queues.index(target) for target in targets]]
        report["brier_score"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        bins = np.minimum((confidence * 10).astype(int), 9)
        ece = 0.0
        for index in range(10):
            mask = bins == index
            count = int(mask.sum())
            accuracy = float(correct[mask].mean()) if count else None
            mean_confidence = float(confidence[mask].mean()) if count else None
            if count:
                ece += count / n * abs(accuracy - mean_confidence)
            report["reliability_bins"].append(
                {
                    "lower": index / 10,
                    "upper": (index + 1) / 10,
                    "count": count,
                    "accuracy": accuracy,
                    "mean_confidence": mean_confidence,
                }
            )
        report["calibration_error"] = ece
        order = np.argsort(-confidence, kind="stable")
        sorted_confidence = confidence[order]
        cumulative_errors = np.cumsum(~correct[order])
        boot_counts = np.cumsum(weights[:, order], axis=1)
        boot_errors = np.cumsum(weights[:, order] * (~correct[order])[None, :], axis=1)
        # Emit only ends of tied score groups: never split ties using truth or input order.
        ends = np.flatnonzero(np.r_[sorted_confidence[:-1] != sorted_confidence[1:], True])
        for end in ends:
            count, error_count = int(end + 1), int(cumulative_errors[end])
            curve.append(
                {
                    "threshold": float(sorted_confidence[end]),
                    "auto_count": count,
                    "review_count": n - count,
                    "automation_coverage": count / n,
                    "error_count": error_count,
                    "routing_error": error_count / count,
                    "routing_error_ci95": _interval(
                        _ratio(boot_errors[:, end], boot_counts[:, end]), ci_enabled
                    ),
                    "bootstrap_valid_replicates": int((boot_counts[:, end] > 0).sum()),
                }
            )
        report["selection_method"] = "uncalibrated maximum probability >= threshold"
    else:
        curve.append({"threshold": None, **operating})
        report["selection_method"] = "native rule matches; no confidence threshold available"
    report["risk_coverage_curve"] = curve
    return report
