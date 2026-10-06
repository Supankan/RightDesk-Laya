"""Portable benchmark summaries; complete curves and bins remain in metrics.json."""

import csv
from pathlib import Path


def _percent(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def write_report(directory: Path, run: dict) -> None:
    lines = [
        "# " + run.get("report_title", "RightDesk baseline benchmark"),
        "",
        f"Evaluation: **{run['evaluation_split']}** ({run['evaluation_cases']:,} cases). "
        f"Baseline training used {run['training_cases']:,} cases. "
        + ("Laya is zero-shot." if "laya" in run["models_run"] else ""),
        "",
        "Targets are provisional scenario-derived eventual queues, not reviewed intake judgments.",
        run.get("timing_note", "All model latencies use serial single-ticket calls."),
        "Probabilities are uncalibrated; no automatic-routing threshold has been selected.",
        "",
        "Accuracy uses all cases; abstentions count as incorrect. "
        "Macro F1 includes all configured queues.",
        "Routing error uses only routed cases. Comparing different coverage levels "
        "is not a matched-risk comparison.",
        "",
        "| Model | Accuracy | Macro F1 | Coverage | Routing error | "
        "Routed / errors | Brier | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in run["metrics"].items():
        point = report["operating_point"]
        brier = "N/A" if report["brier_score"] is None else f"{report['brier_score']:.4f}"
        lines.append(
            f"| {name} | {_percent(report['accuracy'])} | {report['macro_f1']:.4f} | "
            f"{_percent(point['automation_coverage'])} | {_percent(point['routing_error'])} | "
            f"{point['auto_count']} / {point['error_count']} | {brier} | "
            f"{report['latency_p95_ms']:.3f} |"
        )
    lines += [
        "",
        "## Uncertainty and interpretation",
        "",
        "95% percentile bootstrap intervals resample whole normalized-message groups, "
        "with a fixed seed.",
        "These are pointwise descriptive intervals, not simultaneous bands or safety guarantees.",
        "Sparse routes and zero-error samples can give misleading or degenerate intervals; "
        "inspect counts.",
        "Timing: " + run["latency_method"],
        "No sufficiency judgments or useful questions are inferred by these baselines.",
        "",
    ]
    for name, report in run["metrics"].items():
        point = report["operating_point"]
        lines += [
            f"## {name}",
            "",
            f"Accuracy 95% interval: {report['accuracy_ci95']}",
            f"Routing-error 95% interval: {point['routing_error_ci95']}",
            "",
            "### Per-queue performance",
            "",
            "| Queue | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
        for queue, metrics in report["per_queue"].items():
            lines.append(
                f"| {queue} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                f"{metrics['f1']:.3f} | {metrics['support']} |"
            )
        matrix = report["confusion_matrix"]
        lines += [
            "",
            "### Confusion matrix",
            "",
            "Rows are reference queues; columns are predictions.",
            "",
            "| Reference | " + " | ".join(matrix["columns"]) + " |",
            "|---|" + "---:|" * len(matrix["columns"]),
        ]
        for queue, counts in zip(matrix["rows"], matrix["counts"], strict=True):
            lines.append(f"| {queue} | " + " | ".join(map(str, counts)) + " |")
        lines += [
            "",
            "### Diagnostic threshold examples",
            "",
            "These fixed score cutoffs are examples, not chosen operating policies.",
            "",
            "| Minimum probability | Routed | Errors | Coverage | Routing error |",
            "|---|---:|---:|---:|---:|",
        ]
        if report["brier_score"] is None:
            lines.append("Rules have no probability score; use the native operating point above.")
        else:
            for threshold in (0.5, 0.7, 0.9, 0.95):
                eligible = [
                    p
                    for p in report["risk_coverage_curve"]
                    if p["threshold"] is not None and p["threshold"] >= threshold
                ]
                selected = eligible[-1] if eligible else report["risk_coverage_curve"][0]
                lines.append(
                    f"| {threshold:.2f} | {selected['auto_count']} | {selected['error_count']} | "
                    f"{_percent(selected['automation_coverage'])} | "
                    f"{_percent(selected['routing_error'])} |"
                )
        lines += [
            "",
            f"Full metrics, reliability bins, and the curve: "
            f"[{name}/metrics.json]({name}/metrics.json).",
            "",
        ]
        curve = report["risk_coverage_curve"]
        columns = [
            "threshold",
            "auto_count",
            "review_count",
            "automation_coverage",
            "error_count",
            "routing_error",
        ]
        with (directory / name / "risk_coverage.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(curve)
    lines += [
        "## Reproducibility",
        "",
        "`run.json` records configuration, dataset/code hashes, package versions, "
        "and artifact checksums.",
        "`pipeline.joblib` is a locally generated sklearn artifact; only load trusted model files.",
        "Calibration is always reserved. Test evaluation occurs only when explicitly requested.",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
