"""Publish small aggregate results from verified local artifacts; no ticket text or labels."""

import json
from pathlib import Path

from rightdesk.data.importers.abcd import checksum

ROOT = Path(__file__).resolve().parents[1]


def verified(relative):
    directory = ROOT / relative
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    for name, expected in run["artifacts"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or checksum(path) != expected:
            raise ValueError(f"Changed artifact: {name}")
    return run


def main():
    final = verified("artifacts/final-test-v2")
    e2e = verified("artifacts/end-to-end-v2")
    development = verified("artifacts/shadow/development-v1")
    protocol = final["protocol"]
    metrics = {
        name: {
            key: value
            for key, value in report.items()
            if key
            in {
                "accuracy",
                "accuracy_ci95",
                "macro_f1",
                "brier_score",
                "calibration_error",
                "operating_point",
                "frozen_candidate",
                "enabled_policy",
                "policy_enabled",
                "model_revision",
                "calibration_version",
                "cases",
                "duplicate_groups",
                "uncertainty",
            }
        }
        for name, report in final["metrics"].items()
    }
    summary = {
        "schema_version": 1,
        "status": "implementation complete; local research prototype",
        "completed_at": final["completed_at"],
        "final_test_cases": final["evaluation_cases"],
        "reference": protocol["reference"],
        "dataset_manifest_sha256": protocol["input_sha256"]["manifest"],
        "final_protocol_sha256": final["protocol_sha256"],
        "final_run_sha256": checksum(ROOT / "artifacts/final-test-v2/run.json"),
        "development_run_sha256": protocol["input_sha256"]["source"],
        "calibration_run_sha256": protocol["input_sha256"]["calibration"],
        "packages": protocol["packages"],
        "base_models_refitted": False,
        "thresholds_reselected": False,
        "gate_changes": False,
        "failed_cases": final["failed_cases"],
        "metrics": metrics,
        "end_to_end": {
            key: e2e[key]
            for key in (
                "primary_unchanged",
                "restart_idempotent",
                "primary_verification",
                "shadow_models",
            )
        },
        "development_shadow": {
            "events": development["events"],
            "cases": development["cases"],
            "models": development["models"],
        },
        "human_intake_annotations": 0,
        "human_question_usefulness": None,
        "limitations": [
            "provisional eventual-owner targets",
            "no validated automatic routing",
            "no measured customer benefit",
            "test consumed; no further tuning",
        ],
    }
    (ROOT / "docs/results.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# RightDesk results and completion",
        "",
        "All fourteen implementation phases are complete as a local research prototype.",
        "The held-out test has **1,004 cases**, with zero failed inference cases. The models,",
        "calibrators and gates were frozen before scoring; no refitting or retuning occurred.",
        "Targets are provisional scenario-derived eventual owners, not reviewed intake judgments.",
        "",
        "## Held-out comparison",
        "",
        "| Variant | Accuracy | Macro F1 | Brier | ECE | Native coverage | Enabled coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def number(value):
        return "N/A" if value is None else f"{value:.4f}"

    for name, report in metrics.items():
        lines.append(
            f"| {name} | {report['accuracy']:.2%} | {report['macro_f1']:.4f} | "
            f"{number(report['brier_score'])} | {number(report['calibration_error'])} | "
            f"{report['operating_point']['automation_coverage']:.2%} | "
            f"{report['enabled_policy']['automation_coverage']:.2%} |"
        )
    lines += [
        "",
        "TF-IDF/logistic outperformed the configured zero-shot Laya model on these labels.",
        "This benchmark provides **no evidence of a Laya advantage**. The comparison is between",
        "practical systems: TF-IDF is trained on 5,792 examples and Laya is used zero-shot.",
        "It does not establish an inherent limit of Laya or compare equal training budgets.",
        "",
        "Temperature scaling improved TF-IDF Brier from 0.2676 to 0.2373 and ECE from",
        "0.1358 to 0.0411. Laya Brier improved from 0.6759 to 0.6277 and ECE from 0.2024",
        "to 0.1157. Argmax predictions and accuracy stay the same under temperature scaling.",
        "",
        "All enabled policy coverage is **0%**. No candidate passed the earlier full calibration",
        "gate, and final-test observations do not change that decision. Native rule coverage",
        "is a separate diagnostic; it is not enabled automation.",
        "",
        "The 500-replicate normalized-message cluster bootstrap gives descriptive pointwise",
        "intervals, not safety guarantees. Exact intervals, counts, candidate risks, revisions",
        "and provenance hashes are preserved in [results.json](results.json). Full local",
        "confusion matrices, per-queue metrics, reliability bins and curves remain under",
        "`artifacts/final-test-v2/`. Timing mixes serial baseline calls and amortized Laya",
        "batches, so it is not a like-for-like online latency comparison.",
        "",
        "## Operational demonstration",
        "",
        "The real-model process demonstration completed **14 synthetic tickets, 81 events and",
        "84 proposals** (21 arrived-input checkpoints × 4 models). Every model had zero failures.",
        "The primary state remained unchanged. Independent worker processes resumed correctly,",
        "A third process produced zero new proposals. All 14 lifecycle assertions passed.",
        "Laya proposed one question in this demonstration; no shadow question was sent.",
        "",
        "A separate **50-ticket development replay** produced 172 observed events and 200",
        "proposals. Laya proposed two questions; TF-IDF proposed one. There were no model",
        "failures or enabled routes, and both observers caught up fully with idempotent restart.",
        "Agreement with the incumbent is descriptive agreement, not correctness or benefit.",
        "",
        "## What remains unmeasured",
        "",
        "The 50-case blind reviewer queue has no genuine human annotations. Intake sufficiency,",
        "acceptable queues, question usefulness, inter-reviewer agreement and customer benefit",
        "remain unmeasured. Synthetic lifecycle success does not replace those judgments.",
        "The local prototype uses simulated service times and has no live helpdesk integration.",
        "",
        "## Reproduction and audit",
        "",
        "See [handoff](handoff.md) for setup, phase inventory, checks, exact commands and the",
        "audit record of corrected integration attempts. Source data and model weights stay",
        "local; this report contains aggregate results only. The final test is consumed and",
        "must not be reused to select a model or tune thresholds.",
        "",
    ]
    (ROOT / "docs/results.md").write_text("\n".join(lines), encoding="utf-8")
    print(ROOT / "docs/results.md")


if __name__ == "__main__":
    main()
