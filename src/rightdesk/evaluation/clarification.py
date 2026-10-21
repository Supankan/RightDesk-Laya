"""Offline question-selection coverage and explicitly labeled human usefulness metrics."""

import json
import tempfile
from collections import Counter
from pathlib import Path

from rightdesk.config import _load
from rightdesk.data.importers.abcd import checksum
from rightdesk.policy.clarification import ClarificationConfig, IntakePolicy
from rightdesk.policy.routing import RoutingSettings
from rightdesk.schemas import ModelPrediction, TicketInput, WorkflowConfig
from rightdesk.services import prediction_rows
from rightdesk.storage import Store, encode, utc_now


def score_usefulness(proposals: dict[str, str | None], labels: list[dict]) -> dict:
    joined = [label for label in labels if label["case_id"] in proposals]
    known = [label for label in joined if label["useful_question_ids"] is not None]
    asked = [label for label in known if proposals[label["case_id"]] is not None]
    helpful = sum(proposals[label["case_id"]] in label["useful_question_ids"] for label in asked)
    positive = [label for label in known if label["useful_question_ids"]]
    positives_helped = sum(
        proposals[label["case_id"]] in label["useful_question_ids"] for label in positive
    )
    sufficient = [label for label in joined if label["sufficient_information_at_intake"] is True]
    return {
        "individual_judgments": len(joined),
        "known_usefulness_judgments": len(known),
        "asked_with_known_usefulness": len(asked),
        "useful_asked": helpful,
        "useful_question_precision": helpful / len(asked) if asked else None,
        "positive_usefulness_judgments": len(positive),
        "useful_question_recall": positives_helped / len(positive) if positive else None,
        "already_sufficient_judgments": len(sufficient),
        "ask_rate_when_already_sufficient": sum(
            proposals[label["case_id"]] is not None for label in sufficient
        )
        / len(sufficient)
        if sufficient
        else None,
        "limitation": (
            "Latest individual human judgments on the joined subset; unknowns excluded per field. "
            "No consensus or population estimate."
        ),
    }


def evaluate_clarification(
    store: Store, run_id: str, model_id: str, catalog: Path, output: Path
) -> Path:
    if output.exists():
        raise ValueError("Clarification output exists; choose a fresh directory")
    run = store.run(run_id)
    rows = prediction_rows(store, run_id, model_id)
    if not rows:
        raise ValueError("No saved predictions for the requested model")
    model = next(model for model in store.models(run_id) if model["model_id"] == model_id)
    workflow_data = model["workflow"]
    workflow_data["policy"].update(
        version="question-prototype:" + model_id,
        clarification_enabled=True,
        max_questions=1,
        routing_threshold=None,
        sufficient_information_threshold=None,
        calibration_version=rows[0]["prediction"]["calibration_version"],
    )
    workflow = WorkflowConfig.model_validate(workflow_data)
    config = _load(catalog, ClarificationConfig)
    policy = IntakePolicy(
        workflow,
        RoutingSettings(
            model_revision=rows[0]["prediction"]["model_revision"],
            calibration_version=rows[0]["prediction"]["calibration_version"],
            enabled=False,
            native_rules=rows[0]["prediction"]["queue_probabilities"] is None,
        ),
        config,
    )
    proposals, decisions = {}, []
    basis = Counter()
    for row in rows:
        case = store.one(
            "SELECT ticket_json FROM cases WHERE dataset_id=? AND case_id=?",
            (run["dataset_id"], row["case_id"]),
        )
        ticket = TicketInput.model_validate_json(case["ticket_json"])
        prediction = ModelPrediction.for_workflow(row["prediction"], workflow)
        choice = policy.selector.select(ticket, prediction, [])
        decision = policy.decide(ticket, prediction, [])
        proposals[ticket.case_id] = decision.question_id
        decisions.append({"decision": decision.model_dump(), "eligibility": choice.model_dump()})
        basis[choice.basis] += 1
    review_rows = [
        row for row in store.latest_reviews(run["dataset_id"]) if row["case_id"] in proposals
    ]
    labels = [json.loads(row["label_json"]) for row in review_rows]
    usefulness = score_usefulness(proposals, labels)
    asked_count = sum(question is not None for question in proposals.values())
    report = {
        "schema_version": 1,
        "phase": 11,
        "created_at": utc_now(),
        "run_id": run_id,
        "model_id": model_id,
        "source_run_sha256": run["source_sha256"],
        "dataset_id": run["dataset_id"],
        "evaluation_split": run["split"],
        "cases": len(rows),
        "proposed_questions": asked_count,
        "question_coverage": asked_count / len(rows),
        "review_count": len(rows) - asked_count,
        "question_counts": dict(Counter(q for q in proposals.values() if q)),
        "eligibility_counts": dict(basis),
        "human_usefulness": usefulness,
        "workflow": workflow.model_dump(),
        "catalog": config.model_dump(),
        "catalog_sha256": checksum(catalog),
        "test_evaluated": False,
        "routing_gate_enabled": False,
        "sufficiency_estimated": False,
        "policy_code_sha256": checksum(Path(__file__).parents[1] / "policy/clarification.py"),
        "evaluation_code_sha256": checksum(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".questions-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "proposals.jsonl").write_text(
            "".join(encode(row) + "\n" for row in decisions), encoding="utf-8"
        )
        (staging / "review_snapshot.jsonl").write_text(
            "".join(
                encode(
                    {
                        "annotation_id": row["annotation_id"],
                        "revision": row["revision"],
                        "label": json.loads(row["label_json"]),
                    }
                )
                + "\n"
                for row in review_rows
            ),
            encoding="utf-8",
        )
        precision = usefulness["useful_question_precision"]
        text = [
            "# RightDesk bounded clarification prototype",
            "",
            f"Saved model: {model_id}; split: {run['split']}; cases: {len(rows)}.",
            f"Questions proposed: {asked_count} ({100 * asked_count / len(rows):.2f}%); "
            f"review: {len(rows) - asked_count}.",
            "These are offline proposals. No question was sent, routing enabled, "
            "or sufficiency probability inferred.",
            "",
            f"Known human usefulness judgments: {usefulness['known_usefulness_judgments']}.",
            "Useful-question precision: "
            + ("not measurable yet" if precision is None else f"{precision:.4f}")
            + ".",
            "Usefulness requires actual individual human judgments; "
            "unknown fields remain excluded.",
            "Source eventual-owner labels and synthetic response scripts "
            "are not usefulness labels.",
            "",
            "Eligibility: " + encode(dict(basis)),
            "Question counts: " + encode(report["question_counts"]),
            "",
            "The catalog maps narrowly ambiguous queue pairs to configured questions. "
            "It rejects diffuse distributions, high-confidence choices, unsupported languages, "
            "missing probability evidence, and exhausted budgets. Queue uncertainty alone does not "
            "establish a useful question. The original routing experiment "
            "and final test remain unchanged.",
            "",
        ]
        (staging / "report.md").write_text("\n".join(text), encoding="utf-8")
        report["artifacts"] = {p.name: checksum(p) for p in staging.iterdir() if p.is_file()}
        (staging / "run.json").write_text(encode(report) + "\n", encoding="utf-8")
        staging.rename(output)
    return output / "report.md"
