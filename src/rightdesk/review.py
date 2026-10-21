"""Blind review queues, revisioned intake judgments, and reviewer agreement."""

import hashlib
import json
from itertools import combinations
from typing import Annotated, Literal

from pydantic import Field

from rightdesk.schemas import Contract, NonEmpty, ReviewedLabel, WorkflowConfig
from rightdesk.services import prediction_rows
from rightdesk.storage import Store, encode, utc_now


class QueueRequest(Contract):
    title: NonEmpty
    dataset_id: NonEmpty
    strategy: Literal["random", "disagreement", "errors", "low_confidence"] = "random"
    partition: Literal["development", "calibration"] = "development"
    run_id: NonEmpty | None = None
    model_id: NonEmpty | None = None
    comparison_model: NonEmpty | None = None
    seed: Annotated[int, Field(ge=0)] = 42
    limit: Annotated[int, Field(ge=1, le=5000)] = 50


class ReviewSubmission(Contract):
    dataset_id: NonEmpty
    case_id: NonEmpty
    reviewer_id: NonEmpty
    expected_revision: Annotated[int, Field(ge=0)]
    acceptable_queues_at_intake: list[NonEmpty] | None = None
    sufficient_information_at_intake: bool | None = None
    useful_question_ids: list[NonEmpty] | None = None
    notes: Annotated[str, Field(max_length=5000)] = ""


def create_queue(store: Store, request: QueueRequest) -> str:
    request = QueueRequest.model_validate(request)
    store.dataset(request.dataset_id)
    cases = store.query(
        "SELECT case_id,group_key FROM cases WHERE dataset_id=? AND partition_role=?",
        (request.dataset_id, request.partition),
    )
    eligible = {row["case_id"] for row in cases}
    if request.strategy != "random":
        if request.run_id is None or request.model_id is None:
            raise ValueError("This sampling strategy requires a run and model")
        run = store.run(request.run_id)
        if run["dataset_id"] != request.dataset_id:
            raise ValueError("Queue run/dataset mismatch")
        rows = prediction_rows(store, request.run_id, request.model_id)
        if not rows:
            raise ValueError("Sampling model has no predictions")
        if request.strategy == "errors":
            eligible &= {row["case_id"] for row in rows if not row["correct"]}
        elif request.strategy == "low_confidence":
            scored = [row for row in rows if row["confidence"] is not None]
            if not scored:
                raise ValueError("Selected model has no probability scores")
            scores = {row["case_id"]: row["confidence"] for row in scored}
            eligible &= scores.keys()
        else:
            if not request.comparison_model or request.comparison_model == request.model_id:
                raise ValueError("Disagreement sampling requires a different comparison model")
            other = {
                row["case_id"]: row["predicted_queue"]
                for row in prediction_rows(store, request.run_id, request.comparison_model)
            }
            if {row["case_id"] for row in rows} != other.keys():
                raise ValueError("Sampling models must have the same case membership")
            eligible &= {
                row["case_id"] for row in rows if row["predicted_queue"] != other[row["case_id"]]
            }

    def rank(row):
        digest = hashlib.sha256(f"{request.seed}:{row['case_id']}".encode()).hexdigest()
        return (
            (scores[row["case_id"]], digest) if request.strategy == "low_confidence" else (digest,)
        )

    selected, seen_groups = [], set()
    for row in sorted((row for row in cases if row["case_id"] in eligible), key=rank):
        if row["group_key"] not in seen_groups:
            selected.append(row["case_id"])
            seen_groups.add(row["group_key"])
        if len(selected) == request.limit:
            break
    if not selected:
        raise ValueError("No cases match the review queue criteria")
    payload = {"request": request.model_dump(), "case_ids": selected}
    queue_id = "queue-" + hashlib.sha256(encode(payload).encode()).hexdigest()[:20]
    with store.connection(write=True) as connection:
        if connection.execute(
            "SELECT 1 FROM review_queues WHERE queue_id=?", (queue_id,)
        ).fetchone():
            return queue_id
        connection.execute(
            "INSERT INTO review_queues VALUES(?,?,?,?,?)",
            (queue_id, request.dataset_id, request.title, encode(request.model_dump()), utc_now()),
        )
        connection.executemany(
            "INSERT INTO review_items VALUES(?,?,?,?)",
            [
                (queue_id, index, request.dataset_id, case_id)
                for index, case_id in enumerate(selected)
            ],
        )
    return queue_id


def list_queues(store: Store):
    return store.query("""SELECT q.*,COUNT(i.case_id) AS cases FROM review_queues q
        JOIN review_items i USING(queue_id) GROUP BY q.queue_id ORDER BY q.created_at DESC""")


def queue_items(store: Store, queue_id: str, reviewer_id: str):
    return store.query(
        """SELECT i.case_id,i.dataset_id,i.position,
        COALESCE((SELECT MAX(r.revision) FROM reviews r WHERE r.dataset_id=i.dataset_id
        AND r.case_id=i.case_id AND r.reviewer_id=?),0) AS revision
        FROM review_items i WHERE i.queue_id=? ORDER BY i.position""",
        (reviewer_id, queue_id),
    )


def blind_case(store: Store, dataset_id: str, case_id: str, reviewer_id: str):
    row = store.one(
        "SELECT ticket_json,partition_role FROM cases WHERE dataset_id=? AND case_id=?",
        (dataset_id, case_id),
    )
    # This response deliberately contains neither predictions nor source references.
    latest = store.query(
        """SELECT label_json,notes,revision FROM reviews
        WHERE dataset_id=? AND case_id=? AND reviewer_id=? ORDER BY revision DESC LIMIT 1""",
        (dataset_id, case_id, reviewer_id),
    )
    return {
        "ticket": json.loads(row["ticket_json"]),
        "partition": row["partition_role"],
        "own_review": (
            {
                "label": json.loads(latest[0]["label_json"]),
                "notes": latest[0]["notes"],
                "revision": latest[0]["revision"],
            }
            if latest
            else None
        ),
    }


def submit_review(store: Store, submission: ReviewSubmission) -> int:
    submission = ReviewSubmission.model_validate(submission)
    workflow = WorkflowConfig.model_validate(store.dataset(submission.dataset_id)["workflow"])
    # The source proxy target is copied unchanged; reviewers provide intake judgments only.
    reference = store.one(
        "SELECT label_json FROM reference_labels WHERE dataset_id=? AND case_id=?",
        (submission.dataset_id, submission.case_id),
    )
    eventual = json.loads(reference["label_json"])["eventual_owner_queue"]
    label = ReviewedLabel.for_workflow(
        {
            "case_id": submission.case_id,
            "workflow_id": workflow.workflow_id,
            "workflow_version": workflow.version,
            "eventual_owner_queue": eventual,
            "acceptable_queues_at_intake": submission.acceptable_queues_at_intake,
            "sufficient_information_at_intake": submission.sufficient_information_at_intake,
            "useful_question_ids": submission.useful_question_ids,
            "review_status": "reviewed",
            "reviewer_id": submission.reviewer_id,
        },
        workflow,
    )
    with store.connection(write=True) as connection:
        current = connection.execute(
            """SELECT COALESCE(MAX(revision),0) FROM reviews
            WHERE dataset_id=? AND case_id=? AND reviewer_id=?""",
            (submission.dataset_id, submission.case_id, submission.reviewer_id),
        ).fetchone()[0]
        if current != submission.expected_revision:
            raise ValueError("Review changed in another session; reload before saving")
        revision = current + 1
        connection.execute(
            """INSERT INTO reviews(dataset_id,case_id,reviewer_id,revision,
            label_json,notes,created_at) VALUES(?,?,?,?,?,?,?)""",
            (
                submission.dataset_id,
                submission.case_id,
                submission.reviewer_id,
                revision,
                encode(label.model_dump()),
                submission.notes,
                utc_now(),
            ),
        )
    return revision


def review_progress(store: Store, queue_id: str, reviewer_id: str):
    items = queue_items(store, queue_id, reviewer_id)
    return {
        "total": len(items),
        "completed": sum(row["revision"] > 0 for row in items),
        "remaining": sum(row["revision"] == 0 for row in items),
    }


def export_reviews(store: Store, dataset_id: str, history=False) -> str:
    store.dataset(dataset_id)
    rows = (
        store.query(
            "SELECT * FROM reviews WHERE dataset_id=? ORDER BY annotation_id", (dataset_id,)
        )
        if history
        else store.latest_reviews(dataset_id)
    )
    return "".join(
        encode(
            {
                **{key: value for key, value in row.items() if key != "label_json"},
                "label": json.loads(row["label_json"]),
            }
        )
        + "\n"
        for row in rows
    )


def agreement(store: Store, dataset_id: str) -> dict:
    latest = store.latest_reviews(dataset_id)
    by_reviewer = {}
    for row in latest:
        by_reviewer.setdefault(row["reviewer_id"], {})[row["case_id"]] = json.loads(
            row["label_json"]
        )
    result = []
    for left, right in combinations(sorted(by_reviewer), 2):
        shared = sorted(by_reviewer[left].keys() & by_reviewer[right].keys())
        pairs = [(by_reviewer[left][case], by_reviewer[right][case]) for case in shared]
        known = [
            (a["sufficient_information_at_intake"], b["sufficient_information_at_intake"])
            for a, b in pairs
            if a["sufficient_information_at_intake"] is not None
            and b["sufficient_information_at_intake"] is not None
        ]
        observed = sum(a == b for a, b in known) / len(known) if known else None
        kappa = None
        if known:
            pa, pb = (sum(pair[index] for pair in known) / len(known) for index in (0, 1))
            expected = pa * pb + (1 - pa) * (1 - pb)
            if expected < 1 and len(known) >= 2:
                kappa = (observed - expected) / (1 - expected)
        queues = [
            (set(a["acceptable_queues_at_intake"]), set(b["acceptable_queues_at_intake"]))
            for a, b in pairs
            if a["acceptable_queues_at_intake"] is not None
            and b["acceptable_queues_at_intake"] is not None
        ]
        result.append(
            {
                "reviewer_a": left,
                "reviewer_b": right,
                "shared_cases": len(shared),
                "sufficiency_pairs": len(known),
                "sufficiency_agreement": observed,
                "sufficiency_kappa": kappa,
                "queue_pairs": len(queues),
                "exact_queue_agreement": sum(a == b for a, b in queues) / len(queues)
                if queues
                else None,
                "mean_queue_jaccard": sum(
                    len(a & b) / len(a | b) if a | b else 1.0 for a, b in queues
                )
                / len(queues)
                if queues
                else None,
            }
        )
    return {
        "individual_latest_reviews": len(latest),
        "reviewers": len(by_reviewer),
        "pairs": result,
        "limitation": (
            "Selected overlapping cases; unknown fields excluded per metric. "
            "No automatic consensus."
        ),
    }
