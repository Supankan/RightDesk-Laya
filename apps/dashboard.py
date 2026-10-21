"""Local experiment explorer and blind human review. No inference runs in this app."""

import csv
import io
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rightdesk.review import (
    QueueRequest,
    ReviewSubmission,
    agreement,
    blind_case,
    create_queue,
    export_reviews,
    list_queues,
    queue_items,
    review_progress,
    submit_review,
)
from rightdesk.services import model_summaries, prediction_rows, reviewed_evaluation
from rightdesk.storage import Store
from rightdesk.ui.outcomes import render_final, render_shadow
from rightdesk.ui.simulator import render_simulator

ROOT = Path(__file__).resolve().parents[1]


def percent(value):
    return "—" if value is None or pd.isna(value) else f"{100 * value:.2f}%"


def overview(store, run, models):
    summaries = model_summaries(store, run["run_id"])
    calibration = run["kind"] == "calibration"
    st.subheader("Model comparison")
    st.caption(
        "Reference: provisional scenario-derived eventual owner. "
        "Human intake judgments are separate."
    )
    a, b, c, d = st.columns(4)
    a.metric("Evaluation tickets", summaries[0]["cases"])
    b.metric("Model variants", len(summaries))
    c.metric("Best queue accuracy", percent(max(row["accuracy"] for row in summaries)))
    if calibration:
        d.metric("Policies passing gate", sum(row["gate_passed"] for row in summaries))
        st.info(
            "Candidate threshold coverage is shown separately from enabled shadow-policy coverage. "
            "A failed gate leaves the policy in review-only mode."
        )
    else:
        d.metric("Evaluation split", "Development")
        st.caption(
            "Prediction coverage below counts model choices; "
            "these benchmark runs do not execute a routing policy."
        )
    frame = pd.DataFrame(summaries)
    columns = ["model", "accuracy", "macro_f1", "brier", "ece", "coverage"]
    if not calibration:
        columns += ["routing_error"]
    display = frame[columns].copy()
    for column in [
        "accuracy",
        "coverage",
        "routing_error",
        "candidate_coverage",
        "candidate_error",
    ]:
        if column in display:
            display[column] = display[column].map(percent)
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "model": "Model",
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "brier": "Brier",
            "ece": "ECE",
            "coverage": "Enabled coverage" if calibration else "Prediction coverage",
            "routing_error": "Routing error",
        },
    )
    if calibration:
        st.markdown("**Frozen candidate validation**")
        candidate_columns = [
            "model",
            "candidate_coverage",
            "candidate_error",
            "group_error_upper",
            "gate_passed",
        ]
        candidates = frame[candidate_columns].copy()
        for column in ["candidate_coverage", "candidate_error", "group_error_upper"]:
            candidates[column] = candidates[column].map(percent)
        st.dataframe(
            candidates,
            hide_index=True,
            width="stretch",
            column_config={
                "model": "Model",
                "candidate_coverage": "Candidate coverage",
                "candidate_error": "Ticket error",
                "group_error_upper": "Group error upper",
                "gate_passed": "Gate passed",
            },
        )
    chart = go.Figure()
    chart.add_bar(
        x=frame["model"], y=frame["accuracy"], name="Queue accuracy", marker_color="#087f8c"
    )
    chart.add_bar(
        x=frame["model"],
        y=frame["coverage"],
        name="Enabled policy coverage" if calibration else "Prediction coverage",
        marker_color="#93b9c4",
    )
    chart.update_layout(
        height=340,
        barmode="group",
        yaxis_tickformat=".0%",
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", y=1.15),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(chart, width="stretch")
    selected = st.selectbox("Inspect model metrics", list(models), key="metrics_model")
    metrics = models[selected]["metrics"]
    quality = metrics.get("quality", metrics)
    left, right = st.columns(2)
    with left:
        st.markdown("**Risk and coverage · descriptive curve**")
        points = [row for row in quality["risk_coverage_curve"] if row["routing_error"] is not None]
        fig = go.Figure(
            go.Scatter(
                x=[p["automation_coverage"] for p in points],
                y=[p["routing_error"] for p in points],
                mode="lines+markers",
                line_color="#087f8c",
                name=selected,
            )
        )
        fig.update_layout(
            height=300,
            xaxis_title="Prediction coverage",
            yaxis_title="Routing error",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            margin=dict(t=15, b=30),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Curves describe saved predictions. Inspecting them does not change a frozen threshold."
        )
    with right:
        st.markdown("**Probability reliability**")
        bins = [row for row in quality["reliability_bins"] if row["count"]]
        if bins:
            fig = go.Figure(
                go.Scatter(
                    x=[r["mean_confidence"] for r in bins],
                    y=[r["accuracy"] for r in bins],
                    mode="lines+markers",
                    marker_color="#087f8c",
                    text=[f"n={r['count']}" for r in bins],
                )
            )
            fig.add_scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="#a0a9ba"),
                showlegend=False,
            )
            fig.update_layout(
                height=300,
                xaxis_title="Mean confidence",
                yaxis_title="Observed accuracy",
                xaxis_range=[0, 1],
                yaxis_range=[0, 1],
                showlegend=False,
                margin=dict(t=15, b=30),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("This model does not produce probability estimates.")
    with st.expander("Per-queue metrics and confusion matrix"):
        st.dataframe(pd.DataFrame(quality["per_queue"]).T, width="stretch")
        matrix = quality["confusion_matrix"]
        st.dataframe(
            pd.DataFrame(matrix["counts"], index=matrix["rows"], columns=matrix["columns"]),
            width="stretch",
        )
    reviewed = reviewed_evaluation(store, run["run_id"], selected)
    with st.expander("Human-reviewed subset"):
        st.json(reviewed)
    with st.expander("Run provenance"):
        st.write("Run ID:", run["run_id"])
        st.write("Dataset fingerprint:", run["dataset_id"])
        st.write("Created (UTC):", run["created_at"])
        st.download_button(
            "Download original run manifest",
            json.dumps(run["manifest"], indent=2),
            file_name=f"{run['run_id']}.json",
            mime="application/json",
        )


def explorer(store, run, models):
    st.subheader("Ticket explorer")
    st.caption(
        "Inspect first customer messages alongside saved predictions and provisional references."
    )
    model = st.selectbox("Prediction model", list(models), key="explorer_model")
    rows = prediction_rows(store, run["run_id"], model)
    a, b, c = st.columns(3)
    with a:
        status = st.selectbox("Prediction outcome", ["All", "Incorrect", "Correct", "Abstained"])
    with b:
        queue = st.selectbox(
            "Predicted queue",
            ["All"] + sorted({row["predicted_queue"] for row in rows if row["predicted_queue"]}),
        )
    with c:
        action = st.selectbox("Policy action", ["All"] + sorted({row["action"] for row in rows}))
    search = st.text_input("Search ticket text or case ID")
    confidence = st.slider("Minimum confidence (probability models)", 0.0, 1.0, 0.0, 0.05)
    filtered = [
        row
        for row in rows
        if (
            status == "All"
            or status == "Correct"
            and row["correct"]
            or status == "Incorrect"
            and not row["correct"]
            or status == "Abstained"
            and row["predicted_queue"] is None
        )
        and (queue == "All" or row["predicted_queue"] == queue)
        and (action == "All" or row["action"] == action)
        and (not search or search.casefold() in (row["text"] + " " + row["case_id"]).casefold())
        and (confidence == 0 or row["confidence"] is not None and row["confidence"] >= confidence)
    ]
    st.caption(
        f"{len(filtered)} matching tickets. Table shows up to 200; downloads include every match."
    )
    if not filtered:
        st.info("No tickets match these filters.")
        return
    fields = [
        "case_id",
        "text",
        "predicted_queue",
        "reference_queue",
        "correct",
        "confidence",
        "action",
    ]
    st.dataframe(
        pd.DataFrame([{key: row[key] for key in fields} for row in filtered[:200]]),
        hide_index=True,
        width="stretch",
    )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in filtered)
    st.download_button(
        "Download filtered predictions",
        stream.getvalue(),
        file_name="rightdesk-predictions.csv",
        mime="text/csv",
    )
    chosen = st.selectbox("Inspect ticket", [row["case_id"] for row in filtered])
    row = next(row for row in filtered if row["case_id"] == chosen)
    st.text(row["text"])
    left, right = st.columns(2)
    left.json(row["prediction"])
    with right:
        if row["decision"]:
            st.json(row["decision"])
        else:
            st.info("Saved model prediction; this benchmark has no policy decision.")


def queue_creator(store, run, models):
    with st.expander("Create a review queue"):
        with st.form("create_review_queue"):
            title = st.text_input("Queue title", value="Intake review")
            strategy = st.selectbox(
                "Sampling strategy", ["random", "disagreement", "errors", "low_confidence"]
            )
            partition = st.selectbox("Review partition", ["development", "calibration"])
            model = st.selectbox("Sampling model", list(models))
            comparison = st.selectbox(
                "Comparison model", list(models), index=min(1, len(models) - 1)
            )
            limit = st.number_input("Maximum cases", min_value=1, max_value=5000, value=50)
            seed = st.number_input("Sampling seed", min_value=0, value=42)
            create = st.form_submit_button("Create queue")
        if create:
            try:
                queue_id = create_queue(
                    store,
                    QueueRequest(
                        title=title,
                        dataset_id=run["dataset_id"],
                        strategy=strategy,
                        partition=partition,
                        run_id=run["run_id"],
                        model_id=model,
                        comparison_model=comparison,
                        limit=int(limit),
                        seed=int(seed),
                    ),
                )
                st.success(f"Queue ready: {queue_id}")
            except ValueError as exc:
                st.error(str(exc))


def human_review(store, run, models):
    st.subheader("Human intake review")
    st.caption(
        "Judge the first customer message. Model outputs and source queue labels "
        "are hidden until you save."
    )
    st.caption("Record at least one intake judgment. Unknown fields may stay unknown.")
    queue_creator(store, run, models)
    queues = [row for row in list_queues(store) if row["dataset_id"] == run["dataset_id"]]
    if not queues:
        st.info("Create a review queue to begin.")
        return
    chosen_queue = st.selectbox(
        "Review queue",
        [row["queue_id"] for row in queues],
        format_func=lambda key: next(
            f"{row['title']} · {row['cases']} cases" for row in queues if row["queue_id"] == key
        ),
    )
    queue = next(row for row in queues if row["queue_id"] == chosen_queue)
    config = json.loads(queue["config_json"])
    st.caption(
        f"Selection: {config['strategy']} · {config['partition']}. "
        "Selected subsets do not estimate population performance."
    )
    reviewer = st.text_input(
        "Reviewer ID",
        key="reviewer_id",
        max_chars=80,
        help="Use your own stable ID. This local workbench does not authenticate identities.",
    )
    if not reviewer.strip():
        st.info("Enter your reviewer ID to open a blind review.")
        return
    reviewer = reviewer.strip()
    if "review_notice" in st.session_state:
        st.success(st.session_state.pop("review_notice"))
    progress = review_progress(store, chosen_queue, reviewer)
    st.progress(
        progress["completed"] / progress["total"],
        text=f"{progress['completed']} of {progress['total']} reviewed",
    )
    show_completed = st.checkbox("Include my completed cases")
    if st.session_state.pop("advance_review", False):
        st.session_state.pop("review_case", None)
    active_case = st.session_state.get("review_case")
    items = [
        row
        for row in queue_items(store, chosen_queue, reviewer)
        if show_completed or row["revision"] == 0 or row["case_id"] == active_case
    ]
    if items:
        options = [row["case_id"] for row in items]
        if st.session_state.get("review_case") not in options:
            st.session_state.pop("review_case", None)
        selected = st.selectbox("Review case", options, key="review_case")
        payload = blind_case(store, run["dataset_id"], selected, reviewer)
        own = payload["own_review"]
        context = f"{run['dataset_id']}:{reviewer}:{selected}"
        if st.session_state.get("review_context") != context:
            st.session_state["review_context"] = context
            st.session_state["review_revision"] = own["revision"] if own else 0
        expected_revision = st.session_state["review_revision"]
        workflow = store.dataset(run["dataset_id"])["workflow"]
        label = own["label"] if own else {}
        st.text(payload["ticket"]["text"])
        with st.expander("Queue and question definitions"):
            for definition in workflow["queues"]:
                st.write(f"{definition['id']}: {definition['description']}")
            for definition in workflow["questions"]:
                st.write(f"{definition['id']}: {definition['text']}")
        key = f"{context}:{expected_revision}"
        acceptable = label.get("acceptable_queues_at_intake")
        useful = label.get("useful_question_ids")
        sufficiency = label.get("sufficient_information_at_intake")
        with st.form("review_form:" + key):
            queue_options = ["Unknown", "Select acceptable queues", "No queue is acceptable yet"]
            queue_mode = st.selectbox(
                "Queue judgment",
                queue_options,
                index=0 if acceptable is None else 1 if acceptable else 2,
            )
            selected_queues = st.multiselect(
                "Acceptable queues",
                [row["id"] for row in workflow["queues"]],
                default=acceptable or [],
            )
            sufficient = st.radio(
                "Is there enough information to route?",
                ["Unknown", "Yes", "No"],
                index=0 if sufficiency is None else 1 if sufficiency else 2,
                horizontal=True,
            )
            question_options = [
                "Unknown",
                "Select useful questions",
                "No configured question is useful",
            ]
            question_mode = st.selectbox(
                "Clarification judgment",
                question_options,
                index=0 if useful is None else 1 if useful else 2,
            )
            selected_questions = st.multiselect(
                "Useful questions",
                [row["id"] for row in workflow["questions"]],
                default=useful or [],
            )
            notes = st.text_area(
                "Reviewer notes", value=own["notes"] if own else "", max_chars=5000
            )
            save = st.form_submit_button("Save judgment", type="primary")
        if save:
            try:
                if (
                    queue_mode == queue_options[0]
                    and sufficient == "Unknown"
                    and question_mode == question_options[0]
                ):
                    raise ValueError("Choose at least one intake judgment before saving.")
                if queue_mode == queue_options[1] and not selected_queues:
                    raise ValueError(
                        "Select an acceptable queue, or explicitly choose no acceptable queue."
                    )
                if question_mode == question_options[1] and not selected_questions:
                    raise ValueError(
                        "Select a useful question, or explicitly choose no useful question."
                    )
                revision = submit_review(
                    store,
                    ReviewSubmission(
                        dataset_id=run["dataset_id"],
                        case_id=selected,
                        reviewer_id=reviewer,
                        expected_revision=expected_revision,
                        acceptable_queues_at_intake=None
                        if queue_mode == queue_options[0]
                        else selected_queues
                        if queue_mode == queue_options[1]
                        else [],
                        sufficient_information_at_intake={
                            "Unknown": None,
                            "Yes": True,
                            "No": False,
                        }[sufficient],
                        useful_question_ids=None
                        if question_mode == question_options[0]
                        else selected_questions
                        if question_mode == question_options[1]
                        else [],
                        notes=notes,
                    ),
                )
                st.session_state["review_notice"] = (
                    f"Judgment saved · revision {revision}. Next pending case is ready."
                )
                st.session_state.pop("review_context", None)
                st.session_state["advance_review"] = True
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Reload saved judgment", key="reload_review"):
            st.session_state.pop("review_context", None)
            st.rerun()
        if own:
            with st.expander("Reveal comparison for my saved judgment"):
                reference = store.one(
                    "SELECT label_json FROM reference_labels WHERE dataset_id=? AND case_id=?",
                    (run["dataset_id"], selected),
                )
                st.write(
                    "Provisional eventual-owner label:",
                    json.loads(reference["label_json"])["eventual_owner_queue"],
                )
                comparisons = []
                for model in models:
                    rows = [
                        row
                        for row in prediction_rows(store, run["run_id"], model)
                        if row["case_id"] == selected
                    ]
                    if rows:
                        comparisons.append(
                            {
                                field: rows[0][field]
                                for field in ["predicted_queue", "confidence", "action"]
                            }
                            | {"model": model}
                        )
                if comparisons:
                    st.dataframe(pd.DataFrame(comparisons), hide_index=True, width="stretch")
                else:
                    st.caption(
                        "This case is outside the selected run. Choose a development run "
                        "to inspect development predictions."
                    )
    else:
        st.success("You have completed this queue. Include completed cases to revise a judgment.")
    with st.expander("Reviewer agreement and exports"):
        results = agreement(store, run["dataset_id"])
        st.caption(results["limitation"])
        if results["pairs"]:
            st.dataframe(pd.DataFrame(results["pairs"]), hide_index=True, width="stretch")
        else:
            st.info("Agreement becomes available when two reviewers annotate overlapping cases.")
        st.download_button(
            "Download latest reviews",
            export_reviews(store, run["dataset_id"]),
            file_name="rightdesk-latest-annotations.jsonl",
            mime="application/x-ndjson",
        )
        st.download_button(
            "Download annotation history",
            export_reviews(store, run["dataset_id"], history=True),
            file_name="rightdesk-review-history.jsonl",
            mime="application/x-ndjson",
        )


def main():
    st.set_page_config(page_title="RightDesk", page_icon="📬", layout="wide")
    st.title("RightDesk")
    st.caption("Support intake research · experiment comparison & human review")
    database = Path(
        os.environ.get("RIGHTDESK_DATABASE", ROOT / "artifacts/workbench/rightdesk.sqlite3")
    )
    if not database.exists():
        st.info(
            "Prepare the local workbench with `rightdesk store bootstrap`, then reload this page."
        )
        return
    store = Store(database)
    try:
        runs = store.list_runs()
    except ValueError as exc:
        st.error(str(exc))
        return
    st.sidebar.title("Workbench")
    st.sidebar.caption("Saved experiments · no model inference on page changes")
    if not runs:
        st.info(
            "The experiment catalog is empty. "
            "Register a completed run with `rightdesk store register-run`."
        )
        return
    selected = st.sidebar.selectbox(
        "Experiment",
        [row["run_id"] for row in runs],
        format_func=lambda key: next(
            f"{row['kind']} · {row['cases']} tickets · {key[-6:]}"
            for row in runs
            if row["run_id"] == key
        ),
    )
    run = store.run(selected)
    models = {row["model_id"]: row for row in store.models(selected)}
    st.sidebar.write("Split:", run["split"])
    st.sidebar.caption("Final-test data is excluded from this workbench.")
    comparison, tickets, reviews, primary, shadow, final = st.tabs(
        [
            "Experiments",
            "Ticket explorer",
            "Human review",
            "Primary helpdesk",
            "Shadow observer",
            "Final results",
        ]
    )
    with comparison:
        overview(store, run, models)
    with tickets:
        explorer(store, run, models)
    with reviews:
        human_review(store, run, models)
    with primary:
        render_simulator(ROOT)
    with shadow:
        render_shadow(ROOT)
    with final:
        render_final(ROOT)


if __name__ == "__main__":
    main()
