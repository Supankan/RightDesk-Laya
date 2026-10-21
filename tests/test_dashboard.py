"""Streamlit interactions operate on stored data and require explicit human judgments."""

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest
from test_workbench import submission
from test_workbench import workbench as source_workbench

from rightdesk.data.importers.abcd import checksum
from rightdesk.review import QueueRequest, create_queue, submit_review
from rightdesk.simulation.engine import Simulator
from rightdesk.storage import Store

workbench = source_workbench
ROOT = Path(__file__).resolve().parents[1]


def dashboard(monkeypatch, path):
    monkeypatch.setenv("RIGHTDESK_DATABASE", str(path))
    monkeypatch.setenv("RIGHTDESK_SIM_DATABASE", str(path.parent / "simulation.sqlite3"))
    monkeypatch.setenv("RIGHTDESK_SHADOW_DATABASE", str(path.parent / "shadow.sqlite3"))
    monkeypatch.setenv("RIGHTDESK_FINAL_DIR", str(path.parent / "final"))
    return AppTest.from_file(str(ROOT / "apps/dashboard.py"), default_timeout=30).run()


def element(elements, label):
    return next(value for value in elements if value.label == label)


def test_missing_and_empty_catalog_are_actionable(tmp_path, monkeypatch):
    missing = dashboard(monkeypatch, tmp_path / "missing.sqlite3")
    assert not missing.exception
    assert "bootstrap" in missing.info[0].value
    store = Store(tmp_path / "empty.sqlite3")
    store.initialize()
    empty = dashboard(monkeypatch, store.path)
    assert not empty.exception
    assert "empty" in empty.info[0].value


def test_comparison_preserves_failed_gate_and_explorer_filters(workbench, monkeypatch):
    app = dashboard(monkeypatch, workbench["store"].path)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Experiments",
        "Ticket explorer",
        "Human review",
        "Primary helpdesk",
        "Shadow observer",
        "Final results",
    ]
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Policies passing gate"] == "0"
    frame = app.dataframe[0].value
    assert frame.iloc[0]["coverage"] == "0.00%"
    assert app.dataframe[1].value.iloc[0]["candidate_coverage"] == "100.00%"
    app.sidebar.selectbox[0].set_value(workbench["run_id"]).run()
    assert not app.exception
    element(app.selectbox, "Prediction outcome").set_value("Incorrect").run()
    element(app.text_input, "Search ticket text or case ID").set_value("NO SUCH MESSAGE").run()
    assert not app.exception
    assert any("No tickets match" in value.value for value in app.info)


def start_review(workbench, monkeypatch):
    create_queue(
        workbench["store"], QueueRequest(title="Blind", dataset_id=workbench["dataset_id"], limit=3)
    )
    app = dashboard(monkeypatch, workbench["store"].path)
    element(app.text_input, "Reviewer ID").set_value("alice").run()
    assert not app.exception
    return app


def test_review_requires_judgment_then_saves_and_advances(workbench, monkeypatch):
    app = start_review(workbench, monkeypatch)
    original = element(app.selectbox, "Review case").value
    assert not any("Reveal comparison" in value.label for value in app.expander)
    element(app.button, "Save judgment").click().run()
    assert app.error
    assert not workbench["store"].latest_reviews(workbench["dataset_id"])
    element(app.radio, "Is there enough information to route?").set_value("No")
    element(app.selectbox, "Queue judgment").set_value("No queue is acceptable yet")
    element(app.button, "Save judgment").click().run()
    assert not app.exception
    assert not app.error
    reviews = workbench["store"].latest_reviews(workbench["dataset_id"])
    assert len(reviews) == 1
    import json

    label = json.loads(reviews[0]["label_json"])
    assert label["sufficient_information_at_intake"] is False
    assert label["acceptable_queues_at_intake"] == []
    assert label["useful_question_ids"] is None
    assert element(app.selectbox, "Review case").value != original
    element(app.checkbox, "Include my completed cases").check().run()
    element(app.selectbox, "Review case").set_value(original).run()
    assert any("Reveal comparison" in value.label for value in app.expander)


def test_stale_form_cannot_silently_overwrite_another_session(workbench, monkeypatch):
    app = start_review(workbench, monkeypatch)
    selected = element(app.selectbox, "Review case").value
    submit_review(
        workbench["store"],
        submission(workbench, case=selected, sufficient_information_at_intake=True),
    )
    element(app.radio, "Is there enough information to route?").set_value("No")
    element(app.button, "Save judgment").click().run()
    assert not app.exception
    assert any("another session" in error.value for error in app.error)
    assert workbench["store"].latest_reviews(workbench["dataset_id"])[0]["revision"] == 1


def test_primary_simulator_intake_reply_service_and_reopen(workbench, monkeypatch):
    app = dashboard(monkeypatch, workbench["store"].path)
    element(app.button, "Create simulation").click().run()
    assert not app.exception
    element(app.text_input, "New case ID").set_value("ui-case")
    element(app.text_area, "First customer message").set_value("charge refund")
    element(app.button, "Submit customer ticket").click().run()
    assert not app.exception and not app.error
    simulator = Simulator(workbench["store"].path.parent / "simulation.sqlite3")
    session = simulator.sessions()[0]["simulation_id"]
    assert simulator.ticket(session, "ui-case").status == "awaiting_customer"
    element(app.text_area, "Customer reply").set_value("This is a new charge.")
    element(app.button, "Send simulated reply").click().run()
    assert not app.exception and not app.error
    assert simulator.ticket(session, "ui-case").assigned_queue == "billing_subscriptions"
    element(app.number_input, "Advance by seconds").set_value(45)
    element(app.button, "Advance simulation clock").click().run()
    assert not app.exception and not app.error
    assert simulator.ticket(session, "ui-case").status == "resolved"
    element(app.button, "Apply simulated agent action").click().run()
    assert not app.exception and not app.error
    assert simulator.ticket(session, "ui-case").status == "awaiting_agent"
    assert simulator.verify(session)["verified"]


def test_simulator_clock_form_detects_another_session(workbench, monkeypatch):
    app = dashboard(monkeypatch, workbench["store"].path)
    element(app.button, "Create simulation").click().run()
    simulator = Simulator(workbench["store"].path.parent / "simulation.sqlite3")
    session = simulator.sessions()[0]["simulation_id"]
    simulator.advance(session, 5, "outside")
    element(app.button, "Advance simulation clock").click().run()
    assert not app.exception
    assert any("clock changed" in error.value for error in app.error)
    assert simulator.sessions()[0]["clock"] == 5


def test_final_summary_is_saved_only_and_rejects_tampered_artifacts(workbench, monkeypatch):
    directory = workbench["store"].path.parent / "final"
    directory.mkdir()
    report = directory / "report.md"
    report.write_text("# Fixture final report", encoding="utf-8")
    payload = {
        "purpose": "frozen_final_test",
        "failed_cases": 0,
        "evaluation_cases": 2,
        "artifacts": {"report.md": checksum(report)},
        "metrics": {
            "fixture": {
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "brier_score": 0.8,
                "calibration_error": 0.1,
                "operating_point": {"automation_coverage": 1.0},
                "enabled_policy": {"automation_coverage": 0.0},
            }
        },
    }
    (directory / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    app = dashboard(monkeypatch, workbench["store"].path)
    assert not app.exception
    assert any(
        metric.label == "Held-out test tickets" and metric.value == "2" for metric in app.metric
    )
    report.write_text("modified", encoding="utf-8")
    app.run()
    assert not app.exception
    assert any("Final artifact changed" in value.value for value in app.error)
