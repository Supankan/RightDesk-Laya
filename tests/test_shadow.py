"""Independent observer isolation, event-prefix features, recovery and concurrency."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from rightdesk.policy.clarification import ClarificationConfig, IntakePolicy
from rightdesk.policy.routing import RoutingSettings
from rightdesk.schemas import ModelPrediction, TicketInput, WorkflowConfig
from rightdesk.shadow.engine import PublicStream, ShadowStore, ShadowWorker, export_shadow
from rightdesk.simulation.engine import Simulator
from rightdesk.storage import encode

ROOT = Path(__file__).resolve().parents[1]


class Predictor:
    names = ("test",)
    metadata = {"test": "fixed-v1"}

    def __init__(self, snapshot, error=None, barrier=None):
        self.workflow = WorkflowConfig.model_validate(snapshot["workflow"])
        self.workflows = {"test": self.workflow}
        self.policy = IntakePolicy(
            self.workflow,
            RoutingSettings(model_revision="fixed-v1"),
            ClarificationConfig.model_validate(snapshot["clarification"]),
        )
        self.seen, self.error, self.barrier = [], error, barrier

    def predict(self, name, ticket, asked):
        self.seen.append((ticket.model_dump(), list(asked)))
        if self.barrier and len(self.seen) == 1:
            self.barrier.wait(timeout=10)
        if self.error:
            raise self.error("Intentional model failure")
        prediction = ModelPrediction.for_workflow(
            {
                "case_id": ticket.case_id,
                "workflow_id": self.workflow.workflow_id,
                "workflow_version": self.workflow.version,
                "model_revision": "fixed-v1",
                "queue_probabilities": {
                    q: {"billing_subscriptions": 0.5, "returns_refunds": 0.5}.get(q, 0.0)
                    for q in self.workflow.queue_ids
                },
            },
            self.workflow,
        )
        return prediction, self.policy.decide(ticket, prediction, asked)

    def failure(self, name, ticket):
        return self.policy.decide(ticket, None, [])


@pytest.fixture
def environment(tmp_path):
    primary = Simulator(tmp_path / "primary.sqlite3")
    primary.initialize()
    key = primary.create(
        ROOT / "configs/simulator/retail_v1.yaml", "test", {"private_future_plan": "DO NOT READ"}
    )
    stream = PublicStream(primary.path, key)
    store = ShadowStore(tmp_path / "shadow.sqlite3")
    store.initialize()
    return primary, key, stream, store


def ticket(key="one", text="I have a charge and refund issue."):
    return TicketInput(case_id=key, conversation_id=key, text=text, channel="chat", language="en")


def primary_snapshot(primary, key):
    session, states, events = primary.capture(key)
    return encode([session, [s.model_dump() for s in states], [e.model_dump() for e in events]])


def test_readonly_handle_denies_writes_and_private_projection_reads(environment):
    primary, key, stream, store = environment
    assert "private_future_plan" not in encode(stream.snapshot())
    with stream.connection() as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("UPDATE simulations SET clock=900")
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("SELECT state_json FROM tickets")
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("PRAGMA query_only=OFF")
    with pytest.raises(ValueError, match="different"):
        ShadowWorker(stream, ShadowStore(primary.path), Predictor(stream.snapshot()))
    assert primary.sessions()[0]["clock"] == 0


def test_prefix_only_features_restart_and_no_primary_mutation(environment, tmp_path):
    primary, key, stream, store = environment
    initial = primary.intake(key, ticket(), "create")
    primary.reply(key, "one", "FUTURE CUSTOMER SENTINEL", initial["version"], "reply")
    before = primary_snapshot(primary, key)
    predictor = Predictor(stream.snapshot())
    worker = ShadowWorker(stream, store, predictor)
    first = worker.run(max_events=1)
    assert first["processed_events"] == 1 and first["unobserved_events"] > 0
    assert predictor.seen[0][0]["text"] == ticket().text
    assert predictor.seen[0][1] == []
    assert "FUTURE" not in encode(predictor.seen)
    resumed_predictor = Predictor(stream.snapshot())
    resumed = ShadowWorker(stream, ShadowStore(store.path), resumed_predictor)
    assert resumed.observer_id == worker.observer_id
    result = resumed.run()
    assert result["unobserved_events"] == 0
    text, asked = resumed_predictor.seen[-1]
    assert "FUTURE CUSTOMER SENTINEL" in text["text"]
    assert "Support question:" in text["text"]
    assert "incumbent" not in text["text"] and "DO NOT READ" not in text["text"]
    assert asked == ["charge_or_refund"]
    captured = store.capture(worker.observer_id)
    decisions = [json.loads(row["decision_json"]) for row in captured["proposals"]]
    assert [d["action"] for d in decisions] == ["clarify", "review"]
    assert resumed.run()["new_proposals"] == 0
    assert primary_snapshot(primary, key) == before
    report_path = export_shadow(store, worker.observer_id, tmp_path / "export")
    report = json.loads(report_path.read_text())
    assert report["models"]["test"]["proposals"] == 2
    assert report["question_usefulness"] is None


def test_new_arrivals_after_catchup_are_processed(environment):
    primary, key, stream, store = environment
    worker = ShadowWorker(stream, store, Predictor(stream.snapshot()))
    assert worker.run()["new_proposals"] == 0
    primary.intake(key, ticket(), "first")
    assert worker.run()["new_proposals"] == 1
    primary.intake(key, ticket("two"), "second")
    assert worker.run()["new_proposals"] == 1
    assert len(store.capture(worker.observer_id)["cases"]) == 2


def test_export_pairs_native_queue_prediction_with_primary_decision(environment, tmp_path):
    primary, key, stream, store = environment
    primary.intake(key, ticket(text="Please reset my password."), "create")
    worker = ShadowWorker(stream, store, Predictor(stream.snapshot()))
    worker.run()
    report = json.loads(export_shadow(store, worker.observer_id, tmp_path / "export").read_text())
    assert report["models"]["test"]["queue_pairs"] == 1
    assert report["models"]["test"]["queue_agreements"] == 0


def test_frozen_adapter_loads_laya_before_first_prediction_and_reuses_instance(environment):
    pytest.importorskip("scipy")
    from rightdesk.shadow.models import FrozenPredictor

    _, _, stream, _ = environment
    fake = Predictor(stream.snapshot())

    class LocalModel:
        loaded = False
        loads = 0

        def load(self):
            if not self.loaded:
                self.loads += 1
                self.loaded = True

        def predict_one(self, value):
            assert self.loaded
            return fake.predict("test", value, [])[0]

    adapter = FrozenPredictor.__new__(FrozenPredictor)
    adapter.models = {"laya": LocalModel()}
    adapter.calibrators = {}
    adapter.workflows = {"laya": fake.workflow}
    adapter.policies = {"laya": fake.policy}
    adapter.source_workflow = fake.workflow
    for _ in range(2):
        prediction, decision = adapter.predict("laya", ticket(), [])
        assert prediction is not None and decision.action == "clarify"
    assert adapter.models["laya"].loads == 1


def test_model_failure_persists_review_without_fake_prediction(environment):
    primary, key, stream, store = environment
    primary.intake(key, ticket(), "create")
    worker = ShadowWorker(stream, store, Predictor(stream.snapshot(), error=RuntimeError))
    worker.run()
    row = store.capture(worker.observer_id)["proposals"][0]
    assert row["prediction_json"] is None
    assert json.loads(row["decision_json"])["reason_code"] == "model_error"
    assert json.loads(row["error_json"])["type"] == "RuntimeError"
    assert worker.run()["new_proposals"] == 0


def test_interrupt_before_commit_replays_exactly_once(environment):
    primary, key, stream, store = environment
    primary.intake(key, ticket(), "create")
    worker = ShadowWorker(stream, store, Predictor(stream.snapshot(), error=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        worker.run()
    capture = store.capture(worker.observer_id)
    assert capture["observer"]["cursor"] == 0 and not capture["proposals"]
    recovered = ShadowWorker(stream, store, Predictor(stream.snapshot()))
    assert recovered.run()["new_proposals"] == 1
    assert len(store.capture(worker.observer_id)["proposals"]) == 1


def test_concurrent_workers_commit_one_result_per_input(environment):
    primary, key, stream, store = environment
    primary.intake(key, ticket(), "create")
    barrier = Barrier(2)
    workers = [
        ShadowWorker(stream, store, Predictor(stream.snapshot(), barrier=barrier)) for _ in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda worker: worker.run(), workers))
    assert sum(r["new_proposals"] for r in results) == 1
    assert len(store.capture(workers[0].observer_id)["proposals"]) == 1


def test_resume_rejects_changed_event_prefix(environment, monkeypatch):
    primary, key, stream, store = environment
    primary.intake(key, ticket(), "create")
    predictor = Predictor(stream.snapshot())
    worker = ShadowWorker(stream, store, predictor)
    worker.run()
    original = stream.events

    def tampered(after=0, limit=500):
        events = original(after, limit)
        if events:
            events[0] = events[0].model_copy(update={"command_id": "mutated"})
        return events

    monkeypatch.setattr(stream, "events", tampered)
    with pytest.raises(ValueError, match="prefix changed"):
        ShadowWorker(stream, store, predictor)
