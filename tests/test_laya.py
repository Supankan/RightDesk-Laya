import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from rightdesk.config import load_workflow
from rightdesk.models.laya_model import LayaConfig, LayaError, LayaModel
from rightdesk.schemas import TicketInput

ROOT = Path(__file__).resolve().parents[1]


class FakeAgent:
    device = "cpu"
    tok = SimpleNamespace(encode=lambda text, **kwargs: text.split())

    def __init__(self):
        self.calls = []
        self.change = lambda rows: rows

    def predict_batch(self, states, questions, **kwargs):
        self.calls.append((states, copy.deepcopy(questions), kwargs))
        queues = list(questions["owner_queue"]["criteria"])
        return self.change(
            [
                {
                    "answers": {
                        "owner_queue": {
                            "choice": queues[0],
                            "probabilities": dict.fromkeys(queues, 0.1667),
                            "confidence": 0.99,
                            "action": {"act_probability": 0.99},
                        }
                    }
                }
                for _ in states
            ]
        )


@pytest.fixture
def model():
    workflow = load_workflow(ROOT / "configs/workflows/retail_support.yaml")
    model = LayaModel(workflow, ROOT / "configs/models/laya_v1.yaml")
    model.agent, model.fitted = FakeAgent(), True
    return model


def ticket(identifier, text="Please help me", language="en"):
    return TicketInput(
        case_id=identifier, conversation_id=identifier, text=text, language=language, channel="chat"
    )


def test_batched_alignment_rounding_and_no_fabricated_sufficiency(model):
    inputs = [ticket("one"), ticket("two", "refund please")]
    predictions = model.predict_batch(inputs)
    assert [p.case_id for p in predictions] == ["one", "two"]
    assert sum(predictions[0].queue_probabilities.values()) == pytest.approx(1)
    assert predictions[0].sufficient_information_probability is None
    assert predictions[0].calibration_version is None
    states, questions, kwargs = model.agent.calls[0]
    assert states == [entry.text for entry in inputs]
    assert set(questions) == {"owner_queue"}
    assert kwargs["batch_size"] == 4
    assert model.last_raw[0]["answers"]["owner_queue"]["probabilities"]["account"] == 0.1667


@pytest.mark.parametrize(
    "change",
    [
        lambda rows: [],
        lambda rows: [{"answers": {}}],
        lambda rows: [
            {"answers": {"owner_queue": {"choice": "account", "probabilities": {"account": 1}}}}
        ],
    ],
)
def test_malformed_responses_fail_closed(model, change):
    model.agent.change = change
    with pytest.raises(LayaError):
        model.predict_one(ticket("one"))


@pytest.mark.parametrize("value", [float("nan"), -0.1, 0.8, True, "0.1667"])
def test_invalid_probabilities_not_silently_repaired(model, value):
    def change(rows):
        rows[0]["answers"]["owner_queue"]["probabilities"]["account"] = value
        return rows

    model.agent.change = change
    with pytest.raises(LayaError):
        model.predict_one(ticket("one"))


def test_unsupported_or_long_inputs_are_explicit_failures(model):
    with pytest.raises(LayaError, match="English"):
        model.predict_one(ticket("one", language="fr"))
    with pytest.raises(LayaError, match="token budget"):
        model.predict_one(ticket("one", "word " * 313))
    assert not model.agent.calls


def test_unloaded_and_empty_batches(model):
    assert model.predict_batch([]) == []
    model.fitted = False
    with pytest.raises(LayaError, match="Load Laya"):
        model.predict_one(ticket("one"))


def test_revision_must_be_immutable(model):
    config = model.config.model_dump()
    config["revision"] = "main"
    with pytest.raises(ValueError):
        LayaConfig.model_validate(config)


def test_loaded_model_is_reused(model):
    previous = model.agent
    model.load()
    assert model.agent is previous
    model.predict_one(ticket("one"))
    model.predict_one(ticket("two"))
    assert len(previous.calls) == 2
