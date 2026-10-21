"""Final-test controls must fail before any test data is opened."""

import json
from pathlib import Path

import pytest

pytest.importorskip("scipy")
from rightdesk.evaluation import final


def test_protocol_rejects_changed_inputs_before_test_access(tmp_path, monkeypatch):
    paths = {}
    for key in ("source", "calibration", "manifest", "adapter"):
        path = tmp_path / key
        if key in ("source", "calibration"):
            path.mkdir()
            (path / "run.json").write_text("{}")
        else:
            path.write_text("{}")
        paths[key] = str(path)
    payload = {
        "purpose": "frozen_final_test",
        "variants": [f"{n}_{k}" for n, k in final.VARIANTS],
        **paths,
        "input_sha256": dict.fromkeys(paths, "incorrect"),
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(payload))
    monkeypatch.setattr(final, "load_partition", lambda *args: pytest.fail("Test opened too soon"))
    with pytest.raises(ValueError, match="Frozen input changed"):
        final.final_evaluate(protocol, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_final_output_is_never_overwritten(tmp_path):
    with pytest.raises(ValueError, match="never overwrites"):
        final.final_evaluate(Path("missing"), tmp_path)


def test_existing_protocol_cannot_be_replaced(tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        final.freeze_protocol(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)
