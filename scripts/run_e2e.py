"""Explicit real-model lifecycle demonstration with process restarts and source isolation."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rightdesk.data.importers.abcd import checksum, write_json
from rightdesk.shadow.engine import PublicStream, ShadowStore, export_shadow
from rightdesk.simulation.engine import Simulator
from rightdesk.simulation.scenarios import run_demo
from rightdesk.storage import encode, utc_now

ROOT = Path(__file__).resolve().parents[1]


def snapshot(simulator, key):
    session, states, events = simulator.capture(key)
    return encode([session, [s.model_dump() for s in states], [e.model_dump() for e in events]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--models", default="majority,rules,tfidf_logistic,laya")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a fresh output directory")
    output.mkdir(parents=True)
    simulator = Simulator(output / "primary.sqlite3")
    simulator.initialize()
    primary_report = run_demo(
        simulator,
        ROOT / "configs/simulator/retail_v1.yaml",
        ROOT / "configs/simulator/demo_v1.yaml",
        output / "primary",
    )
    run = json.loads(Path(primary_report).with_name("run.json").read_text(encoding="utf-8"))
    key = run["simulation_id"]
    before = snapshot(simulator, key)
    database = output / "shadow.sqlite3"
    base = [
        sys.executable,
        "-m",
        "rightdesk.cli",
        "shadow",
        "run",
        key,
        "--primary",
        str(simulator.path),
        "--database",
        str(database),
        "--models",
        args.models,
    ]
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    process_results = []
    for limit in (1, 10000, 10000):
        result = subprocess.run(
            [*base, "--max-events", str(limit)],
            cwd=ROOT,
            env=environment,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        print(result.stdout, flush=True)
        process_results.append(json.loads(result.stdout))
    if before != snapshot(simulator, key):
        raise ValueError("Observer mutated primary state")
    if process_results[-1]["new_proposals"] != 0 or process_results[-1]["unobserved_events"] != 0:
        raise ValueError("Restart/idempotency check failed")
    observer_id = process_results[-1]["observer_id"]
    store = ShadowStore(database)
    shadow_report = export_shadow(store, observer_id, output / "shadow")
    captured = store.capture(observer_id)
    report = json.loads(shadow_report.read_text(encoding="utf-8"))
    if any(model["failures"] or model["route"] for model in report["models"].values()):
        raise ValueError("Unexpected model failures or enabled routes in end-to-end run")
    if report["events"] != simulator.verify(key)["events"]:
        raise ValueError("Incomplete shadow observation")
    if PublicStream(simulator.path, key).backlog(captured["observer"]["cursor"])[
        "unobserved_events"
    ]:
        raise ValueError("Unexpected shadow backlog")
    write_json(
        output / "run.json",
        {
            "schema_version": 1,
            "phase": 14,
            "purpose": "end_to_end_process_demonstration",
            "completed_at": utc_now(),
            "simulation_id": key,
            "observer_id": observer_id,
            "models": args.models.split(","),
            "primary_unchanged": True,
            "restart_idempotent": True,
            "process_results": process_results,
            "primary_verification": simulator.verify(key),
            "shadow_models": report["models"],
            "scope": "synthetic functional checks; accuracy and operational benefit unmeasured",
            "artifacts": {
                p.relative_to(output).as_posix(): checksum(p)
                for folder in (output / "primary", output / "shadow")
                for p in sorted(folder.rglob("*"))
                if p.is_file()
            },
        },
    )
    print(output / "run.json", flush=True)


if __name__ == "__main__":
    main()
