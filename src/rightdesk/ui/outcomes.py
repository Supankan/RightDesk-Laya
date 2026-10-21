"""Saved shadow and final summaries. No predictors or final-test tickets are loaded."""

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from rightdesk.data.importers.abcd import checksum
from rightdesk.shadow.engine import PublicStream, ShadowStore


def render_shadow(root: Path):
    st.subheader("Independent shadow observer")
    st.caption("Read-only primary events → separate shadow database → proposals only")
    store = ShadowStore(
        Path(
            os.environ.get("RIGHTDESK_SHADOW_DATABASE", root / "artifacts/shadow/rightdesk.sqlite3")
        )
    )
    try:
        observers = store.observers()
        if not observers:
            st.info("Run `rightdesk shadow run SIMULATION_ID` in a terminal, then refresh.")
            return
        observer_id = st.selectbox("Shadow observer", [r["observer_id"] for r in observers])
        capture = store.capture(observer_id)
        observer = capture["observer"]
        binding = json.loads(observer["binding_json"])
        stream = PublicStream(Path(binding["primary_path"]), binding["simulation_id"])
        a, b, c = st.columns(3)
        a.metric("Observed events", len(capture["events"]))
        b.metric("Shadow proposals", len(capture["proposals"]))
        c.metric("Unobserved events", stream.backlog(observer["cursor"])["unobserved_events"])
        st.caption(f"Simulation: {binding['simulation_id']} · cursor {observer['cursor']}")
        st.info("Routing is disabled. Clarification proposals are experimental and never sent.")
        rows = []
        for row in capture["proposals"]:
            decision = json.loads(row["decision_json"])
            rows.append(
                {
                    "event": row["event_id"],
                    "case": row["case_id"],
                    "model": row["model_id"],
                    "action": decision["action"],
                    "question": decision["question_id"],
                    "reason": decision["reason_code"],
                    "failed": row["error_json"] is not None,
                    "latency_ms": row["latency_ms"],
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with st.expander("Observer provenance"):
            st.json(binding)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        st.error(f"Shadow summary unavailable: {exc}")


def render_final(root: Path):
    st.subheader("Frozen final evaluation")
    directory = Path(os.environ.get("RIGHTDESK_FINAL_DIR", root / "artifacts/final-test-v2"))
    if not (directory / "run.json").exists():
        st.info(
            "No completed final-test report is available. Partial runs are not shown as results."
        )
        return
    try:
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if run.get("purpose") != "frozen_final_test" or run.get("failed_cases") != 0:
            raise ValueError("Expected completed frozen final evaluation")
        for name, expected in run["artifacts"].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or checksum(path) != expected:
                raise ValueError(f"Final artifact changed: {name}")
        st.metric("Held-out test tickets", run["evaluation_cases"])
        st.caption(
            "Provisional eventual-owner labels. Test tickets are excluded from human review."
        )
        rows = [
            {
                "variant": name,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "brier": m["brier_score"],
                "ece": m["calibration_error"],
                "native_coverage": m["operating_point"]["automation_coverage"],
                "enabled_coverage": m["enabled_policy"]["automation_coverage"],
            }
            for name, m in run["metrics"].items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.info(
            "Final results do not change routing gates. Human question usefulness is unmeasured."
        )
        st.download_button(
            "Download final report",
            (directory / "report.md").read_text(encoding="utf-8"),
            file_name="rightdesk-final-report.md",
            mime="text/markdown",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Final report unavailable: {exc}")
