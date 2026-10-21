"""Primary helpdesk controls backed by the same state machine as the CLI."""

import os
from collections import Counter
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from rightdesk.schemas import TicketInput
from rightdesk.simulation.engine import Simulator
from rightdesk.storage import encode


def render_simulator(root: Path):
    st.subheader("Primary helpdesk simulator")
    st.caption(
        "Local simulated operations · fixed incumbent rules · virtual time · no Laya execution"
    )
    database = Path(
        os.environ.get("RIGHTDESK_SIM_DATABASE", root / "artifacts/simulator/rightdesk.sqlite3")
    )
    simulator = Simulator(database)
    config = root / "configs/simulator/retail_v1.yaml"
    with st.expander("Create a simulation"):
        with st.form("new_simulation"):
            name = st.text_input("Simulation name", value="Manual helpdesk")
            create = st.form_submit_button("Create simulation")
        if create:
            try:
                simulator.initialize()
                identifier = simulator.create(config, name)
                st.session_state["simulation_select"] = identifier
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    sessions = simulator.sessions()
    if not sessions:
        st.info(
            "Create a simulation, or run `rightdesk simulate demo` "
            "for the built-in lifecycle examples."
        )
        return
    identifier = st.selectbox(
        "Simulation",
        [session["simulation_id"] for session in sessions],
        format_func=lambda key: next(
            session["name"] + " · " + key[-6:]
            for session in sessions
            if session["simulation_id"] == key
        ),
        key="simulation_select",
    )
    session = next(session for session in sessions if session["simulation_id"] == identifier)
    states = simulator.tickets(identifier)
    counts = Counter(state.status for state in states)
    a, b, c, d = st.columns(4)
    a.metric("Virtual seconds", session["clock"])
    b.metric("Primary tickets", len(states))
    c.metric("Waiting for customer", counts["awaiting_customer"])
    d.metric("Awaiting agent", counts["awaiting_agent"])
    st.caption(
        f"Assigned: {counts['assigned']} · resolved: {counts['resolved']}. "
        "Each assigned ticket has a fixed simulated service delay; "
        "this is not a staffing or SLA model."
    )
    clock_key = "sim_clock_token:" + identifier
    st.session_state.setdefault(clock_key, session["clock"])
    with st.form("advance_clock:" + identifier):
        seconds = st.number_input(
            "Advance by seconds", min_value=0, max_value=10000000, value=30, step=5
        )
        advance = st.form_submit_button("Advance simulation clock")
    if advance:
        try:
            simulator.advance(identifier, int(seconds), uuid4().hex, st.session_state[clock_key])
            st.session_state.pop(clock_key, None)
            for key in list(st.session_state):
                if key.startswith("sim_case_token:" + identifier + ":"):
                    st.session_state.pop(key, None)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    with st.expander("New customer ticket"):
        command_key = "sim_intake_token:" + identifier
        st.session_state.setdefault(command_key, uuid4().hex)
        token = st.session_state[command_key]
        with st.form("sim_intake:" + token):
            case_id = st.text_input("New case ID", value="manual-" + token[:8])
            text = st.text_area("First customer message")
            language = st.text_input("Ticket language", value="en")
            intake = st.form_submit_button("Submit customer ticket")
        if intake:
            try:
                simulator.intake(
                    identifier,
                    TicketInput(
                        case_id=case_id,
                        conversation_id=case_id,
                        text=text,
                        language=language,
                        channel="chat",
                    ),
                    token,
                )
                st.session_state.pop(command_key, None)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    if states:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "case_id": state.initial_ticket.case_id,
                        "status": state.status,
                        "queue": state.assigned_queue,
                        "questions": len(state.asked_question_ids),
                        "version": state.version,
                    }
                    for state in states
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        case_id = st.selectbox("Primary ticket", [state.initial_ticket.case_id for state in states])
        state = next(state for state in states if state.initial_ticket.case_id == case_id)
        st.text(state.initial_ticket.text)
        for message in state.messages[1:]:
            st.caption(f"Customer reply at virtual second {message.clock}")
            st.text(message.text)
        if state.pending_question_id:
            workflow = simulator.snapshot(identifier)["workflow"]
            question = next(
                q for q in workflow["questions"] if q["id"] == state.pending_question_id
            )
            st.info(question["text"])
            st.caption(f"Customer response deadline: virtual second {state.reply_deadline}.")
        version_key = "sim_case_token:" + identifier + ":" + case_id
        st.session_state.setdefault(version_key, state.version)
        if (
            state.status == "awaiting_customer"
            or state.status == "awaiting_agent"
            and state.asked_question_ids
        ):
            with st.form("sim_reply:" + identifier + ":" + case_id):
                reply = st.text_area("Customer reply")
                send = st.form_submit_button("Send simulated reply")
            if send:
                try:
                    simulator.reply(
                        identifier, case_id, reply, st.session_state[version_key], uuid4().hex
                    )
                    st.session_state.pop(version_key, None)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        actions = {
            "awaiting_agent": ["assign"],
            "assigned": ["reassign", "resolve"],
            "resolved": ["reopen"],
        }.get(state.status, [])
        if actions:
            with st.form("sim_agent:" + identifier + ":" + case_id):
                action = st.selectbox("Agent action", actions)
                queue = st.selectbox(
                    "Agent target queue",
                    [queue["id"] for queue in simulator.snapshot(identifier)["workflow"]["queues"]],
                )
                agent = st.text_input("Agent ID", value="operator")
                act = st.form_submit_button("Apply simulated agent action")
            if act:
                try:
                    simulator.agent(
                        identifier,
                        case_id,
                        action,
                        queue if action in ("assign", "reassign") else None,
                        agent,
                        st.session_state[version_key],
                        uuid4().hex,
                    )
                    st.session_state.pop(version_key, None)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    if st.button("Refresh simulator state"):
        for key in list(st.session_state):
            if key.startswith(("sim_clock_token:", "sim_case_token:")):
                st.session_state.pop(key, None)
        st.rerun()
    with st.expander("Public event stream"):
        events = simulator.events(identifier, limit=10000)
        st.caption(
            "Only events that have already happened appear here. "
            "Independent shadow consumption is Phase 13."
        )
        if events:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "event_id": e.event_id,
                            "clock": e.clock,
                            "case_id": e.case_id,
                            "kind": e.kind,
                            "actor": e.actor,
                        }
                        for e in events[-100:]
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.download_button(
                "Download public events",
                "".join(encode(e.model_dump()) + "\n" for e in events),
                file_name=identifier + "-events.jsonl",
                mime="application/x-ndjson",
            )
