# Architecture

The CLI and Streamlit dashboard call shared application services. Model
adapters produce predictions; application policy produces proposed actions.

```text
TicketInput + WorkflowConfig
             |
      Model adapter
             |
      ModelPrediction
             |
     Application policy
             |
      PolicyDecision ----> Shadow log and evaluation
                                  ^
                            ReviewedLabel
```

Long experiment runs will use the CLI and write results under `artifacts/`.
The dashboard will read those results and capture reviewer annotations in
SQLite. Filtering results should not trigger model inference again.

Start with rules, a conventional classifier, and Laya behind one interface.
Clarification selection requires a separate evaluation. Shadow operation
records proposals while the incumbent process continues to handle tickets.

Phase 7 implements probability calibration and deterministic route/review proposals
in `evaluation/calibration*.py` and `policy/routing.py`. Generated policy snapshots
bind model revisions, calibration versions, and frozen thresholds. Validation
gates can leave a policy in review-only mode. These proposals never execute ticket
routing. See [calibration and routing](calibration-routing.md).

Phases 8-10 implement the catalog in `storage.py`, ingestion/query operations in
`services.py`, Parquet snapshots in `exports.py`, and human review in `review.py`.
The Streamlit dashboard and workbench CLI share these services. Dataset-version
keys isolate cases and labels; run fingerprints bind saved predictions to historical
configuration and metrics. Human annotations have independent append-only revisions
and do not change source references or model artifacts. See [workbench](workbench.md).

Phase 11 composes a predefined question selector with the existing route gate in
`policy/clarification.py`. Offline question proposals and human usefulness metrics
are separate from routing calibration and source eventual-owner labels.

Phase 12 implements the independent primary system in `simulation/`: fixed incumbent
rules, ticket state transitions, customer and agent commands, a virtual clock,
idempotency, and an append-only event stream. Its SQLite database is separate from
the experiment/reviewer catalog. Events expose only actions/messages that have
happened; future scripts remain in the scenario driver. The dashboard and CLI
share this simulator API. See [clarification and primary helpdesk](clarification-simulator.md).

Phase 13 implements `shadow/`: a SQLite read-only public-event adapter, an independent
proposal database, prefix-only message projection, model/policy snapshots, atomic
event/cursor commits, restart verification and optimistic concurrent-worker protection.
It never calls primary mutation methods. Dashboard monitoring does not load models.
See [shadow worker](shadow.md).

Phase 14 freezes development models, calibrators, thresholds, evaluation code and
runtime versions before final-test scoring. Native choices, candidate thresholds and
enabled policy are reported separately. `scripts/run_e2e.py` exercises real models in
separate CLI processes and verifies source isolation and restart idempotence.
See [handoff](handoff.md) and [results](results.md).
