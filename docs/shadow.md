# Independent shadow worker

## Data flow

Primary simulator SQLite → read-only `PublicStream` → prefix projection → frozen
model → saved temperature calibrator → disabled routing gate / bounded question
selector → independent shadow SQLite. Primary tickets continue under the incumbent.

`PublicStream` uses `mode=ro`, `query_only=ON` and an SQL authorizer. It can read
public events and the simulation configuration; reads of primary ticket projections
and commands are denied. Scenario plans, expected outcomes, source references and
human reviews are never queried. The observer has no primary write connection.
This is application isolation for a local prototype, not a security sandbox against
a malicious local process that can open the underlying files itself.

The input at `ticket_created` is exactly the original customer text. At
`customer_replied`, it contains only the event prefix: original text plus arrived
customer messages and delivered support-question text, separated by explicit markers.
The current primary assignment, agent actions, primary decision and future events
never enter model text. Models see text, not case IDs. The worker consumes all
events for ordering/budget tracking but infers only at those two input events.

## Frozen models and workflow bridge

All three baselines are loaded from integrity-checked development artifacts without
fitting. Laya loads the local pinned checkpoint once per process, on first use.
Temperature calibration is applied under the original v1 workflow identity first.

The simulator uses workflow v2 with the same queue IDs and descriptions and three
predefined questions. The adapter checks that queue definitions match exactly,
then records the v1→v2 bridge explicitly. Calibration on initial v1 messages does
not validate follow-up dialogue or certify v2. Live shadow routing is always
disabled. Predictions remain visible for comparison; questions are experimental.

Each model may propose at most one question per case. Both actual primary questions
and that model's earlier proposals count against eligibility. A proposal is never
sent, and its hypothetical answer is never invented or fed back into the stream.
After an actual customer reply, only the real primary dialogue is observed.

## Persistence and recovery

The independent schema contains `observers`, `observed_events`, `observed_cases`
and `proposals`. An observer ID binds the primary path/session, workflow snapshot,
model/policy hashes, input contract, code fingerprints and observer name.
Different configurations receive separate observers.

For each event, inference runs outside a write transaction. A short transaction
checks the previous cursor and atomically appends the event, all model proposals,
projected history and the next cursor. A competing winner forces a retry.
Duplicate computation is possible with concurrent workers; duplicate persisted
proposals are prevented by primary keys. Event/proposal rows are immutable.

Restart verifies the already observed public-event prefix before continuing.
A process crash before commit leaves the event uncommitted and safe to replay.
A model exception records a null prediction, error details and a review decision,
then advances the cursor. It is not automatically retried as if it had succeeded.
Use a new observer name for an intentional reprocessing experiment. Do not edit history.

## Commands

```powershell
uv run --no-sync rightdesk simulate list
uv run --no-sync rightdesk shadow run SIMULATION_ID
uv run --no-sync rightdesk shadow run SIMULATION_ID --follow --poll-seconds 2
uv run --no-sync rightdesk shadow run SIMULATION_ID --models rules,tfidf_logistic --name small-v1
uv run --no-sync rightdesk shadow list
uv run --no-sync rightdesk shadow export OBSERVER_ID artifacts/shadow/new-export
```

Default primary: `artifacts/simulator/rightdesk.sqlite3`.
Default observer: `artifacts/shadow/rightdesk.sqlite3`.
Override with `--primary` and `--database`. Identical paths are rejected before
initialization. `--max-events` bounds one drain; `--follow` repeats drains until
Ctrl+C. All waiting is in the CLI process, not the dashboard.

The Shadow observer tab reads saved proposals and current event backlog. Reload to
refresh; inference never runs from the page. `RIGHTDESK_SHADOW_DATABASE` selects a
different observer database. Initial loading is included in the first prediction's
wall time, so these per-event timings are not warm latency benchmarks.

Exports contain observed events, exact model inputs and their hashes, predictions,
decisions, errors, model/policy provenance and file checksums. The report pairs a
shadow proposal with the next incumbent decision at the same input checkpoint,
without crossing another customer input. Action agreement and native queue
agreement have separate denominators. Agreement is not correctness or evidence of
time saved. Human question usefulness remains null.
