# RightDesk handoff

## Completion scope

All fourteen implementation phases are complete. The deliverable is a local,
reproducible research workbench, simulator and independent observer, with saved
experiments, a blind reviewer workflow, final evaluation and process-level checks.
It is not a deployed helpdesk, and no live customer system is connected.

| Phases | Delivered |
|---|---|
| 1–3 | Strict contracts, pinned ABCD import, provisional mapping, isolated partitions |
| 4–6 | Majority, native rules, TF-IDF/logistic and pinned local Laya comparison |
| 7 | Group-isolated calibration, threshold selection and independent gate |
| 8–10 | Experiment catalog, Parquet exports, dashboard, blind revisioned human review |
| 11–12 | Bounded predefined questions and independent persistent primary simulator |
| 13 | Read-only shadow worker, atomic cursor, process restart and failure handling |
| 14 | Frozen held-out evaluation, end-to-end run, CI, package and handoff |

## Run the existing workspace

From the RightDesk directory:

```powershell
uv sync --locked --extra baseline --extra ml --extra data --extra ui
uv run --no-sync streamlit run apps/dashboard.py
```

The dashboard binds to loopback port 8501. It contains Experiments, Ticket explorer,
Human review, Primary helpdesk, Shadow observer, and Final results. Long inference
jobs run explicitly in the CLI. Model and data downloads never start on page refresh.
The reviewer ID is an annotation label, not authentication; keep this local.

Override catalog/simulator/shadow/report locations with `RIGHTDESK_DATABASE`,
`RIGHTDESK_SIM_DATABASE`, `RIGHTDESK_SHADOW_DATABASE`, `RIGHTDESK_FINAL_DIR`.

## Rebuild on a fresh checkout

The repository contains code, configuration, tests, lockfile and written results.
Datasets, local model weights, trained artifacts and SQLite databases are ignored.
The original manifest hashes document this workspace; rebuilding creates new
timestamps/run IDs and measured timings. Review the commands in the linked guides:

1. Install the locked extras above.
2. `rightdesk data download-abcd`, `rightdesk data import-abcd`, and
   `rightdesk data prepare-abcd` reproduce pinned preparation and partitions.
3. `rightdesk benchmark-baselines` fits the training-only baselines.
4. `rightdesk download-laya` explicitly downloads the pinned checkpoint;
   `rightdesk benchmark-laya` reuses baselines and runs the development comparison.
5. `rightdesk predict-calibration`, then `rightdesk calibrate-routing` freeze policies.
6. `rightdesk store bootstrap` prepares the development/calibration workbench.
7. `rightdesk simulate demo --output artifacts/simulator/new-demo` creates the primary stream.
8. `rightdesk shadow run SIMULATION_ID` observes it independently.
9. `python scripts/run_e2e.py artifacts/new-end-to-end` exercises real models across restarts.

Use `rightdesk COMMAND --help` for flags/output locations. The README and guides
include the defaults and ordering. Load joblib artifacts only from trusted local runs.
The pinned model occupies roughly 843 MB plus its Python runtime; CPU inference
over the complete dataset takes several minutes. No external model API is used.

## Final-test protocol and audit

```powershell
rightdesk final-evaluation freeze artifacts/my-final-protocol.json
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
rightdesk final-evaluation run artifacts/my-final-protocol.json artifacts/my-final-run
```

These commands are for an independently frozen replication, not for further tuning
on the already consumed test. The protocol pins development artifacts, calibration,
dataset manifest, adapter, source code, runtime, variants, seed and 500 bootstrap
replicates. Models are not fitted and thresholds/gates do not change. The test
records are opened only after the pins are checked. Review data excludes test cases.
Native model coverage, frozen candidate thresholds and enabled policy coverage
have separate fields. Test metrics never enable a failed calibration gate.

Local completed report: `artifacts/final-test-v2/report.md` and `run.json`, using
`artifacts/final-protocol-v4.json`. An initial attempt was interrupted after 604
Laya predictions when a `rules_raw`/`rules_native` artifact lookup mismatch was
found. No final-test metrics were inspected. The lookup was corrected, a preflight
variant check added, and the frozen protocol reissued before the complete rerun.
Models, calibrators, thresholds and statistical choices were unchanged. The partial
attempt and original protocols are retained for audit rather than overwritten.

Local process demonstration: `artifacts/end-to-end-v2/run.json`. It verifies
primary-state equality, complete event observation and zero proposals on a third
process restart. Its 14 synthetic lifecycle cases are functional checks, not
representative accuracy or customer-benefit evidence. The first integration attempt
exercised fail-closed handling and exposed a missing lazy model-load call; it has
no completed success manifest. The corrected run requires zero model failures.

## Verification

```powershell
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests apps scripts
uv run --no-sync ruff format --check src tests apps scripts
uv lock --check --offline
uv build --offline
```

The fixture suite uses temporary data and fake SDK outputs. Real inference and
the process demonstration are explicit operations. GitHub Actions runs the fixture
suite and Ruff without downloading datasets or weights. A workflow file does not
mean a hosted run has already succeeded; inspect the GitHub run after pushing.

Completion checks: **175 tests passed**, Ruff lint and formatting passed, the
97-package lockfile checked offline, and both wheel and source distributions built.
Distribution contents were inspected to exclude local data, weights and environments.

## Scientific and product limits

- Source eventual-owner labels are provisional and may require later context.
- The prepared 50-case human review queue has no real judgments yet. Do not replace
  missing sufficiency/usefulness labels with source labels or synthetic answers.
- No Phase 7 candidate passed the complete gate. Routing stays disabled.
- The new question catalog and follow-up context are exploratory and unvalidated.
- The primary system, service times and replies in the demo are simulated.
- Laya is a zero-shot checkpoint; TF-IDF is supervised on the training partition.
  This compares practical configured systems, not equal training budgets.
- Final-test curves and intervals are descriptive. Do not select a new threshold
  from them or reuse the test for iterative tuning.

Further human annotation, representative operational data and prospective validation
would constitute a new research iteration. They are not fabricated completion claims.
