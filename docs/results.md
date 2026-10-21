# RightDesk results and completion

All fourteen implementation phases are complete as a local research prototype.
The held-out test has **1,004 cases**, with zero failed inference cases. The models,
calibrators and gates were frozen before scoring; no refitting or retuning occurred.
Targets are provisional scenario-derived eventual owners, not reviewed intake judgments.

## Held-out comparison

| Variant | Accuracy | Macro F1 | Brier | ECE | Native coverage | Enabled coverage |
|---|---:|---:|---:|---:|---:|---:|
| majority_raw | 25.10% | 0.0669 | 0.8143 | 0.0141 | 100.00% | 0.00% |
| majority_temperature | 25.10% | 0.0669 | 0.8147 | 0.0211 | 100.00% | 0.00% |
| rules_native | 36.25% | 0.5095 | N/A | N/A | 40.74% | 0.00% |
| tfidf_logistic_raw | 81.47% | 0.8250 | 0.2676 | 0.1358 | 100.00% | 0.00% |
| tfidf_logistic_temperature | 81.47% | 0.8250 | 0.2373 | 0.0411 | 100.00% | 0.00% |
| laya_raw | 47.01% | 0.4661 | 0.6759 | 0.2024 | 100.00% | 0.00% |
| laya_temperature | 47.01% | 0.4661 | 0.6277 | 0.1157 | 100.00% | 0.00% |

TF-IDF/logistic outperformed the configured zero-shot Laya model on these labels.
This benchmark provides **no evidence of a Laya advantage**. The comparison is between
practical systems: TF-IDF is trained on 5,792 examples and Laya is used zero-shot.
It does not establish an inherent limit of Laya or compare equal training budgets.

Temperature scaling improved TF-IDF Brier from 0.2676 to 0.2373 and ECE from
0.1358 to 0.0411. Laya Brier improved from 0.6759 to 0.6277 and ECE from 0.2024
to 0.1157. Argmax predictions and accuracy stay the same under temperature scaling.

All enabled policy coverage is **0%**. No candidate passed the earlier full calibration
gate, and final-test observations do not change that decision. Native rule coverage
is a separate diagnostic; it is not enabled automation.

The 500-replicate normalized-message cluster bootstrap gives descriptive pointwise
intervals, not safety guarantees. Exact intervals, counts, candidate risks, revisions
and provenance hashes are preserved in [results.json](results.json). Full local
confusion matrices, per-queue metrics, reliability bins and curves remain under
`artifacts/final-test-v2/`. Timing mixes serial baseline calls and amortized Laya
batches, so it is not a like-for-like online latency comparison.

## Operational demonstration

The real-model process demonstration completed **14 synthetic tickets, 81 events and
84 proposals** (21 arrived-input checkpoints × 4 models). Every model had zero failures.
The primary state remained unchanged. Independent worker processes resumed correctly,
A third process produced zero new proposals. All 14 lifecycle assertions passed.
Laya proposed one question in this demonstration; no shadow question was sent.

A separate **50-ticket development replay** produced 172 observed events and 200
proposals. Laya proposed two questions; TF-IDF proposed one. There were no model
failures or enabled routes, and both observers caught up fully with idempotent restart.
Agreement with the incumbent is descriptive agreement, not correctness or benefit.

## What remains unmeasured

The 50-case blind reviewer queue has no genuine human annotations. Intake sufficiency,
acceptable queues, question usefulness, inter-reviewer agreement and customer benefit
remain unmeasured. Synthetic lifecycle success does not replace those judgments.
The local prototype uses simulated service times and has no live helpdesk integration.

## Reproduction and audit

See [handoff](handoff.md) for setup, phase inventory, checks, exact commands and the
audit record of corrected integration attempts. Source data and model weights stay
local; this report contains aggregate results only. The final test is consumed and
must not be reused to select a model or tune thresholds.
