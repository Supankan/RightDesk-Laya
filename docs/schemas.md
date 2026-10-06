# Implemented v1 contracts

Implemented in `src/rightdesk/schemas.py`. All contracts reject unknown fields
and implicit string/boolean coercion. Text is preserved; empty or whitespace-only
required text is rejected. Configuration IDs use lowercase snake_case.

| Contract | Fields |
|---|---|
| TicketInput | case_id, conversation_id, text, language, channel |
| WorkflowConfig | workflow_id, version, execution_mode, queues, questions, policy |
| ModelPrediction | case_id, workflow_id, workflow_version, queue_probabilities, predicted_queue, abstention_reason, sufficient_information_probability, suggested_question_id, model_revision, calibration_version |
| PolicyDecision | case_id, workflow_id, workflow_version, action, target_queue, question_id, reason_code, policy_version, execution_mode |
| ReviewedLabel | case_id, workflow_id, workflow_version, eventual_owner_queue, acceptable_queues_at_intake, sufficient_information_at_intake, useful_question_ids, review_status, reviewer_id |

## Validation rules

- Ticket IDs and nonempty text are required. Language may be unknown; channels
  are `chat`, `email`, or `form`.
- Queue and question IDs are unique within a workflow version. The prototype
  accepts 2-20 queues and a 0-1 question budget. These are application limits.
- When provided, queue probabilities cover the configured queues, lie in [0, 1], and sum to 1
  within an absolute tolerance of 1e-6 (no normalization is performed).
  NaN, infinity, booleans, and numeric strings are rejected.
  Sufficiency probability is null when unknown, otherwise in [0, 1].
- Phase 4 permits null queue_probabilities for non-probabilistic rules. A rule
  match supplies predicted_queue; abstention supplies no_rule_match or
  conflicting_rules. Abstention cannot carry a queue or probability distribution.
  A prediction must provide a distribution, a queue, or an abstention reason.
  If a queue and distribution are both supplied, the queue must maximize it.
  `selected_queue()` resolves probability-only records, breaking ties alphabetically.
  Rules do not fabricate one-hot confidence; no sufficiency probability is inferred.
- A suggested question refers to a configured question, or is null.
- `route` requires a target queue and no question. `clarify` requires a question
  and no target queue. `review` has neither destination field.
- Route reasons: explicit_rule or meets_routing_threshold. Clarify reason:
  missing_information. Review reasons: missing_information, low_confidence,
  unsupported_input, or model_error. Clarification requires the feature to be
  enabled, a configured question, and a nonzero question budget.
- Null reviewed labels mean unknown or unreviewed; an empty reviewed list has a
  different meaning. The eventual owner does not establish intake sufficiency.
- Labels stay outside inference inputs. Record workflow, model, calibration,
  and policy revisions so results are reproducible.
- Unreviewed records may contain a dataset-derived eventual owner, but all human
  intake judgments and reviewer_id must be null. Reviewed records require a
  reviewer and at least one intake judgment; other judgments may remain unknown.
- Execution mode is shadow only. These records describe proposals, not actions.
- Workflow IDs and versions must match supplied context; decisions must also
  match policy.version. Workflow and policy versions must be bumped when their
  definitions change. Version strings alone do not enforce that discipline.

## Constructing records

TicketInput and WorkflowConfig can be constructed normally. Predictions,
decisions, and labels require the workflow to validate their references. Use
`RecordType.for_workflow(data, workflow)`; constructing them without workflow
context raises a validation error. For JSON input, use
`RecordType.model_validate_json(text, context={"workflow": workflow})`.

```python
from rightdesk.config import load_workflow
from rightdesk.schemas import ModelPrediction, PolicyDecision, ReviewedLabel, TicketInput

workflow = load_workflow("configs/workflows/retail_support.yaml")
ticket = TicketInput(
    case_id="demo-1", conversation_id="conversation-1",
    text="I cannot sign in.", language="en", channel="chat",
)
identity = dict(
    case_id=ticket.case_id,
    workflow_id=workflow.workflow_id,
    workflow_version=workflow.version,
)
# Illustrative probabilities, not a model run or calibrated result.
prediction = ModelPrediction.for_workflow({
    **identity,
    "queue_probabilities": {
        "account": 0.75, "orders_delivery": 0.05, "returns_refunds": 0.05,
        "billing_subscriptions": 0.05, "product_policy": 0.05, "technical_support": 0.05,
    },
    "model_revision": "example-v1",
}, workflow)
decision = PolicyDecision.for_workflow({
    **identity, "action": "review", "reason_code": "low_confidence",
    "policy_version": workflow.policy.version,
}, workflow)
label = ReviewedLabel.for_workflow({**identity}, workflow)
assert prediction.sufficient_information_probability is None
assert label.sufficient_information_at_intake is None
```

Models are boundary-validated records. Revalidate through `for_workflow` after
editing or loading records; do not use `model_construct` or unchecked
`model_copy(update=...)` for incoming data. Record validation does not calculate
threshold decisions or establish whether a prediction is correct; those belong
to the later policy and evaluation phases.

## Configuration loading

`load_workflow` and `load_experiment` accept a path and return validated objects.
The safe YAML loader rejects duplicate keys, unknown fields, and invalid types;
errors include the source path. YAML version fields must be quoted strings.
Use `resolve_config_path(config_path, reference)` for file references relative
to the containing YAML. The CLI's `--experiment` check also loads the linked
workflow. Dataset paths, queue mapping, and Laya revision remain nullable in
the experiment template; passing validation does not mean it can be executed.

`policy.version` is required. Both thresholds and calibration_version remain
null until calibration work. No default threshold or confidence is invented.

## Laya decision fields

The bounded questions are `owner_queue`, `sufficient_information`, and the
experimental `clarification_question`. The adapter will retain distributions
and metadata in ModelPrediction. Workflow policy owns the final action.

The queue definitions and first question template are in
`configs/workflows/retail_support.yaml`. They require review against the dataset
before any queue mapping or calibration is considered final.
