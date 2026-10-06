# Data plan

1. Use [ABCD](https://github.com/asappresearch/abcd) for the first retail-support
   experiment. Define and version a reviewed mapping from its intents to our
   six queues. Preserve its official final test split.
2. Audit varied opening messages to establish annotation rules. Keep scenario
   labels and future dialogue out of intake features. Information sufficiency
   and useful questions need separate review.
3. Maintain train, development, calibration, and final-test roles. Keep
   conversations and duplicates together. Fit calibration and choose thresholds
   before examining final-test outcomes.
4. Evaluate [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive)
   separately if multilingual intent experiments are needed. Its voice-assistant
   domain does not establish multilingual support performance. Keep parallel
   language variants together across splits.
5. Use reviewed pilot tickets later to evaluate actual operational relevance.

Source manifests should record URLs, versions, licenses, checksums, preparation
rules, queue mappings, and split assignments. Public research data, translated
examples, synthetic examples, and pilot data must remain identifiable in results.

## Phase 2 implementation

ABCD v1.1 is downloaded from the official repository at commit
`6b8700ce67c6b37b062dd7a60abc76d7ef832a97`. The original gzip archive and MIT
license are kept unchanged under `data/raw/abcd/<revision>/`. The source manifest
records their URLs, byte sizes, SHA-256 checksums, and retrieval time.

```powershell
uv run rightdesk data download-abcd
uv run rightdesk data import-abcd
```

Downloads occur only in `download-abcd`. Repeating that command verifies local
checksums and reuses files. `--revision` accepts only a full commit SHA; if using
a different revision, pass its manifest explicitly to the importer. The importer
uses no network and verifies both source files before reading the archive.

For a separately named run:

```powershell
uv run rightdesk data import-abcd --source-manifest data/manifests/abcd-source-6b8700ce67c6b37b062dd7a60abc76d7ef832a97.json --output-dir data/processed/abcd-intake-copy --manifest data/manifests/abcd-intake-copy.json
```

CLI paths are relative to the current directory; paths *inside manifests* are
relative to the manifest file. Existing preparation outputs are never overwritten.
Temporary directories keep interrupted downloads/imports from exposing partially
written data. The manifest is published last as the completion record; if writing
it fails after the output directory is published, preserve the directory for
inspection and choose fresh paths for the next run.

### Intake boundary

For each conversation, take exactly the first `customer` turn in `original`.
Preserve its text, including case and whitespace. A greeting such as "Hi" stays
the intake message. Do not search later turns for a more informative request.
Empty first messages, missing customers, malformed intake turns, or missing
source labels are explicitly rejected and counted. Duplicate conversation IDs
abort preparation, including duplicates across official splits.

The model input is serialized as TicketInput only: case_id, conversation_id,
text, language, and channel. Language is set to `en` based on the corpus, not
detected per message. IDs use `abcd:<convo_id>` and `abcd:<convo_id>:intake`.
Neither scenario fields, future replies, delexicalized targets, nor split names
are copied into tickets. Corpus IDs are for joins, not classifier features.

`references.jsonl` contains source flow/subflow labels, source split, conversation
ID, conversation index, and the selected turn index (zero-based). These are
scenario-derived targets, not human ReviewedLabel records and not queue labels.
Personal/order/product scenario details are not copied into either prepared file.
An eventual target does not imply sufficient evidence in the initial message.

### Current output

| Official split | Source conversations | Prepared tickets | Rejected |
|---|---:|---:|---:|
| train | 8,034 | 8,034 | 0 |
| dev | 1,004 | 1,004 | 0 |
| test | 1,004 | 1,004 | 0 |
| Total | 10,042 | 10,042 | 0 |

The preparation manifest records counts, output checksums, source manifest/archive
checksums, extraction rules, and hashes of the importer and ticket schema code.
JSONL keeps Phase 2 dependency-light and inspectable; later experiment artifacts
can use Parquet. The raw archive is read into memory during preparation.

The test split has undergone structural ingestion checks only. No routing
performance has been measured or used to tune decisions. Queue mapping,
duplicate-text handling, and reserving a calibration partition belong to Phase 3.
Their manifest fields are currently null. No human intake judgments have been
created. This research corpus does not establish production support performance.

## Phase 3: queue mapping and isolated partitions

Phase 2 artifacts and their historical manifests remain unchanged. Phase 3 adds
`configs/mappings/abcd_retail_v1.yaml`, `configs/data/abcd_partitions_v1.yaml`,
and the prepared dataset under `data/processed/abcd-partitions-v1/`.

### Mapping decisions

Mapping v1 explicitly lists each supported flow/subflow combination from the
training taxonomy. Unknown combinations cause preparation to fail with counts
per unmapped combination. They are never silently assigned a fallback queue.
Workflow identity/version and all target queue references are validated.

| Source flow | Default target | Exceptions |
|---|---|---|
| account_access | account | None |
| manage_account | account | Missing credit and service changes: billing; shipping questions: orders |
| order_issue | orders_delivery | Mystery fees: billing |
| product_defect | returns_refunds | None |
| purchase_dispute | billing_subscriptions | Already-returned charges: returns; stock availability: product |
| shipping_issue | orders_delivery | None |
| single_item_query | product_policy | None |
| storewide_query | product_policy | None |
| subscription_inquiry | billing_subscriptions | None |
| troubleshoot_site | technical_support | Includes technical card/checkout faults |

The configuration includes rationales and exact subflow IDs. This is an
agent-authored provisional ownership convention, pending human review. It is
not a native ABCD queue ontology. Boundary cases such as promotional disputes
and charges for returned goods particularly need review. Mapping revisions must
receive new versions and new output paths; do not edit historical artifacts.

### Duplicate and split rules

1. Hash the first customer message after Unicode NFKC normalization, case folding,
   and whitespace collapse. Preserve original ticket text. Punctuation is retained;
   semantic/near-duplicate detection is not implemented.
2. Preserve every official test case. Exclude matching train/dev cases from active
   partitions. No non-test case is promoted into test.
3. Keep remaining official dev cases as development. Exclude matching train cases.
4. Divide remaining official-train duplicate groups using a seeded SHA-256 threshold
   (seed 42, calibration fraction 0.15). Whole groups go to train or calibration.
   This is an approximate group fraction, not an exact or stratified ticket quota.
   Label values are not used to choose partitions.
5. Check unique case/conversation identities, complete ticket/reference joins,
   nonempty partitions, all official test IDs retained, and zero cross-partition
   duplicate groups. Report any missing queues rather than silently reshuffling.

Reading test input hashes for isolation and labels for structural mapping/counts
does not train a model or select a threshold. Test outcomes remain unused for
model or policy selection. Repeated messages within a partition are retained;
later uncertainty estimates should account for these groups rather than assuming
all cases are independent.

### Results

| Source | Train | Development | Calibration | Test | Excluded |
|---|---:|---:|---:|---:|---:|
| Official train | 5,792 | 0 | 1,063 | 0 | 1,179 |
| Official dev | 0 | 863 | 0 | 0 | 141 |
| Official test | 0 | 0 | 0 | 1,004 | 0 |

All six queues occur in every partition. Of 1,320 exclusions, 1,224 match a test
message and 96 match a remaining development message. No source data was deleted.
There are 195 repeated-message groups across the source corpus, including 68
groups with differing mapped queues. These conflicts are retained and reported;
an ambiguous greeting can legitimately precede different issues. They are not
resolved using hidden scenario labels or treated as known intake sufficiency.

### Outputs and reproducibility

- Each role has `tickets.jsonl` containing only TicketInput fields and a separate
  `labels.jsonl` containing unreviewed ReviewedLabel records. Only eventual owner
  is filled; intake sufficiency, acceptable queues, questions, and reviewer stay null.
- `assignments.jsonl` accounts for every imported case, including excluded cases,
  source location/labels, duplicate key, chosen partition, and exclusion reason.
  It must not be passed to prediction adapters as input features.
- `report.json` records queue counts, source-to-partition counts, repeated and
  conflicting groups, exclusions, unmapped labels, and preservation checks.
- `data/manifests/abcd-partitions-v1.json` stores the report, configuration and
  input-manifest hashes, implementation hashes, algorithm versions, and all
  output checksums. Manifest paths are relative when possible; references across
  Windows drives use absolute paths and require adjustment when moved.

The partition command validates input checksums and refuses existing output paths.
New files are staged before publication, with the manifest written last. Fixed
inputs/config produce identical data artifacts regardless of input row order;
manifest timestamps and path references may differ between runs. The baseline
experiment references mapping v1 and this manifest. Calibration data is reserved,
but no probability calibration or model fitting has occurred yet.
