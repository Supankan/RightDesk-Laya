"""Versioned queue targets and deterministic partitions with duplicate isolation."""

import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from pydantic import Field

from rightdesk.config import _load, load_workflow, resolve_config_path
from rightdesk.data.importers.abcd import (
    IngestionError,
    SourceReference,
    checksum,
    write_json,
)
from rightdesk.data.importers.abcd import (
    relative_path as source_relative_path,
)
from rightdesk.data.mapping import load_mapping
from rightdesk.data.preprocessing import NORMALIZATION_VERSION, duplicate_key
from rightdesk.schemas import Contract, NonEmpty, ReviewedLabel, TicketInput

ROLES = ("train", "development", "calibration", "test")
ALGORITHM_VERSION = "test-dev-priority-hash-calibration-v1"


def relative_path(path: Path, manifest_path: Path) -> str:
    """Use portable relative references when possible; Windows drives may differ."""
    try:
        return source_relative_path(path, manifest_path)
    except ValueError:
        return path.resolve().as_posix()


class PartitionConfig(Contract):
    version: NonEmpty
    intake_manifest: NonEmpty
    workflow: NonEmpty
    mapping: NonEmpty
    seed: Annotated[int, Field(ge=0)]
    calibration_fraction: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    output_directory: NonEmpty
    manifest: NonEmpty


def verified_intake(path: Path) -> tuple[dict[str, TicketInput], dict[str, SourceReference]]:
    """Check artifact integrity and a one-to-one join before any partition is written."""
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if (
        metadata.get("dataset") != "abcd"
        or metadata.get("preparation_version") != "abcd-first-customer-v1"
    ):
        raise IngestionError("Expected an ABCD first-customer-v1 intake manifest")
    artifacts = {}
    for name in ("tickets.jsonl", "references.jsonl", "rejections.jsonl"):
        item = metadata["outputs"][name]
        file = resolve_config_path(path, item["path"])
        if checksum(file) != item["sha256"] or file.stat().st_size != item["size_bytes"]:
            raise IngestionError(f"Intake checksum mismatch: {file}")
        artifacts[name] = file
    source_path = resolve_config_path(path, metadata["source_manifest"])
    if checksum(source_path) != metadata["source_manifest_sha256"]:
        raise IngestionError("Source manifest checksum mismatch")
    records = []
    for name, schema in (("tickets.jsonl", TicketInput), ("references.jsonl", SourceReference)):
        indexed = {}
        with artifacts[name].open(encoding="utf-8") as stream:
            for line in stream:
                record = schema.model_validate_json(line)
                if record.case_id in indexed:
                    raise IngestionError(f"Duplicate case ID in {name}: {record.case_id}")
                indexed[record.case_id] = record
        records.append(indexed)
    tickets, references = records
    if not tickets or tickets.keys() != references.keys():
        raise IngestionError("Tickets and references must have identical nonempty case IDs")
    conversations = set()
    for key, ticket in tickets.items():
        if ticket.conversation_id != references[key].conversation_id:
            raise IngestionError(f"Ticket/reference conversation mismatch: {key}")
        if ticket.conversation_id in conversations:
            raise IngestionError("Expected exactly one intake case per conversation")
        conversations.add(ticket.conversation_id)
    counts = Counter(ref.source_split for ref in references.values())
    if any(
        counts[split] != metadata["counts"][split]["accepted"] for split in ("train", "dev", "test")
    ):
        raise IngestionError("Intake manifest counts do not match references")
    return tickets, references


def assign_partitions(
    tickets: dict[str, TicketInput],
    references: dict[str, SourceReference],
    seed: int,
    calibration_fraction: float,
) -> list[dict]:
    """Assign by input-only duplicate keys; source labels never select a partition."""
    if type(seed) is not int or seed < 0 or not 0 < calibration_fraction < 1:
        raise IngestionError("Invalid seed or calibration fraction")
    groups = defaultdict(list)
    for key, ticket in tickets.items():
        groups[duplicate_key(ticket.text)].append(key)
    assignments = []
    for group, keys in sorted(groups.items()):
        sources = {references[key].source_split for key in keys}
        if "test" in sources:
            owner = "test"
        elif "dev" in sources:
            owner = "development"
        else:
            value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16) / 2**256
            owner = "calibration" if value < calibration_fraction else "train"
        for key in sorted(keys):
            source = references[key].source_split
            excluded = (owner == "test" and source != "test") or (
                owner == "development" and source != "dev"
            )
            assignments.append(
                {
                    "case_id": key,
                    "conversation_id": tickets[key].conversation_id,
                    "source_split": source,
                    "duplicate_group": group,
                    "partition": None if excluded else owner,
                    "exclusion_reason": f"duplicate_of_{owner}" if excluded else None,
                }
            )
    return sorted(assignments, key=lambda item: item["case_id"])


def prepare_partitions(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = _load(config_path, PartitionConfig)
    intake_path = resolve_config_path(config_path, config.intake_manifest)
    workflow_path = resolve_config_path(config_path, config.workflow)
    mapping_path = resolve_config_path(config_path, config.mapping)
    output_dir = resolve_config_path(config_path, config.output_directory)
    manifest_path = resolve_config_path(config_path, config.manifest)
    if output_dir.exists() or manifest_path.exists():
        raise IngestionError("Partition output or manifest exists; choose fresh paths")
    if manifest_path.is_relative_to(output_dir):
        raise IngestionError("Partition manifest must be outside the output directory")
    workflow = load_workflow(workflow_path)
    mapping = load_mapping(mapping_path, workflow)
    tickets, references = verified_intake(intake_path)
    lookup = mapping.lookup()
    unmapped = Counter(
        (ref.flow, ref.subflow)
        for ref in references.values()
        if (ref.flow, ref.subflow) not in lookup
    )
    if unmapped:
        detail = "; ".join(
            f"{flow}/{subflow}: {count}" for (flow, subflow), count in sorted(unmapped.items())
        )
        raise IngestionError(f"Unmapped source labels (no outputs written): {detail}")
    targets = {key: lookup[(ref.flow, ref.subflow)] for key, ref in references.items()}
    assignments = assign_partitions(tickets, references, config.seed, config.calibration_fraction)
    counts = Counter(row["partition"] for row in assignments if row["partition"])
    if any(counts[role] == 0 for role in ROLES):
        raise IngestionError(f"Every partition must be nonempty; got {dict(counts)}")
    group_roles = defaultdict(set)
    for row in assignments:
        if row["partition"]:
            group_roles[row["duplicate_group"]].add(row["partition"])
    if any(len(roles) != 1 for roles in group_roles.values()):
        raise IngestionError("Duplicate group leaked between partitions")
    expected_test = {key for key, ref in references.items() if ref.source_split == "test"}
    if {row["case_id"] for row in assignments if row["partition"] == "test"} != expected_test:
        raise IngestionError("Official test membership changed")
    queue_counts = {role: dict.fromkeys(sorted(workflow.queue_ids), 0) for role in ROLES}
    source_counts = {
        split: dict.fromkeys((*ROLES, "excluded"), 0) for split in ("train", "dev", "test")
    }
    conflicts = defaultdict(set)
    group_counts = Counter(row["duplicate_group"] for row in assignments)
    for row in assignments:
        source_counts[row["source_split"]][row["partition"] or "excluded"] += 1
        conflicts[row["duplicate_group"]].add(targets[row["case_id"]])
        if row["partition"]:
            queue_counts[row["partition"]][targets[row["case_id"]]] += 1
    report = {
        "source_cases": len(tickets),
        "partition_counts": dict(counts),
        "excluded_count": sum(row["partition"] is None for row in assignments),
        "exclusion_reasons": dict(
            Counter(row["exclusion_reason"] for row in assignments if row["exclusion_reason"])
        ),
        "source_to_partition_counts": source_counts,
        "queue_counts": queue_counts,
        "duplicate_groups": sum(size > 1 for size in group_counts.values()),
        "conflicting_queue_groups": sum(len(values) > 1 for values in conflicts.values()),
        "active_groups_by_partition": dict(
            Counter(next(iter(roles)) for roles in group_roles.values())
        ),
        "missing_queues": {
            role: [q for q, n in queues.items() if not n] for role, queues in queue_counts.items()
        },
        "cross_partition_duplicate_groups": 0,
        "official_test_preserved": True,
        "unmapped_labels": [],
        "mapping_status": mapping.status,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".abcd-partitions-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for role in ROLES:
            (staging / role).mkdir()
            with (
                (staging / role / "tickets.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                ) as inputs,
                (staging / role / "labels.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                ) as labels,
            ):
                for row in assignments:
                    if row["partition"] != role:
                        continue
                    key = row["case_id"]
                    inputs.write(tickets[key].model_dump_json() + "\n")
                    label = ReviewedLabel.for_workflow(
                        {
                            "case_id": key,
                            "workflow_id": workflow.workflow_id,
                            "workflow_version": workflow.version,
                            "eventual_owner_queue": targets[key],
                        },
                        workflow,
                    )
                    labels.write(label.model_dump_json() + "\n")
        with (staging / "assignments.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in assignments:
                key = row["case_id"]
                stream.write(
                    json.dumps(
                        {**references[key].model_dump(), **row, "target_queue": targets[key]},
                        sort_keys=True,
                    )
                    + "\n"
                )
        write_json(staging / "report.json", report)
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        artifacts = {
            file.relative_to(staging).as_posix(): {
                "path": relative_path(output_dir / file.relative_to(staging), manifest_path),
                "sha256": checksum(file),
                "size_bytes": file.stat().st_size,
            }
            for file in files
        }
        provenance = {
            name: {"path": relative_path(path, manifest_path), "sha256": checksum(path)}
            for name, path in (
                ("intake_manifest", intake_path),
                ("workflow", workflow_path),
                ("mapping", mapping_path),
                ("partition_config", config_path),
            )
        }
        manifest = {
            "schema_version": 1,
            "dataset": "abcd",
            "created_at": datetime.now(UTC).isoformat(),
            "partition_version": config.version,
            "algorithm_version": ALGORITHM_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "seed": config.seed,
            "calibration_fraction": config.calibration_fraction,
            "calibration_rule": (
                "Seeded SHA256 threshold on remaining official-train text groups; no labels"
            ),
            "duplicate_rule": (
                "Keep test; exclude matching non-test. Keep remaining dev; exclude matching train"
            ),
            "queue_mapping_id": mapping.mapping_id,
            "queue_mapping_version": mapping.version,
            "mapping_status": mapping.status,
            "workflow_id": workflow.workflow_id,
            "workflow_version": workflow.version,
            "provenance": provenance,
            "implementation_sha256": {
                name: checksum(Path(__file__).parent / name)
                for name in ("splits.py", "mapping.py", "preprocessing.py")
            },
            "report": report,
            "outputs": artifacts,
        }
        staging.rename(output_dir)
    write_json(manifest_path, manifest)
    return manifest_path
