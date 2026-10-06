"""Reproducible ABCD download and first-customer-message ingestion."""

import gzip
import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from urllib.request import urlopen

from pydantic import Field, ValidationError

from rightdesk.schemas import Contract, NonEmpty, TicketInput

REVISION = "6b8700ce67c6b37b062dd7a60abc76d7ef832a97"
REPOSITORY = "https://github.com/asappresearch/abcd"
PREPARATION_VERSION = "abcd-first-customer-v1"
SPLITS = ("train", "dev", "test")
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class IngestionError(ValueError):
    """Invalid source data or incompatible existing output."""


class SourceFile(Contract):
    path: NonEmpty
    url: NonEmpty
    sha256: SHA256
    size_bytes: Annotated[int, Field(ge=0)]


class SourceManifest(Contract):
    schema_version: Literal[1] = 1
    dataset: Literal["abcd"] = "abcd"
    dataset_version: Literal["1.1"] = "1.1"
    repository: Literal[REPOSITORY] = REPOSITORY
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    license: Literal["MIT"] = "MIT"
    retrieved_at: NonEmpty
    archive: SourceFile
    license_file: SourceFile


class SourceReference(Contract):
    """Dataset targets and provenance; never pass this record to a model."""

    case_id: NonEmpty
    conversation_id: NonEmpty
    source_conversation_id: int | NonEmpty
    source_split: Literal["train", "dev", "test"]
    source_index: Annotated[int, Field(ge=0)]
    intake_turn_index: Annotated[int, Field(ge=0)]
    flow: NonEmpty
    subflow: NonEmpty


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(path: Path, manifest_path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), manifest_path.resolve().parent)).as_posix()


def write_json(path: Path, data: dict) -> None:
    # Exclusive creation prevents a repeated run from overwriting provenance.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_source_manifest(path: Path) -> SourceManifest:
    try:
        manifest = SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))
        for item in (manifest.archive, manifest.license_file):
            source = (path.parent / item.path).resolve()
            if source.stat().st_size != item.size_bytes or checksum(source) != item.sha256:
                raise IngestionError(f"Source checksum mismatch: {source}")
        return manifest
    except (OSError, ValidationError) as exc:
        raise IngestionError(f"Cannot validate source manifest {path}: {exc}") from exc


def download_abcd(data_dir: Path, revision: str = REVISION) -> Path:
    """Download only on explicit invocation; retain original archive and license."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise IngestionError("revision must be a full lowercase 40-character Git commit SHA")
    data_dir = data_dir.resolve()
    manifest_path = data_dir / "manifests" / f"abcd-source-{revision}.json"
    if manifest_path.exists():
        existing = load_source_manifest(manifest_path)
        if existing.revision != revision:
            raise IngestionError("Existing source manifest has a different revision")
        return manifest_path
    raw_dir = data_dir / "raw" / "abcd" / revision
    if raw_dir.exists():
        raise IngestionError(f"Unmanifested source directory already exists: {raw_dir}")
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/asappresearch/abcd/{revision}"
    files = {}
    with tempfile.TemporaryDirectory(prefix=".abcd-download-", dir=raw_dir.parent) as temporary:
        staging = Path(temporary)
        for key, remote, name in (
            ("archive", "data/abcd_v1.1.json.gz", "abcd_v1.1.json.gz"),
            ("license_file", "LICENSE", "LICENSE"),
        ):
            url = f"{base}/{remote}"
            target = staging / name
            with urlopen(url, timeout=60) as response, target.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            files[key] = SourceFile(
                path=relative_path(raw_dir / name, manifest_path),
                url=url,
                sha256=checksum(target),
                size_bytes=target.stat().st_size,
            )
        # Check compression integrity before publishing downloaded files.
        with gzip.open(staging / "abcd_v1.1.json.gz", "rb") as archive:
            while archive.read(1024 * 1024):
                pass
        manifest = SourceManifest(
            revision=revision,
            retrieved_at=datetime.now(UTC).isoformat(),
            **files,
        )
        staging.rename(raw_dir)
    write_json(manifest_path, manifest.model_dump())
    return manifest_path


def extract_case(record: dict, split: str, index: int) -> tuple[TicketInput, SourceReference]:
    """Read exactly the first customer turn, including greeting-only openings."""
    source_id = record.get("convo_id")
    if type(source_id) not in (str, int) or not str(source_id).strip():
        raise IngestionError("invalid_conversation_id")
    conversation_id = f"abcd:{source_id}"
    turns = record.get("original")
    if not isinstance(turns, list):
        raise IngestionError("missing_original_conversation")
    for turn_index, turn in enumerate(turns):
        if (
            not isinstance(turn, list)
            or len(turn) != 2
            or not isinstance(turn[0], str)
            or turn[0]
            not in {
                "agent",
                "customer",
                "action",
            }
        ):
            raise IngestionError("invalid_turn_before_intake")
        if turn[0] == "customer":
            # Do not skip a blank first customer message in favor of a later one.
            if not isinstance(turn[1], str) or not turn[1].strip():
                raise IngestionError("empty_or_invalid_first_customer_message")
            ticket = TicketInput(
                case_id=f"{conversation_id}:intake",
                conversation_id=conversation_id,
                text=turn[1],
                language="en",
                channel="chat",
            )
            scenario = record.get("scenario")
            if not isinstance(scenario, dict):
                raise IngestionError("missing_scenario_labels")
            try:
                reference = SourceReference(
                    case_id=ticket.case_id,
                    conversation_id=conversation_id,
                    source_conversation_id=source_id,
                    source_split=split,
                    source_index=index,
                    intake_turn_index=turn_index,
                    flow=scenario.get("flow"),
                    subflow=scenario.get("subflow"),
                )
            except ValidationError as exc:
                raise IngestionError("invalid_source_labels") from exc
            return ticket, reference
    raise IngestionError("no_customer_message")


def import_abcd(source_manifest: Path, output_dir: Path, manifest_path: Path) -> Path:
    """Prepare separate JSONL streams, preserving official splits in references."""
    source_manifest, output_dir, manifest_path = (
        path.resolve() for path in (source_manifest, output_dir, manifest_path)
    )
    if output_dir.exists() or manifest_path.exists():
        raise IngestionError("Output directory or preparation manifest exists; choose fresh paths")
    if manifest_path.is_relative_to(output_dir):
        raise IngestionError("Preparation manifest must be outside the output directory")
    source = load_source_manifest(source_manifest)
    archive_path = source_manifest.parent / source.archive.path
    try:
        with gzip.open(archive_path, "rt", encoding="utf-8") as stream:
            dataset = json.load(stream)
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
        raise IngestionError(f"Cannot read ABCD archive: {exc}") from exc
    if not isinstance(dataset, dict) or set(dataset) != set(SPLITS):
        raise IngestionError("Expected an ABCD object with exactly train, dev, and test splits")
    if any(not isinstance(dataset[split], list) or not dataset[split] for split in SPLITS):
        raise IngestionError("Every source split must be a nonempty conversation list")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        split: {"source": len(dataset[split]), "accepted": 0, "rejected": 0} for split in SPLITS
    }
    seen = set()
    with tempfile.TemporaryDirectory(prefix=".abcd-import-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        with (
            (staging / "tickets.jsonl").open("w", encoding="utf-8", newline="\n") as tickets,
            (staging / "references.jsonl").open("w", encoding="utf-8", newline="\n") as references,
            (staging / "rejections.jsonl").open("w", encoding="utf-8", newline="\n") as rejections,
        ):
            for split in SPLITS:
                for index, record in enumerate(dataset[split]):
                    if isinstance(record, dict):
                        source_id = record.get("convo_id")
                        if type(source_id) in (str, int) and str(source_id).strip():
                            key = str(source_id)
                            if key in seen:
                                raise IngestionError(
                                    f"Duplicate conversation ID across source: {key}"
                                )
                            seen.add(key)
                    try:
                        if not isinstance(record, dict):
                            raise IngestionError("invalid_conversation_record")
                        ticket, reference = extract_case(record, split, index)
                    except IngestionError as exc:
                        counts[split]["rejected"] += 1
                        rejections.write(
                            json.dumps(
                                {
                                    "source_split": split,
                                    "source_index": index,
                                    "reason": str(exc),
                                }
                            )
                            + "\n"
                        )
                        continue
                    tickets.write(ticket.model_dump_json() + "\n")
                    references.write(reference.model_dump_json() + "\n")
                    counts[split]["accepted"] += 1
        if not any(count["accepted"] for count in counts.values()):
            raise IngestionError("No valid intake cases were found")
        outputs = {
            name: {
                "path": relative_path(output_dir / name, manifest_path),
                "sha256": checksum(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in ("tickets.jsonl", "references.jsonl", "rejections.jsonl")
        }
        manifest = {
            "schema_version": 1,
            "dataset": "abcd",
            "dataset_version": source.dataset_version,
            "revision": source.revision,
            "license": source.license,
            "created_at": datetime.now(UTC).isoformat(),
            "source_manifest": relative_path(source_manifest, manifest_path),
            "source_manifest_sha256": checksum(source_manifest),
            "source_archive_sha256": source.archive.sha256,
            "preparation_version": PREPARATION_VERSION,
            "importer_sha256": checksum(Path(__file__)),
            "ticket_schema_sha256": checksum(Path(__file__).parents[2] / "schemas.py"),
            "input_rule": "First customer message in original; exact text; no later turns",
            "language_rule": "en from the English corpus, not language detection",
            "source_type": "human research dialogues with fictional customer scenarios",
            "label_rule": "scenario.flow/subflow only; not reviewed intake labels",
            "split_rule": "Official train/dev/test preserved in references; no repartitioning",
            "queue_mapping_version": None,
            "calibration_split": None,
            "counts": counts,
            "outputs": outputs,
        }
        staging.rename(output_dir)
    write_json(manifest_path, manifest)
    return manifest_path
