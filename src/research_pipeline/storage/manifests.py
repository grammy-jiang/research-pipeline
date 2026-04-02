"""Manifest read/write utilities."""

import json
import logging
from pathlib import Path

from research_pipeline.models.manifest import RunManifest, StageRecord

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "run_manifest.json"


def load_manifest(run_root: Path) -> RunManifest | None:
    """Load a run manifest from disk.

    Args:
        run_root: Root run directory.

    Returns:
        Parsed RunManifest, or None if not found.
    """
    path = run_root / MANIFEST_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = RunManifest.model_validate(data)
    logger.debug("Loaded manifest for run %s", manifest.run_id)
    return manifest


def save_manifest(run_root: Path, manifest: RunManifest) -> Path:
    """Save a run manifest to disk.

    Args:
        run_root: Root run directory.
        manifest: The manifest to save.

    Returns:
        Path to the saved manifest file.
    """
    path = run_root / MANIFEST_FILENAME
    data = manifest.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.debug("Saved manifest for run %s", manifest.run_id)
    return path


def update_stage(
    manifest: RunManifest,
    stage_record: StageRecord,
) -> RunManifest:
    """Update or add a stage record in the manifest.

    Args:
        manifest: The run manifest.
        stage_record: Stage record to update.

    Returns:
        Updated manifest.
    """
    manifest.stages[stage_record.stage_name] = stage_record
    return manifest


def write_jsonl(path: Path, records: list[dict]) -> None:  # type: ignore[type-arg]
    """Write a list of dicts as JSONL.

    Args:
        path: Output file path.
        records: List of dictionaries to write.
    """
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    logger.debug("Wrote %d records to %s", len(records), path)


def read_jsonl(path: Path) -> list[dict]:  # type: ignore[type-arg]
    """Read a JSONL file into a list of dicts.

    Args:
        path: Input file path.

    Returns:
        List of parsed dictionaries.
    """
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.debug("Read %d records from %s", len(records), path)
    return records
