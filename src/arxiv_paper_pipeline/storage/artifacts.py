"""Artifact registration and hashing utilities."""

import logging
from pathlib import Path

from arxiv_paper_pipeline.infra.clock import utc_now
from arxiv_paper_pipeline.infra.hashing import sha256_file
from arxiv_paper_pipeline.models.manifest import ArtifactRecord

logger = logging.getLogger(__name__)


def register_artifact(
    path: Path,
    artifact_type: str,
    producer: str,
    run_root: Path,
    inputs: list[str] | None = None,
    tool_fingerprint: str | None = None,
) -> ArtifactRecord:
    """Create an artifact record for a file.

    Args:
        path: Path to the artifact file.
        artifact_type: Type identifier (atom_xml, pdf, markdown, etc.).
        producer: Stage name that produced this artifact.
        run_root: Run root directory (for relative path calculation).
        inputs: Input artifact IDs.
        tool_fingerprint: Tool fingerprint if applicable.

    Returns:
        ArtifactRecord for the file.
    """
    try:
        rel_path = str(path.relative_to(run_root))
    except ValueError:
        rel_path = str(path)

    file_hash = sha256_file(path) if path.exists() else ""

    record = ArtifactRecord(
        artifact_id=f"{producer}:{path.name}",
        artifact_type=artifact_type,
        path=rel_path,
        sha256=file_hash,
        producer=producer,
        inputs=inputs or [],
        tool_fingerprint=tool_fingerprint,
        created_at=utc_now(),
    )

    logger.debug("Registered artifact: %s (%s)", record.artifact_id, artifact_type)
    return record
