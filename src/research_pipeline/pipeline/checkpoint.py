"""Enhanced checkpoint support for pipeline stage tracking.

Writes a checkpoint JSON file after each stage with timing, artifact
counts, and output hashes.  On resume, verifies artifact integrity
by comparing hashes — stages with unchanged outputs are safely skipped.
"""

import json
import logging
from pathlib import Path

from research_pipeline.infra.clock import utc_now
from research_pipeline.infra.hashing import sha256_file

logger = logging.getLogger(__name__)


class StageCheckpoint:
    """Represents the checkpoint state for a single pipeline stage."""

    def __init__(
        self,
        stage: str,
        status: str = "pending",
        started_at: str = "",
        ended_at: str = "",
        duration_ms: int = 0,
        artifact_count: int = 0,
        artifact_hashes: dict[str, str] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        self.stage = stage
        self.status = status
        self.started_at = started_at
        self.ended_at = ended_at
        self.duration_ms = duration_ms
        self.artifact_count = artifact_count
        self.artifact_hashes = artifact_hashes or {}
        self.errors = errors or []

    def to_dict(self) -> dict[str, object]:
        """Serialize to dict."""
        return {
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "artifact_count": self.artifact_count,
            "artifact_hashes": self.artifact_hashes,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "StageCheckpoint":
        """Deserialize from dict."""
        return cls(
            stage=str(data.get("stage", "")),
            status=str(data.get("status", "pending")),
            started_at=str(data.get("started_at", "")),
            ended_at=str(data.get("ended_at", "")),
            duration_ms=int(data.get("duration_ms", 0)),
            artifact_count=int(data.get("artifact_count", 0)),
            artifact_hashes=dict(data.get("artifact_hashes", {})),  # type: ignore[arg-type]
            errors=list(data.get("errors", [])),  # type: ignore[arg-type]
        )


def compute_stage_hashes(output_paths: list[Path]) -> dict[str, str]:
    """Compute SHA-256 hashes for all stage output artifacts.

    Args:
        output_paths: List of output file paths.

    Returns:
        Dict mapping filename to SHA-256 hash.
    """
    hashes: dict[str, str] = {}
    for path in output_paths:
        if path.exists() and path.is_file():
            hashes[path.name] = sha256_file(path)
    return hashes


def write_checkpoint(
    run_root: Path,
    stage: str,
    status: str,
    started_at: str,
    output_paths: list[Path],
    errors: list[str] | None = None,
) -> StageCheckpoint:
    """Write a checkpoint file for a completed stage.

    Args:
        run_root: Pipeline run root directory.
        stage: Stage name.
        status: Stage status ("completed" or "failed").
        started_at: ISO timestamp of stage start.
        output_paths: List of output artifact paths.
        errors: Any errors that occurred.

    Returns:
        The written StageCheckpoint.
    """
    ended_at = utc_now().isoformat()
    hashes = compute_stage_hashes(output_paths)

    checkpoint = StageCheckpoint(
        stage=stage,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        artifact_count=len(output_paths),
        artifact_hashes=hashes,
        errors=errors or [],
    )

    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{stage}.checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(checkpoint.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    logger.info(
        "Checkpoint written: %s (status=%s, artifacts=%d)",
        stage,
        status,
        len(hashes),
    )
    return checkpoint


def load_checkpoint(run_root: Path, stage: str) -> StageCheckpoint | None:
    """Load a checkpoint file for a stage.

    Args:
        run_root: Pipeline run root directory.
        stage: Stage name.

    Returns:
        StageCheckpoint if found, None otherwise.
    """
    checkpoint_path = run_root / "checkpoints" / f"{stage}.checkpoint.json"
    if not checkpoint_path.exists():
        return None

    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return StageCheckpoint.from_dict(data)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load checkpoint for %s: %s", stage, exc)
        return None


def is_stage_valid(
    run_root: Path,
    stage: str,
    current_output_paths: list[Path],
) -> bool:
    """Check if a stage's checkpoint is still valid (artifacts unchanged).

    Args:
        run_root: Pipeline run root directory.
        stage: Stage name.
        current_output_paths: Current output paths to verify.

    Returns:
        True if checkpoint exists, completed, and hashes match.
    """
    cp = load_checkpoint(run_root, stage)
    if cp is None:
        return False
    if cp.status != "completed":
        return False

    # Verify artifact hashes match
    current_hashes = compute_stage_hashes(current_output_paths)
    if not current_hashes:
        # No current artifacts to verify — trust the checkpoint
        return True

    for filename, expected_hash in cp.artifact_hashes.items():
        actual = current_hashes.get(filename)
        if actual is not None and actual != expected_hash:
            logger.info(
                "Checkpoint invalid for %s: %s hash mismatch",
                stage,
                filename,
            )
            return False

    return True


def load_all_checkpoints(run_root: Path) -> dict[str, StageCheckpoint]:
    """Load all checkpoints for a run.

    Args:
        run_root: Pipeline run root directory.

    Returns:
        Dict mapping stage name to checkpoint.
    """
    checkpoint_dir = run_root / "checkpoints"
    if not checkpoint_dir.exists():
        return {}

    checkpoints: dict[str, StageCheckpoint] = {}
    for path in sorted(checkpoint_dir.glob("*.checkpoint.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cp = StageCheckpoint.from_dict(data)
            checkpoints[cp.stage] = cp
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping checkpoint %s: %s", path.name, exc)

    return checkpoints
