"""Workspace layout management for pipeline runs."""

import logging
from pathlib import Path

from research_pipeline.infra.paths import generate_run_id, run_dir

logger = logging.getLogger(__name__)


STAGE_SUBDIRS = {
    "plan": "plan",
    "search": "search",
    "search_raw": "search/raw",
    "screen": "screen",
    "download": "download/pdf",
    "download_root": "download",
    "convert": "convert/markdown",
    "convert_root": "convert",
    "extract": "extract",
    "summarize": "summarize",
    "logs": "logs",
}


def init_run(workspace: Path, run_id: str | None = None) -> tuple[str, Path]:
    """Initialize a new run directory with all stage subdirectories.

    Args:
        workspace: Base workspace directory.
        run_id: Optional run ID. Generated if not provided.

    Returns:
        Tuple of (run_id, run_dir_path).
    """
    if run_id is None:
        run_id = generate_run_id()

    root = run_dir(workspace, run_id)
    root.mkdir(parents=True, exist_ok=True)

    for subdir in STAGE_SUBDIRS.values():
        (root / subdir).mkdir(parents=True, exist_ok=True)

    logger.info("Initialized run directory: %s", root)
    return run_id, root


def get_stage_dir(run_root: Path, stage: str) -> Path:
    """Get the directory for a specific stage within a run.

    Args:
        run_root: Root run directory.
        stage: Stage key from STAGE_SUBDIRS.

    Returns:
        Path to the stage directory.
    """
    subdir = STAGE_SUBDIRS.get(stage, stage)
    d = run_root / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d
