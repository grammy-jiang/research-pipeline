"""Path conventions for runs and artifacts."""

import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path("runs")


def generate_run_id() -> str:
    """Generate a new unique run ID.

    Returns:
        A UUID4-based run identifier.
    """
    return uuid.uuid4().hex[:12]


def run_dir(workspace: Path, run_id: str) -> Path:
    """Return the root directory for a specific run.

    Args:
        workspace: Base workspace directory.
        run_id: Run identifier.

    Returns:
        Path to ``<workspace>/<run_id>/``.
    """
    return workspace / run_id


def latest_run_id(workspace: Path) -> str:
    """Return the most recent run id under *workspace*, or ``""`` if none.

    A run is a workspace subdirectory carrying a ``run_manifest.json``. Used to
    resolve ``run_id=""`` to "latest" for inspect/read tools instead of joining
    an empty id onto the workspace and silently reading the workspace root
    itself (#110). "Most recent" is by the manifest's modification time.
    """
    if not workspace.is_dir():
        return ""
    runs = [
        d
        for d in workspace.iterdir()
        if d.is_dir() and (d / "run_manifest.json").is_file()
    ]
    if not runs:
        return ""
    newest = max(runs, key=lambda d: (d / "run_manifest.json").stat().st_mtime)
    return newest.name


def stage_dir(workspace: Path, run_id: str, stage: str) -> Path:
    """Return the directory for a specific stage within a run.

    Args:
        workspace: Base workspace directory.
        run_id: Run identifier.
        stage: Stage name (plan, search, screen, download, convert, extract, summarize).

    Returns:
        Path to ``<workspace>/<run_id>/<stage>/``.
    """
    d = run_dir(workspace, run_id) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir(workspace: Path, run_id: str) -> Path:
    """Return the logs directory for a run.

    Args:
        workspace: Base workspace directory.
        run_id: Run identifier.

    Returns:
        Path to ``<workspace>/<run_id>/logs/``.
    """
    d = run_dir(workspace, run_id) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


STAGE_NAMES = [
    "plan",
    "search",
    "screen",
    "download",
    "convert",
    "extract",
    "summarize",
]
