"""Watch state + query persistence (#109).

Small JSON-backed helpers for the `watch` feature — last-checked timestamps and
saved queries — shared by the CLI `watch` command and the MCP watch tool, so they
live in Core instead of being reached out of the presentation layer.
"""

from __future__ import annotations

import json
from pathlib import Path

# Default watch state file locations.
DEFAULT_WATCH_DIR = Path.home() / ".cache" / "research-pipeline" / "watch"
DEFAULT_WATCH_STATE = DEFAULT_WATCH_DIR / "watch_state.json"
DEFAULT_QUERIES_FILE = DEFAULT_WATCH_DIR / "watch_queries.json"


def load_watch_state(state_path: Path) -> dict[str, str]:
    """Load the watch state (last-checked timestamps per query).

    Args:
        state_path: Path to the state JSON file.

    Returns:
        Dict mapping query name to ISO-format last-checked timestamp.
    """
    if state_path.exists():
        return json.loads(state_path.read_text())  # type: ignore[no-any-return]
    return {}


def save_watch_state(state_path: Path, state: dict[str, str]) -> None:
    """Save watch state to disk.

    Args:
        state_path: Path to the state JSON file.
        state: Dict mapping query name to ISO timestamp.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


def load_queries(queries_path: Path) -> list[dict[str, str]]:
    """Load saved watch queries.

    Args:
        queries_path: Path to the queries JSON file.

    Returns:
        List of query dicts with 'name' and 'query' keys.
    """
    if not queries_path.exists():
        return []
    data = json.loads(queries_path.read_text())
    if isinstance(data, list):
        return data
    return []
