"""Non-destructive memory updates with validity annotations.

Instead of overwriting memory entries, entries are annotated with
validity windows (``valid_from``, ``valid_until``) and supersession
chains (``superseded_by``).  Old entries remain accessible for audit
but are excluded from active queries by default.

References:
    Deep-research report Theme 12 (Temporal Memory Management) and
    Engineering Gap 7 (Non-Destructive Updates).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_VERSIONED_DIR = Path.home() / ".cache" / "research-pipeline"
DEFAULT_VERSIONED_PATH = DEFAULT_VERSIONED_DIR / "versioned_memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS versioned_entries (
    entry_id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT DEFAULT '',
    valid_from TEXT NOT NULL,
    valid_until TEXT DEFAULT '',
    superseded_by TEXT DEFAULT '',
    version INTEGER DEFAULT 1,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_key ON versioned_entries(key);
CREATE INDEX IF NOT EXISTS idx_valid_until ON versioned_entries(valid_until);
"""


@dataclass
class VersionedEntry:
    """A memory entry with validity annotations.

    Attributes:
        entry_id: Unique identifier for this entry version.
        key: The logical key (e.g. "topic:transformers").
        value: The stored content.
        source: Where this entry originated (run_id, paper_id, etc).
        valid_from: ISO-8601 timestamp when this entry became valid.
        valid_until: ISO-8601 timestamp when this entry expired.
            Empty string means currently valid.
        superseded_by: entry_id of the newer version, if any.
        version: Monotonically increasing version number for this key.
        metadata: Arbitrary additional data.
    """

    entry_id: str
    key: str
    value: str
    source: str = ""
    valid_from: str = ""
    valid_until: str = ""
    superseded_by: str = ""
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Whether this entry is currently valid (not superseded)."""
        return self.valid_until == "" and self.superseded_by == ""


class VersionedMemory:
    """Non-destructive memory store with validity tracking.

    New values for the same key create a new version while the
    previous version gets ``valid_until`` and ``superseded_by``
    annotations.  Active queries only return the latest version;
    full history is available via :meth:`get_history`.

    Args:
        db_path: Path to the SQLite database.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_VERSIONED_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def put(
        self,
        key: str,
        value: str,
        *,
        entry_id: str = "",
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> VersionedEntry:
        """Store a new value for ``key``, superseding the previous version.

        Args:
            key: The logical key.
            value: The content to store.
            entry_id: Custom entry ID. Auto-generated if empty.
            source: Origin identifier.
            metadata: Arbitrary additional data.

        Returns:
            The newly created :class:`VersionedEntry`.
        """
        now = datetime.now(tz=UTC).isoformat()
        conn = self._get_conn()

        # Find current active version for this key
        current = self._get_active_row(key)

        # Version uses MAX across all entries (including superseded)
        max_row = conn.execute(
            "SELECT MAX(version) FROM versioned_entries WHERE key = ?",
            (key,),
        ).fetchone()
        new_version = (max_row[0] or 0) + 1

        if not entry_id:
            entry_id = f"{key}:v{new_version}"

        entry = VersionedEntry(
            entry_id=entry_id,
            key=key,
            value=value,
            source=source,
            valid_from=now,
            valid_until="",
            superseded_by="",
            version=new_version,
            metadata=metadata or {},
        )

        # Supersede the old version
        if current is not None:
            conn.execute(
                "UPDATE versioned_entries SET valid_until = ?, superseded_by = ? "
                "WHERE entry_id = ?",
                (now, entry_id, current["entry_id"]),
            )

        conn.execute(
            "INSERT INTO versioned_entries "
            "(entry_id, key, value, source, valid_from, valid_until, "
            "superseded_by, version, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.entry_id,
                entry.key,
                entry.value,
                entry.source,
                entry.valid_from,
                entry.valid_until,
                entry.superseded_by,
                entry.version,
                json.dumps(entry.metadata),
            ),
        )
        conn.commit()

        logger.debug(
            "Versioned memory: put %s v%d (supersedes %s)",
            key,
            new_version,
            current["entry_id"] if current else "none",
        )
        return entry

    def get(self, key: str) -> VersionedEntry | None:
        """Get the current active entry for ``key``.

        Returns:
            The active entry, or ``None`` if no entry exists.
        """
        row = self._get_active_row(key)
        if row is None:
            return None
        return self._row_to_entry(row)

    def get_history(self, key: str) -> list[VersionedEntry]:
        """Get all versions of ``key``, newest first.

        Returns:
            List of all versions (active and superseded).
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM versioned_entries WHERE key = ? ORDER BY version DESC",
            (key,),
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_active_entries(self) -> list[VersionedEntry]:
        """Get all currently active entries.

        Returns:
            List of active (non-superseded) entries.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM versioned_entries "
            "WHERE valid_until = '' AND superseded_by = '' "
            "ORDER BY key",
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def count(self, *, active_only: bool = True) -> int:
        """Count entries.

        Args:
            active_only: If True, count only active entries.

        Returns:
            Number of entries.
        """
        conn = self._get_conn()
        if active_only:
            row = conn.execute(
                "SELECT COUNT(*) FROM versioned_entries "
                "WHERE valid_until = '' AND superseded_by = ''",
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM versioned_entries",
            ).fetchone()
        return row[0] if row else 0

    def rollback(self, key: str) -> VersionedEntry | None:
        """Rollback to the previous version of ``key``.

        Removes the current active version's ``superseded_by`` link
        from the previous version and marks the current as invalid.

        Returns:
            The restored previous entry, or None if no history.
        """
        history = self.get_history(key)
        if len(history) < 2:
            return None

        current = history[0]
        previous = history[1]
        now = datetime.now(tz=UTC).isoformat()
        conn = self._get_conn()

        # Mark current as invalid
        conn.execute(
            "UPDATE versioned_entries SET valid_until = ? WHERE entry_id = ?",
            (now, current.entry_id),
        )

        # Restore previous
        conn.execute(
            "UPDATE versioned_entries SET valid_until = '', superseded_by = '' "
            "WHERE entry_id = ?",
            (previous.entry_id,),
        )
        conn.commit()

        logger.debug("Versioned memory: rollback %s to v%d", key, previous.version)
        return self.get(key)

    def _get_active_row(self, key: str) -> sqlite3.Row | None:
        conn = self._get_conn()
        row: sqlite3.Row | None = conn.execute(
            "SELECT * FROM versioned_entries "
            "WHERE key = ? AND valid_until = '' AND superseded_by = '' "
            "ORDER BY version DESC LIMIT 1",
            (key,),
        ).fetchone()
        return row

    def _row_to_entry(self, row: sqlite3.Row) -> VersionedEntry:
        return VersionedEntry(
            entry_id=row["entry_id"],
            key=row["key"],
            value=row["value"],
            source=row["source"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            superseded_by=row["superseded_by"],
            version=row["version"],
            metadata=json.loads(row["metadata"]),
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
