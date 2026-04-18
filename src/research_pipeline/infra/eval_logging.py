"""Three-channel evaluation logging (Claw-Eval inspired).

Implements three independent evaluation channels:

1. **ExecutionTracer** — Structured JSONL traces of pipeline execution
   with timing, causality links, and stage context.
2. **AuditDB** — SQLite audit database for queryable who/what/when
   records across runs.
3. **SnapshotManager** — Filesystem state captures at stage boundaries
   for reproducibility and debugging.

These three channels provide comprehensive observability into pipeline
behavior, enabling post-hoc analysis and debugging.

Usage::

    from research_pipeline.infra.eval_logging import EvalLogger

    eval_log = EvalLogger(run_root)
    eval_log.trace("screen_started", stage="screen", data={"n": 200})
    eval_log.audit("screen", "filter", input_hash="abc", output_hash="def")
    eval_log.snapshot("screen", run_root / "screen")
    summary = eval_log.summary()
"""

import contextlib
import json
import logging
import shutil
import sqlite3
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TraceLevel(StrEnum):
    """Trace event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ExecutionTracer:
    """Channel 1: Structured JSONL execution traces.

    Writes trace events to ``<run_root>/logs/traces.jsonl``.
    Each event includes a unique trace_id, timestamps, parent
    linkage for causality tracking, and arbitrary data payloads.

    Args:
        run_root: Root directory of the pipeline run.
        enabled: When ``False`` all calls are no-ops.
    """

    def __init__(self, run_root: Path, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._log_dir = run_root / "logs"
        self._log_path = self._log_dir / "traces.jsonl"
        self._trace_count = 0

    @property
    def log_path(self) -> Path:
        """Path to the traces JSONL file."""
        return self._log_path

    @property
    def enabled(self) -> bool:
        """Whether tracing is active."""
        return self._enabled

    @property
    def trace_count(self) -> int:
        """Number of traces emitted in this session."""
        return self._trace_count

    def emit(
        self,
        event: str,
        *,
        stage: str = "",
        level: TraceLevel | str = TraceLevel.INFO,
        parent_id: str = "",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a structured execution trace event.

        Args:
            event: Event name (e.g. ``"stage_started"``).
            stage: Pipeline stage name.
            level: Severity level.
            parent_id: ID of the parent trace for causality.
            data: Arbitrary event payload.

        Returns:
            The trace dict that was written.
        """
        trace_id = uuid.uuid4().hex[:12]
        level_str = level.value if isinstance(level, TraceLevel) else str(level)

        record: dict[str, Any] = {
            "trace_id": trace_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "level": level_str,
        }
        if stage:
            record["stage"] = stage
        if parent_id:
            record["parent_id"] = parent_id
        if data:
            record["data"] = data

        if self._enabled:
            self._write(record)
            self._trace_count += 1

        return record

    def _write(self, record: dict[str, Any]) -> None:
        """Append record as a JSONL line."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.warning("Failed to write trace event", exc_info=True)

    def read_traces(
        self,
        *,
        stage: str = "",
        level: str = "",
    ) -> list[dict[str, Any]]:
        """Read traces with optional filtering.

        Args:
            stage: Filter by stage name (empty = all).
            level: Filter by level (empty = all).

        Returns:
            List of trace dicts in chronological order.
        """
        if not self._log_path.exists():
            return []
        traces: list[dict[str, Any]] = []
        with self._log_path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if stage and record.get("stage") != stage:
                    continue
                if level and record.get("level") != level:
                    continue
                traces.append(record)
        return traces


class AuditDB:
    """Channel 2: SQLite audit database.

    Stores structured audit records in
    ``<run_root>/logs/audit.db`` for queryable who/what/when
    analysis across pipeline runs.

    Args:
        run_root: Root directory of the pipeline run.
        enabled: When ``False`` all calls are no-ops.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            run_id TEXT NOT NULL DEFAULT '',
            stage TEXT NOT NULL DEFAULT '',
            action TEXT NOT NULL DEFAULT '',
            input_hash TEXT NOT NULL DEFAULT '',
            output_hash TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            tokens_used INTEGER NOT NULL DEFAULT 0,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            details TEXT NOT NULL DEFAULT '{}'
        )
    """

    _INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_audit_stage ON audit_log(stage)"

    _INDEX_RUN_SQL = "CREATE INDEX IF NOT EXISTS idx_audit_run ON audit_log(run_id)"

    def __init__(self, run_root: Path, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._log_dir = run_root / "logs"
        self._db_path = self._log_dir / "audit.db"
        self._conn: sqlite3.Connection | None = None

    @property
    def db_path(self) -> Path:
        """Path to the SQLite audit database."""
        return self._db_path

    @property
    def enabled(self) -> bool:
        """Whether audit DB is active."""
        return self._enabled

    def _ensure_db(self) -> sqlite3.Connection:
        """Initialize DB connection and schema on first use."""
        if self._conn is not None:
            return self._conn
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute(self._SCHEMA)
            self._conn.execute(self._INDEX_SQL)
            self._conn.execute(self._INDEX_RUN_SQL)
            self._conn.commit()
        except (OSError, sqlite3.Error):
            logger.warning("Failed to initialize audit DB", exc_info=True)
            raise
        return self._conn

    def record(
        self,
        stage: str,
        action: str,
        *,
        run_id: str = "",
        input_hash: str = "",
        output_hash: str = "",
        model: str = "",
        tokens_used: int = 0,
        duration_ms: int = 0,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Insert an audit record.

        Args:
            stage: Pipeline stage.
            action: Action performed.
            run_id: Pipeline run ID.
            input_hash: SHA-256 of input data.
            output_hash: SHA-256 of output data.
            model: LLM model used (if any).
            tokens_used: Token count (if LLM).
            duration_ms: Duration in milliseconds.
            details: Extra JSON-serializable data.

        Returns:
            Row ID of the inserted record.
        """
        if not self._enabled:
            return 0

        try:
            conn = self._ensure_db()
            cursor = conn.execute(
                "INSERT INTO audit_log "
                "(run_id, stage, action, input_hash, output_hash, "
                "model, tokens_used, duration_ms, details) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    stage,
                    action,
                    input_hash,
                    output_hash,
                    model,
                    tokens_used,
                    duration_ms,
                    json.dumps(details or {}, default=str),
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0
        except (OSError, sqlite3.Error):
            logger.warning("Failed to write audit record", exc_info=True)
            return 0

    def query(
        self,
        *,
        stage: str = "",
        action: str = "",
        run_id: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit records with optional filters.

        Args:
            stage: Filter by stage.
            action: Filter by action.
            run_id: Filter by run ID.
            limit: Maximum records to return.

        Returns:
            List of audit record dicts.
        """
        try:
            conn = self._ensure_db()
        except (OSError, sqlite3.Error):
            return []

        conditions: list[str] = []
        params: list[Any] = []

        if stage:
            conditions.append("stage = ?")
            params.append(stage)
        if action:
            conditions.append("action = ?")
            params.append(action)
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)

        rows = conn.execute(
            f"SELECT * FROM audit_log{where} "  # nosec B608
            f"ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()

        columns = [
            "id",
            "timestamp",
            "run_id",
            "stage",
            "action",
            "input_hash",
            "output_hash",
            "model",
            "tokens_used",
            "duration_ms",
            "details",
        ]
        results: list[dict[str, Any]] = []
        for row in rows:
            record = dict(zip(columns, row, strict=False))
            if isinstance(record.get("details"), str):
                with contextlib.suppress(json.JSONDecodeError):
                    record["details"] = json.loads(record["details"])
            results.append(record)
        return results

    def count(self, *, stage: str = "", run_id: str = "") -> int:
        """Count audit records with optional filters.

        Args:
            stage: Filter by stage.
            run_id: Filter by run ID.

        Returns:
            Number of matching records.
        """
        try:
            conn = self._ensure_db()
        except (OSError, sqlite3.Error):
            return 0

        conditions: list[str] = []
        params: list[Any] = []

        if stage:
            conditions.append("stage = ?")
            params.append(stage)
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        row = conn.execute(
            f"SELECT COUNT(*) FROM audit_log{where}",  # nosec B608
            params,
        ).fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class SnapshotManager:
    """Channel 3: Filesystem environment snapshots.

    Captures stage input/output state at stage boundaries for
    reproducibility and post-hoc debugging. Stored under
    ``<run_root>/snapshots/<stage>/``.

    Args:
        run_root: Root directory of the pipeline run.
        enabled: When ``False`` all calls are no-ops.
        max_file_size: Maximum file size to include in snapshot
            (bytes). Files larger than this are recorded as metadata
            only. Defaults to 10 MB.
    """

    _DEFAULT_MAX_SIZE = 10 * 1024 * 1024  # 10 MB

    def __init__(
        self,
        run_root: Path,
        *,
        enabled: bool = True,
        max_file_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        self._enabled = enabled
        self._snapshot_dir = run_root / "snapshots"
        self._max_file_size = max_file_size
        self._snapshot_count = 0

    @property
    def snapshot_dir(self) -> Path:
        """Base directory for all snapshots."""
        return self._snapshot_dir

    @property
    def enabled(self) -> bool:
        """Whether snapshots are active."""
        return self._enabled

    @property
    def snapshot_count(self) -> int:
        """Number of snapshots captured."""
        return self._snapshot_count

    def capture(
        self,
        stage: str,
        source_dir: Path,
        *,
        label: str = "",
    ) -> dict[str, Any]:
        """Capture a filesystem snapshot of a stage directory.

        Args:
            stage: Stage name (e.g. ``"screen"``).
            source_dir: Directory to snapshot.
            label: Optional label (e.g. ``"pre"`` or ``"post"``).

        Returns:
            Snapshot metadata dict with file manifest.
        """
        suffix = f"-{label}" if label else ""
        snap_name = f"{stage}{suffix}"
        dest = self._snapshot_dir / snap_name

        metadata: dict[str, Any] = {
            "stage": stage,
            "label": label,
            "timestamp": datetime.now(UTC).isoformat(),
            "source": str(source_dir),
            "files": [],
        }

        if not self._enabled:
            return metadata

        if not source_dir.exists():
            metadata["error"] = "source_dir_not_found"
            return metadata

        try:
            dest.mkdir(parents=True, exist_ok=True)

            file_manifest: list[dict[str, Any]] = []
            for src_file in sorted(source_dir.rglob("*")):
                if not src_file.is_file():
                    continue

                rel = src_file.relative_to(source_dir)
                file_info: dict[str, Any] = {
                    "path": str(rel),
                    "size": src_file.stat().st_size,
                }

                if src_file.stat().st_size <= self._max_file_size:
                    dst_file = dest / rel
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src_file), str(dst_file))
                    file_info["captured"] = True
                else:
                    file_info["captured"] = False
                    file_info["reason"] = "exceeds_max_size"

                file_manifest.append(file_info)

            metadata["files"] = file_manifest
            metadata["file_count"] = len(file_manifest)
            metadata["total_size"] = sum(f["size"] for f in file_manifest)

            # Write manifest
            manifest_path = dest / "_snapshot_manifest.json"
            manifest_path.write_text(
                json.dumps(metadata, indent=2, default=str),
                encoding="utf-8",
            )

            self._snapshot_count += 1

        except OSError:
            logger.warning(
                "Failed to capture snapshot for %s",
                stage,
                exc_info=True,
            )
            metadata["error"] = "capture_failed"

        return metadata

    def list_snapshots(self) -> list[str]:
        """List all snapshot names.

        Returns:
            Sorted list of snapshot directory names.
        """
        if not self._snapshot_dir.exists():
            return []
        return sorted(d.name for d in self._snapshot_dir.iterdir() if d.is_dir())

    def get_manifest(self, snapshot_name: str) -> dict[str, Any] | None:
        """Read the manifest of a specific snapshot.

        Args:
            snapshot_name: Name of the snapshot directory.

        Returns:
            Manifest dict or ``None`` if not found.
        """
        manifest_path = self._snapshot_dir / snapshot_name / "_snapshot_manifest.json"
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


class EvalLogger:
    """Unified three-channel evaluation logger.

    Combines :class:`ExecutionTracer`, :class:`AuditDB`, and
    :class:`SnapshotManager` into a single facade.

    Args:
        run_root: Root directory of the pipeline run.
        enabled: Master switch — when ``False`` all channels
            are disabled.
        traces_enabled: Override for execution traces.
        audit_enabled: Override for audit DB.
        snapshots_enabled: Override for snapshots.
        max_snapshot_size: Maximum file size for snapshots.
    """

    def __init__(
        self,
        run_root: Path,
        *,
        enabled: bool = True,
        traces_enabled: bool | None = None,
        audit_enabled: bool | None = None,
        snapshots_enabled: bool | None = None,
        max_snapshot_size: int = SnapshotManager._DEFAULT_MAX_SIZE,
    ) -> None:
        t_enabled = traces_enabled if traces_enabled is not None else enabled
        a_enabled = audit_enabled if audit_enabled is not None else enabled
        s_enabled = snapshots_enabled if snapshots_enabled is not None else enabled

        self.tracer = ExecutionTracer(run_root, enabled=t_enabled)
        self.audit = AuditDB(run_root, enabled=a_enabled)
        self.snapshots = SnapshotManager(
            run_root,
            enabled=s_enabled,
            max_file_size=max_snapshot_size,
        )
        self._run_root = run_root

    @property
    def run_root(self) -> Path:
        """Root directory of the pipeline run."""
        return self._run_root

    def trace(
        self,
        event: str,
        *,
        stage: str = "",
        level: TraceLevel | str = TraceLevel.INFO,
        parent_id: str = "",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emit an execution trace event (Channel 1).

        Delegates to :meth:`ExecutionTracer.emit`.
        """
        return self.tracer.emit(
            event,
            stage=stage,
            level=level,
            parent_id=parent_id,
            data=data,
        )

    def record_audit(
        self,
        stage: str,
        action: str,
        *,
        run_id: str = "",
        input_hash: str = "",
        output_hash: str = "",
        model: str = "",
        tokens_used: int = 0,
        duration_ms: int = 0,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Insert an audit record (Channel 2).

        Delegates to :meth:`AuditDB.record`.
        """
        return self.audit.record(
            stage,
            action,
            run_id=run_id,
            input_hash=input_hash,
            output_hash=output_hash,
            model=model,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            details=details,
        )

    def snapshot(
        self,
        stage: str,
        source_dir: Path,
        *,
        label: str = "",
    ) -> dict[str, Any]:
        """Capture a stage snapshot (Channel 3).

        Delegates to :meth:`SnapshotManager.capture`.
        """
        return self.snapshots.capture(stage, source_dir, label=label)

    def summary(self) -> dict[str, Any]:
        """Generate a summary of all three channels.

        Returns:
            Dict with counts and paths for each channel.
        """
        return {
            "traces": {
                "path": str(self.tracer.log_path),
                "count": self.tracer.trace_count,
                "enabled": self.tracer.enabled,
            },
            "audit_db": {
                "path": str(self.audit.db_path),
                "total_records": self.audit.count(),
                "enabled": self.audit.enabled,
            },
            "snapshots": {
                "path": str(self.snapshots.snapshot_dir),
                "count": self.snapshots.snapshot_count,
                "snapshots": self.snapshots.list_snapshots(),
                "enabled": self.snapshots.enabled,
            },
        }

    def close(self) -> None:
        """Release resources (close audit DB connection)."""
        self.audit.close()
