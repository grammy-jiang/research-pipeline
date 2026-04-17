"""Full environment snapshot capture at pipeline stage boundaries.

Goes beyond file-copy snapshots (:class:`SnapshotManager` in
``eval_logging``) by capturing the **complete environment context**:

* Resolved pipeline configuration (sanitised — API keys masked)
* Python dependency versions (``importlib.metadata``)
* System information (platform, CPU, memory, disk)
* Stage-level file manifest with SHA-256 integrity hashes
* Diff between consecutive snapshots (what changed)
* Compressed storage for space efficiency

Designed as Channel 3+ of the Claw-Eval three-channel evaluation
architecture.  Each snapshot is self-contained and can be used for
reproducibility audits, regression debugging, and compliance.

References:
    Claw-Eval three-channel framework (R6), Deep Research Report §6.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemInfo:
    """Immutable snapshot of the host system."""

    platform: str
    python_version: str
    cpu_count: int | None
    total_memory_mb: float | None
    disk_free_mb: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict."""
        return {
            "platform": self.platform,
            "python_version": self.python_version,
            "cpu_count": self.cpu_count,
            "total_memory_mb": self.total_memory_mb,
            "disk_free_mb": self.disk_free_mb,
        }


@dataclass(frozen=True)
class FileEntry:
    """Single file within a snapshot."""

    relative_path: str
    size_bytes: int
    sha256: str
    captured: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict."""
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "captured": self.captured,
        }


@dataclass
class SnapshotDiff:
    """Difference between two environment snapshots."""

    added_files: list[str] = field(default_factory=list)
    removed_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    config_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        """Whether any difference exists."""
        return bool(
            self.added_files
            or self.removed_files
            or self.modified_files
            or self.config_changes,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict."""
        return {
            "added_files": self.added_files,
            "removed_files": self.removed_files,
            "modified_files": self.modified_files,
            "config_changes": {
                k: {"before": v[0], "after": v[1]}
                for k, v in self.config_changes.items()
            },
            "has_changes": self.has_changes,
        }


@dataclass
class EnvironmentSnapshot:
    """Complete environment context at a point in time."""

    snapshot_id: str
    stage: str
    label: str  # "pre" or "post"
    timestamp: str
    system_info: SystemInfo
    config_digest: dict[str, Any]
    dependency_versions: dict[str, str]
    env_vars: dict[str, str]
    files: list[FileEntry]
    total_size_bytes: int
    file_count: int
    compressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict."""
        return {
            "snapshot_id": self.snapshot_id,
            "stage": self.stage,
            "label": self.label,
            "timestamp": self.timestamp,
            "system_info": self.system_info.to_dict(),
            "config_digest": self.config_digest,
            "dependency_versions": self.dependency_versions,
            "env_vars": self.env_vars,
            "files": [f.to_dict() for f in self.files],
            "total_size_bytes": self.total_size_bytes,
            "file_count": self.file_count,
            "compressed": self.compressed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentSnapshot:
        """Deserialise from a plain dict."""
        return cls(
            snapshot_id=data["snapshot_id"],
            stage=data["stage"],
            label=data["label"],
            timestamp=data["timestamp"],
            system_info=SystemInfo(**data["system_info"]),
            config_digest=data.get("config_digest", {}),
            dependency_versions=data.get("dependency_versions", {}),
            env_vars=data.get("env_vars", {}),
            files=[FileEntry(**f) for f in data.get("files", [])],
            total_size_bytes=data.get("total_size_bytes", 0),
            file_count=data.get("file_count", 0),
            compressed=data.get("compressed", False),
        )


# ---------------------------------------------------------------------------
# Sensitive key masking
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS = frozenset(
    {
        "api_key",
        "api_secret",
        "token",
        "password",
        "secret",
        "credential",
        "auth",
        "serpapi_key",
    }
)


def _mask_sensitive(config: dict[str, Any]) -> dict[str, Any]:
    """Return *config* with sensitive values replaced by ``***``."""
    masked: dict[str, Any] = {}
    for key, value in config.items():
        lower = key.lower()
        if any(pat in lower for pat in _SENSITIVE_PATTERNS):
            masked[key] = "***"
        elif isinstance(value, dict):
            masked[key] = _mask_sensitive(value)
        else:
            masked[key] = value
    return masked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PIPELINE_ENV_PREFIX = "RESEARCH_PIPELINE_"


def _collect_env_vars() -> dict[str, str]:
    """Collect pipeline-relevant environment variables (masked)."""
    result: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if key.startswith(_PIPELINE_ENV_PREFIX):
            lower = key.lower()
            if any(pat in lower for pat in _SENSITIVE_PATTERNS):
                result[key] = "***"
            else:
                result[key] = value
    return result


def _collect_system_info(run_root: Path | None = None) -> SystemInfo:
    """Gather current host system metrics."""
    cpu = os.cpu_count()
    try:
        import resource  # Unix only

        total_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, AttributeError):
        total_mem_mb = None

    disk_free: float | None = None
    try:
        check_path = run_root or Path.cwd()
        stat = shutil.disk_usage(check_path)
        disk_free = stat.free / (1024 * 1024)
    except OSError:
        pass

    return SystemInfo(
        platform=platform.platform(),
        python_version=sys.version,
        cpu_count=cpu,
        total_memory_mb=total_mem_mb,
        disk_free_mb=round(disk_free, 1) if disk_free is not None else None,
    )


def _collect_dependency_versions() -> dict[str, str]:
    """Snapshot installed package versions for key dependencies."""
    import contextlib
    from importlib.metadata import PackageNotFoundError, version

    packages = [
        "research-pipeline",
        "pydantic",
        "typer",
        "httpx",
        "rank-bm25",
        "fastmcp",
        "docling",
        "marker-pdf",
        "pymupdf4llm",
        "torch",
        "transformers",
        "sentence-transformers",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        with contextlib.suppress(PackageNotFoundError):
            versions[pkg] = version(pkg)
    return versions


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main capture class
# ---------------------------------------------------------------------------


class EnvironmentSnapshotCapture:
    """Captures full environment snapshots at stage boundaries.

    Args:
        run_root: Root directory of the pipeline run.
        enabled: Master switch — when ``False`` all calls are no-ops.
        compress: If ``True``, gzip the snapshot manifest.
        max_file_size: Files larger than this are hashed but not copied.
        config: Resolved pipeline config dict (will be sanitised).
    """

    DEFAULT_MAX_SIZE = 10 * 1024 * 1024  # 10 MB

    def __init__(
        self,
        run_root: Path,
        *,
        enabled: bool = True,
        compress: bool = False,
        max_file_size: int = DEFAULT_MAX_SIZE,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._run_root = run_root
        self._enabled = enabled
        self._compress = compress
        self._max_file_size = max_file_size
        self._config = _mask_sensitive(config) if config else {}
        self._snapshot_dir = run_root / "env_snapshots"
        self._snapshots: list[EnvironmentSnapshot] = []

    # -- Properties ----------------------------------------------------------

    @property
    def snapshot_dir(self) -> Path:
        """Base directory for all environment snapshots."""
        return self._snapshot_dir

    @property
    def enabled(self) -> bool:
        """Whether capture is active."""
        return self._enabled

    @property
    def snapshot_count(self) -> int:
        """Number of snapshots taken."""
        return len(self._snapshots)

    @property
    def snapshots(self) -> list[EnvironmentSnapshot]:
        """All captured snapshots in order."""
        return list(self._snapshots)

    # -- Capture -------------------------------------------------------------

    def capture(
        self,
        stage: str,
        source_dir: Path,
        *,
        label: str = "post",
    ) -> EnvironmentSnapshot:
        """Capture a full environment snapshot.

        Args:
            stage: Pipeline stage name (e.g. ``"screen"``).
            source_dir: Directory whose files are recorded.
            label: ``"pre"`` for before-execution, ``"post"`` for after.

        Returns:
            The :class:`EnvironmentSnapshot` (even when disabled —
            fields will be empty).
        """
        now = datetime.now(UTC).isoformat()
        snap_id = f"{stage}-{label}-{now.replace(':', '-')}"

        system_info = _collect_system_info(self._run_root)
        dep_versions = _collect_dependency_versions()
        env_vars = _collect_env_vars()

        files: list[FileEntry] = []
        total_size = 0

        if self._enabled and source_dir.exists():
            dest = self._snapshot_dir / snap_id
            try:
                dest.mkdir(parents=True, exist_ok=True)

                for src_file in sorted(source_dir.rglob("*")):
                    if not src_file.is_file():
                        continue
                    rel = str(src_file.relative_to(source_dir))
                    size = src_file.stat().st_size
                    sha = _sha256_file(src_file)
                    total_size += size

                    captured = size <= self._max_file_size
                    if captured:
                        dst = dest / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(src_file), str(dst))

                    files.append(
                        FileEntry(
                            relative_path=rel,
                            size_bytes=size,
                            sha256=sha,
                            captured=captured,
                        )
                    )

            except OSError:
                logger.warning(
                    "Failed to capture env snapshot for stage=%s label=%s",
                    stage,
                    label,
                    exc_info=True,
                )

        snapshot = EnvironmentSnapshot(
            snapshot_id=snap_id,
            stage=stage,
            label=label,
            timestamp=now,
            system_info=system_info,
            config_digest=self._config,
            dependency_versions=dep_versions,
            env_vars=env_vars,
            files=files,
            total_size_bytes=total_size,
            file_count=len(files),
            compressed=self._compress,
        )

        # Persist manifest
        if self._enabled:
            self._write_manifest(snapshot)

        self._snapshots.append(snapshot)
        return snapshot

    def capture_pair(
        self,
        stage: str,
        source_dir: Path,
    ) -> tuple[EnvironmentSnapshot, EnvironmentSnapshot]:
        """Convenience: capture both pre and post snapshots.

        Typically called before and after stage execution, but can
        be used to snapshot the same directory twice for diffing.

        Returns:
            Tuple of (pre_snapshot, post_snapshot).
        """
        pre = self.capture(stage, source_dir, label="pre")
        post = self.capture(stage, source_dir, label="post")
        return pre, post

    # -- Diff ----------------------------------------------------------------

    def diff(
        self,
        before: EnvironmentSnapshot,
        after: EnvironmentSnapshot,
    ) -> SnapshotDiff:
        """Compute the difference between two snapshots.

        Args:
            before: The earlier snapshot.
            after: The later snapshot.

        Returns:
            :class:`SnapshotDiff` with added / removed / modified files
            and config changes.
        """
        before_map = {f.relative_path: f for f in before.files}
        after_map = {f.relative_path: f for f in after.files}

        before_keys = set(before_map.keys())
        after_keys = set(after_map.keys())

        added = sorted(after_keys - before_keys)
        removed = sorted(before_keys - after_keys)
        modified = sorted(
            p
            for p in before_keys & after_keys
            if before_map[p].sha256 != after_map[p].sha256
        )

        config_changes: dict[str, tuple[Any, Any]] = {}
        all_keys = set(before.config_digest) | set(after.config_digest)
        for key in sorted(all_keys):
            bv = before.config_digest.get(key)
            av = after.config_digest.get(key)
            if bv != av:
                config_changes[key] = (bv, av)

        return SnapshotDiff(
            added_files=added,
            removed_files=removed,
            modified_files=modified,
            config_changes=config_changes,
        )

    def diff_latest(self) -> SnapshotDiff | None:
        """Diff the two most recent snapshots (if available)."""
        if len(self._snapshots) < 2:
            return None
        return self.diff(self._snapshots[-2], self._snapshots[-1])

    # -- Verification --------------------------------------------------------

    def verify_integrity(
        self,
        snapshot: EnvironmentSnapshot,
    ) -> list[str]:
        """Verify SHA-256 integrity of captured files.

        Args:
            snapshot: Snapshot to verify.

        Returns:
            List of file paths that failed verification (empty = OK).
        """
        failures: list[str] = []
        snap_dir = self._snapshot_dir / snapshot.snapshot_id
        for fe in snapshot.files:
            if not fe.captured:
                continue
            fpath = snap_dir / fe.relative_path
            if not fpath.exists():
                failures.append(fe.relative_path)
                continue
            actual = _sha256_file(fpath)
            if actual != fe.sha256:
                failures.append(fe.relative_path)
        return failures

    # -- Listing / retrieval ------------------------------------------------

    def list_snapshots(self) -> list[str]:
        """Return snapshot IDs in capture order."""
        return [s.snapshot_id for s in self._snapshots]

    def get_snapshot(self, snapshot_id: str) -> EnvironmentSnapshot | None:
        """Retrieve a snapshot by ID."""
        for s in self._snapshots:
            if s.snapshot_id == snapshot_id:
                return s
        # Try loading from disk
        return self._load_manifest(snapshot_id)

    def get_stage_snapshots(self, stage: str) -> list[EnvironmentSnapshot]:
        """Return all snapshots for a given stage."""
        return [s for s in self._snapshots if s.stage == stage]

    def compute_aggregate_stats(self) -> dict[str, Any]:
        """Aggregate statistics across all snapshots.

        Returns:
            Dict with total_snapshots, stages_covered,
            total_files_captured, total_bytes, integrity_ok.
        """
        stages = set()
        total_files = 0
        total_bytes = 0
        for s in self._snapshots:
            stages.add(s.stage)
            total_files += s.file_count
            total_bytes += s.total_size_bytes
        return {
            "total_snapshots": len(self._snapshots),
            "stages_covered": sorted(stages),
            "total_files_captured": total_files,
            "total_bytes": total_bytes,
        }

    # -- Persistence ---------------------------------------------------------

    def _write_manifest(self, snapshot: EnvironmentSnapshot) -> None:
        """Write snapshot manifest to disk."""
        dest = self._snapshot_dir / snapshot.snapshot_id
        dest.mkdir(parents=True, exist_ok=True)

        data = json.dumps(snapshot.to_dict(), indent=2, default=str)

        if self._compress:
            path = dest / "_env_manifest.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                fh.write(data)
        else:
            path = dest / "_env_manifest.json"
            path.write_text(data, encoding="utf-8")

    def _load_manifest(self, snapshot_id: str) -> EnvironmentSnapshot | None:
        """Load a snapshot manifest from disk."""
        dest = self._snapshot_dir / snapshot_id

        gz_path = dest / "_env_manifest.json.gz"
        plain_path = dest / "_env_manifest.json"

        raw: str | None = None
        if gz_path.exists():
            with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
                raw = fh.read()
        elif plain_path.exists():
            raw = plain_path.read_text(encoding="utf-8")

        if raw is None:
            return None
        return EnvironmentSnapshot.from_dict(json.loads(raw))
