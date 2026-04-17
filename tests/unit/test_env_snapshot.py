"""Tests for infra.env_snapshot — full environment snapshot capture."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from research_pipeline.infra.env_snapshot import (
    EnvironmentSnapshot,
    EnvironmentSnapshotCapture,
    FileEntry,
    SnapshotDiff,
    SystemInfo,
    _collect_dependency_versions,
    _collect_env_vars,
    _collect_system_info,
    _mask_sensitive,
    _sha256_file,
)

# ---------------------------------------------------------------------------
# SystemInfo
# ---------------------------------------------------------------------------


class TestSystemInfo:
    def test_creation(self) -> None:
        si = SystemInfo(
            platform="Linux-x86_64",
            python_version="3.12.0",
            cpu_count=8,
            total_memory_mb=1024.0,
            disk_free_mb=50000.0,
        )
        assert si.cpu_count == 8
        assert si.platform == "Linux-x86_64"

    def test_to_dict_roundtrip(self) -> None:
        si = SystemInfo("Linux", "3.12", 4, 512.0, 10000.0)
        d = si.to_dict()
        assert d["cpu_count"] == 4
        si2 = SystemInfo(**d)
        assert si2 == si

    def test_frozen(self) -> None:
        si = SystemInfo("Linux", "3.12", 4, 512.0, 10000.0)
        with pytest.raises(AttributeError):
            si.cpu_count = 16  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FileEntry
# ---------------------------------------------------------------------------


class TestFileEntry:
    def test_creation(self) -> None:
        fe = FileEntry("a/b.txt", 100, "abc123", True)
        assert fe.relative_path == "a/b.txt"
        assert fe.captured is True

    def test_to_dict(self) -> None:
        fe = FileEntry("x.md", 200, "def456", False)
        d = fe.to_dict()
        assert d["sha256"] == "def456"
        assert d["captured"] is False


# ---------------------------------------------------------------------------
# SnapshotDiff
# ---------------------------------------------------------------------------


class TestSnapshotDiff:
    def test_empty_diff(self) -> None:
        diff = SnapshotDiff()
        assert not diff.has_changes

    def test_has_changes_added(self) -> None:
        diff = SnapshotDiff(added_files=["new.txt"])
        assert diff.has_changes

    def test_has_changes_removed(self) -> None:
        diff = SnapshotDiff(removed_files=["old.txt"])
        assert diff.has_changes

    def test_has_changes_modified(self) -> None:
        diff = SnapshotDiff(modified_files=["changed.txt"])
        assert diff.has_changes

    def test_has_changes_config(self) -> None:
        diff = SnapshotDiff(config_changes={"backend": ("a", "b")})
        assert diff.has_changes

    def test_to_dict(self) -> None:
        diff = SnapshotDiff(
            added_files=["a.txt"],
            config_changes={"k": ("v1", "v2")},
        )
        d = diff.to_dict()
        assert d["added_files"] == ["a.txt"]
        assert d["config_changes"]["k"]["before"] == "v1"


# ---------------------------------------------------------------------------
# EnvironmentSnapshot
# ---------------------------------------------------------------------------


class TestEnvironmentSnapshot:
    def _make_snapshot(self, **kwargs: object) -> EnvironmentSnapshot:
        defaults: dict = {
            "snapshot_id": "test-post-2025",
            "stage": "screen",
            "label": "post",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "system_info": SystemInfo("Linux", "3.12", 4, 512.0, 10000.0),
            "config_digest": {"backend": "docling"},
            "dependency_versions": {"pydantic": "2.0"},
            "env_vars": {},
            "files": [FileEntry("a.txt", 10, "abc", True)],
            "total_size_bytes": 10,
            "file_count": 1,
        }
        defaults.update(kwargs)
        return EnvironmentSnapshot(**defaults)

    def test_to_dict_roundtrip(self) -> None:
        snap = self._make_snapshot()
        d = snap.to_dict()
        snap2 = EnvironmentSnapshot.from_dict(d)
        assert snap2.snapshot_id == snap.snapshot_id
        assert snap2.system_info == snap.system_info
        assert snap2.files[0].sha256 == "abc"

    def test_from_dict_missing_optional(self) -> None:
        d = {
            "snapshot_id": "x",
            "stage": "plan",
            "label": "pre",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "system_info": {
                "platform": "L",
                "python_version": "3",
                "cpu_count": 1,
                "total_memory_mb": None,
                "disk_free_mb": None,
            },
        }
        snap = EnvironmentSnapshot.from_dict(d)
        assert snap.files == []
        assert snap.dependency_versions == {}


# ---------------------------------------------------------------------------
# _mask_sensitive
# ---------------------------------------------------------------------------


class TestMaskSensitive:
    def test_masks_keys(self) -> None:
        cfg = {"api_key": "secret123", "backend": "docling"}
        masked = _mask_sensitive(cfg)
        assert masked["api_key"] == "***"
        assert masked["backend"] == "docling"

    def test_masks_nested(self) -> None:
        cfg = {"conversion": {"token": "xyz", "format": "md"}}
        masked = _mask_sensitive(cfg)
        assert masked["conversion"]["token"] == "***"
        assert masked["conversion"]["format"] == "md"

    def test_masks_password(self) -> None:
        cfg = {"db_password": "p@ss", "host": "localhost"}
        masked = _mask_sensitive(cfg)
        assert masked["db_password"] == "***"

    def test_empty(self) -> None:
        assert _mask_sensitive({}) == {}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_collect_system_info(self) -> None:
        si = _collect_system_info()
        assert si.python_version
        assert si.platform

    def test_collect_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCH_PIPELINE_BACKEND", "docling")
        monkeypatch.setenv("RESEARCH_PIPELINE_API_KEY", "secret")
        monkeypatch.setenv("OTHER_VAR", "ignored")
        result = _collect_env_vars()
        assert result["RESEARCH_PIPELINE_BACKEND"] == "docling"
        assert result["RESEARCH_PIPELINE_API_KEY"] == "***"
        assert "OTHER_VAR" not in result

    def test_collect_dependency_versions(self) -> None:
        versions = _collect_dependency_versions()
        # pydantic should be installed
        assert "pydantic" in versions

    def test_sha256_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = _sha256_file(f)
        assert len(h) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# EnvironmentSnapshotCapture
# ---------------------------------------------------------------------------


class TestEnvironmentSnapshotCapture:
    def _setup_stage_dir(self, tmp_path: Path) -> Path:
        """Create a fake stage directory with files."""
        stage_dir = tmp_path / "run" / "screen"
        stage_dir.mkdir(parents=True)
        (stage_dir / "candidates.jsonl").write_text('{"id": "1"}')
        (stage_dir / "scores.json").write_text('{"score": 0.9}')
        sub = stage_dir / "details"
        sub.mkdir()
        (sub / "meta.txt").write_text("metadata")
        return stage_dir

    def test_capture_basic(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        snap = cap.capture("screen", stage_dir, label="post")
        assert snap.stage == "screen"
        assert snap.label == "post"
        assert snap.file_count == 3
        assert snap.total_size_bytes > 0
        assert cap.snapshot_count == 1

    def test_capture_disabled(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root, enabled=False)
        snap = cap.capture("screen", stage_dir)
        assert snap.file_count == 0
        assert cap.snapshot_count == 1

    def test_capture_missing_source(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        run_root.mkdir(parents=True)
        cap = EnvironmentSnapshotCapture(run_root)
        snap = cap.capture("screen", tmp_path / "nonexistent")
        assert snap.file_count == 0

    def test_capture_with_config(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cfg = {"backend": "docling", "api_key": "secret123"}
        cap = EnvironmentSnapshotCapture(run_root, config=cfg)
        snap = cap.capture("screen", stage_dir)
        assert snap.config_digest["api_key"] == "***"
        assert snap.config_digest["backend"] == "docling"

    def test_capture_writes_manifest(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        snap = cap.capture("screen", stage_dir)
        manifest_path = cap.snapshot_dir / snap.snapshot_id / "_env_manifest.json"
        assert manifest_path.exists()
        loaded = json.loads(manifest_path.read_text())
        assert loaded["stage"] == "screen"

    def test_capture_compressed(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root, compress=True)
        snap = cap.capture("screen", stage_dir)
        gz_path = cap.snapshot_dir / snap.snapshot_id / "_env_manifest.json.gz"
        assert gz_path.exists()
        with gzip.open(gz_path, "rt") as fh:
            loaded = json.loads(fh.read())
        assert loaded["compressed"] is True

    def test_capture_large_file_skipped(self, tmp_path: Path) -> None:
        stage_dir = tmp_path / "run" / "screen"
        stage_dir.mkdir(parents=True)
        big = stage_dir / "big.bin"
        big.write_bytes(b"x" * 200)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root, max_file_size=100)
        snap = cap.capture("screen", stage_dir)
        assert snap.files[0].captured is False
        assert snap.files[0].sha256  # hash still computed

    def test_capture_pair(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        pre, post = cap.capture_pair("screen", stage_dir)
        assert pre.label == "pre"
        assert post.label == "post"
        assert cap.snapshot_count == 2

    def test_diff_no_changes(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        s1 = cap.capture("screen", stage_dir, label="pre")
        s2 = cap.capture("screen", stage_dir, label="post")
        diff = cap.diff(s1, s2)
        assert not diff.has_changes

    def test_diff_added_file(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        s1 = cap.capture("screen", stage_dir, label="pre")
        (stage_dir / "new_file.txt").write_text("new")
        s2 = cap.capture("screen", stage_dir, label="post")
        diff = cap.diff(s1, s2)
        assert "new_file.txt" in diff.added_files

    def test_diff_removed_file(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        s1 = cap.capture("screen", stage_dir, label="pre")
        (stage_dir / "scores.json").unlink()
        s2 = cap.capture("screen", stage_dir, label="post")
        diff = cap.diff(s1, s2)
        assert "scores.json" in diff.removed_files

    def test_diff_modified_file(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        s1 = cap.capture("screen", stage_dir, label="pre")
        (stage_dir / "scores.json").write_text('{"score": 0.5}')
        s2 = cap.capture("screen", stage_dir, label="post")
        diff = cap.diff(s1, s2)
        assert "scores.json" in diff.modified_files

    def test_diff_config_changes(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap1 = EnvironmentSnapshotCapture(run_root, config={"backend": "docling"})
        s1 = cap1.capture("screen", stage_dir, label="pre")

        # Simulate config change by creating snapshot with diff config
        s2_dict = s1.to_dict()
        s2_dict["config_digest"] = {"backend": "marker"}
        s2_dict["snapshot_id"] = "modified"
        s2 = EnvironmentSnapshot.from_dict(s2_dict)

        diff = cap1.diff(s1, s2)
        assert "backend" in diff.config_changes

    def test_diff_latest(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        cap.capture("screen", stage_dir, label="pre")
        (stage_dir / "extra.txt").write_text("x")
        cap.capture("screen", stage_dir, label="post")
        diff = cap.diff_latest()
        assert diff is not None
        assert diff.has_changes

    def test_diff_latest_insufficient(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        run_root.mkdir(parents=True)
        cap = EnvironmentSnapshotCapture(run_root)
        assert cap.diff_latest() is None

    def test_verify_integrity_ok(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        snap = cap.capture("screen", stage_dir)
        failures = cap.verify_integrity(snap)
        assert failures == []

    def test_verify_integrity_tampered(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        snap = cap.capture("screen", stage_dir)
        # Tamper with a captured file
        captured_file = cap.snapshot_dir / snap.snapshot_id / "candidates.jsonl"
        captured_file.write_text("tampered!")
        failures = cap.verify_integrity(snap)
        assert "candidates.jsonl" in failures

    def test_verify_integrity_missing(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        snap = cap.capture("screen", stage_dir)
        captured_file = cap.snapshot_dir / snap.snapshot_id / "candidates.jsonl"
        captured_file.unlink()
        failures = cap.verify_integrity(snap)
        assert "candidates.jsonl" in failures

    def test_list_snapshots(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        cap.capture("screen", stage_dir, label="pre")
        cap.capture("screen", stage_dir, label="post")
        ids = cap.list_snapshots()
        assert len(ids) == 2

    def test_get_snapshot(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        snap = cap.capture("screen", stage_dir)
        retrieved = cap.get_snapshot(snap.snapshot_id)
        assert retrieved is not None
        assert retrieved.stage == "screen"

    def test_get_snapshot_not_found(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        run_root.mkdir(parents=True)
        cap = EnvironmentSnapshotCapture(run_root)
        assert cap.get_snapshot("nonexistent") is None

    def test_get_stage_snapshots(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        cap.capture("screen", stage_dir)
        cap.capture("download", stage_dir)
        assert len(cap.get_stage_snapshots("screen")) == 1
        assert len(cap.get_stage_snapshots("download")) == 1

    def test_aggregate_stats(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        cap.capture("screen", stage_dir)
        cap.capture("download", stage_dir)
        stats = cap.compute_aggregate_stats()
        assert stats["total_snapshots"] == 2
        assert "screen" in stats["stages_covered"]
        assert stats["total_files_captured"] == 6  # 3 + 3

    def test_load_from_disk(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root)
        snap = cap.capture("screen", stage_dir)
        sid = snap.snapshot_id
        # Create a new capture instance (loses in-memory state)
        cap2 = EnvironmentSnapshotCapture(run_root)
        loaded = cap2.get_snapshot(sid)
        assert loaded is not None
        assert loaded.stage == "screen"
        assert loaded.file_count == 3

    def test_load_compressed_from_disk(self, tmp_path: Path) -> None:
        stage_dir = self._setup_stage_dir(tmp_path)
        run_root = tmp_path / "run"
        cap = EnvironmentSnapshotCapture(run_root, compress=True)
        snap = cap.capture("screen", stage_dir)
        sid = snap.snapshot_id
        cap2 = EnvironmentSnapshotCapture(run_root)
        loaded = cap2.get_snapshot(sid)
        assert loaded is not None
        assert loaded.compressed is True
