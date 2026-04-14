"""Tests for research_pipeline.pipeline.checkpoint."""

import json
from pathlib import Path

from research_pipeline.infra.hashing import sha256_file
from research_pipeline.pipeline.checkpoint import (
    StageCheckpoint,
    compute_stage_hashes,
    is_stage_valid,
    load_all_checkpoints,
    load_checkpoint,
    write_checkpoint,
)

# ── helpers ──────────────────────────────────────────────────────────


def _make_file(directory: Path, name: str, content: str) -> Path:
    """Create a file with known content and return its path."""
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


# ── StageCheckpoint roundtrip ────────────────────────────────────────


class TestStageCheckpoint:
    """Tests for StageCheckpoint serialization."""

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        cp = StageCheckpoint(
            stage="search",
            status="completed",
            started_at="2025-01-01T00:00:00",
            ended_at="2025-01-01T00:01:00",
            duration_ms=60000,
            artifact_count=3,
            artifact_hashes={"a.json": "abc123", "b.json": "def456"},
            errors=["warning: slow"],
        )
        data = cp.to_dict()
        restored = StageCheckpoint.from_dict(data)

        assert restored.stage == cp.stage
        assert restored.status == cp.status
        assert restored.started_at == cp.started_at
        assert restored.ended_at == cp.ended_at
        assert restored.duration_ms == cp.duration_ms
        assert restored.artifact_count == cp.artifact_count
        assert restored.artifact_hashes == cp.artifact_hashes
        assert restored.errors == cp.errors

    def test_from_dict_defaults(self) -> None:
        cp = StageCheckpoint.from_dict({})
        assert cp.stage == ""
        assert cp.status == "pending"
        assert cp.artifact_hashes == {}
        assert cp.errors == []


# ── compute_stage_hashes ─────────────────────────────────────────────


class TestComputeStageHashes:
    """Tests for compute_stage_hashes."""

    def test_hashes_real_files(self, tmp_path: Path) -> None:
        f1 = _make_file(tmp_path, "a.txt", "hello")
        f2 = _make_file(tmp_path, "b.txt", "world")

        hashes = compute_stage_hashes([f1, f2])

        assert set(hashes.keys()) == {"a.txt", "b.txt"}
        assert hashes["a.txt"] == sha256_file(f1)
        assert hashes["b.txt"] == sha256_file(f2)

    def test_skips_nonexistent_files(self, tmp_path: Path) -> None:
        existing = _make_file(tmp_path, "exists.txt", "data")
        missing = tmp_path / "missing.txt"

        hashes = compute_stage_hashes([existing, missing])

        assert "exists.txt" in hashes
        assert "missing.txt" not in hashes

    def test_skips_directories(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        hashes = compute_stage_hashes([subdir])

        assert hashes == {}


# ── write_checkpoint ─────────────────────────────────────────────────


class TestWriteCheckpoint:
    """Tests for write_checkpoint."""

    def test_creates_file_in_checkpoints_dir(self, tmp_path: Path) -> None:
        artifact = _make_file(tmp_path, "out.json", '{"key": "value"}')

        write_checkpoint(
            run_root=tmp_path,
            stage="search",
            status="completed",
            started_at="2025-01-01T00:00:00",
            output_paths=[artifact],
        )

        cp_path = tmp_path / "checkpoints" / "search.checkpoint.json"
        assert cp_path.exists()

    def test_stores_correct_fields(self, tmp_path: Path) -> None:
        artifact = _make_file(tmp_path, "out.json", "content")

        cp = write_checkpoint(
            run_root=tmp_path,
            stage="download",
            status="completed",
            started_at="2025-06-01T12:00:00",
            output_paths=[artifact],
            errors=["retry once"],
        )

        assert cp.stage == "download"
        assert cp.status == "completed"
        assert cp.started_at == "2025-06-01T12:00:00"
        assert cp.ended_at != ""
        assert cp.artifact_count == 1
        assert "out.json" in cp.artifact_hashes
        assert cp.errors == ["retry once"]

    def test_hashes_match_sha256(self, tmp_path: Path) -> None:
        artifact = _make_file(tmp_path, "paper.md", "# Title\nBody text")

        cp = write_checkpoint(
            run_root=tmp_path,
            stage="convert",
            status="completed",
            started_at="2025-01-01T00:00:00",
            output_paths=[artifact],
        )

        assert cp.artifact_hashes["paper.md"] == sha256_file(artifact)


# ── load_checkpoint ──────────────────────────────────────────────────


class TestLoadCheckpoint:
    """Tests for load_checkpoint."""

    def test_roundtrip_write_then_load(self, tmp_path: Path) -> None:
        artifact = _make_file(tmp_path, "data.json", '{"x": 1}')
        written = write_checkpoint(
            run_root=tmp_path,
            stage="screen",
            status="completed",
            started_at="2025-01-01T00:00:00",
            output_paths=[artifact],
        )

        loaded = load_checkpoint(tmp_path, "screen")

        assert loaded is not None
        assert loaded.stage == written.stage
        assert loaded.status == written.status
        assert loaded.artifact_hashes == written.artifact_hashes

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        assert load_checkpoint(tmp_path, "nonexistent") is None

    def test_returns_none_for_corrupt_json(self, tmp_path: Path) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        (cp_dir / "bad.checkpoint.json").write_text("NOT JSON", encoding="utf-8")

        assert load_checkpoint(tmp_path, "bad") is None


# ── is_stage_valid ───────────────────────────────────────────────────


class TestIsStageValid:
    """Tests for is_stage_valid."""

    def test_true_when_hashes_match(self, tmp_path: Path) -> None:
        artifact = _make_file(tmp_path, "result.json", "stable content")
        write_checkpoint(
            run_root=tmp_path,
            stage="extract",
            status="completed",
            started_at="2025-01-01T00:00:00",
            output_paths=[artifact],
        )

        assert is_stage_valid(tmp_path, "extract", [artifact]) is True

    def test_false_when_hash_mismatch(self, tmp_path: Path) -> None:
        artifact = _make_file(tmp_path, "result.json", "original")
        write_checkpoint(
            run_root=tmp_path,
            stage="extract",
            status="completed",
            started_at="2025-01-01T00:00:00",
            output_paths=[artifact],
        )
        # Mutate the artifact after checkpoint
        artifact.write_text("modified", encoding="utf-8")

        assert is_stage_valid(tmp_path, "extract", [artifact]) is False

    def test_false_when_no_checkpoint(self, tmp_path: Path) -> None:
        artifact = _make_file(tmp_path, "result.json", "data")

        assert is_stage_valid(tmp_path, "extract", [artifact]) is False

    def test_false_when_status_failed(self, tmp_path: Path) -> None:
        artifact = _make_file(tmp_path, "result.json", "data")
        write_checkpoint(
            run_root=tmp_path,
            stage="download",
            status="failed",
            started_at="2025-01-01T00:00:00",
            output_paths=[artifact],
            errors=["network timeout"],
        )

        assert is_stage_valid(tmp_path, "download", [artifact]) is False

    def test_true_when_no_current_artifacts(self, tmp_path: Path) -> None:
        """Trust checkpoint when there are no current artifacts to verify."""
        artifact = _make_file(tmp_path, "result.json", "data")
        write_checkpoint(
            run_root=tmp_path,
            stage="plan",
            status="completed",
            started_at="2025-01-01T00:00:00",
            output_paths=[artifact],
        )

        assert is_stage_valid(tmp_path, "plan", []) is True


# ── load_all_checkpoints ─────────────────────────────────────────────


class TestLoadAllCheckpoints:
    """Tests for load_all_checkpoints."""

    def test_loads_multiple(self, tmp_path: Path) -> None:
        f1 = _make_file(tmp_path, "a.json", "aaa")
        f2 = _make_file(tmp_path, "b.json", "bbb")

        write_checkpoint(
            run_root=tmp_path,
            stage="plan",
            status="completed",
            started_at="2025-01-01T00:00:00",
            output_paths=[f1],
        )
        write_checkpoint(
            run_root=tmp_path,
            stage="search",
            status="completed",
            started_at="2025-01-01T00:01:00",
            output_paths=[f2],
        )

        all_cps = load_all_checkpoints(tmp_path)

        assert set(all_cps.keys()) == {"plan", "search"}
        assert all_cps["plan"].status == "completed"
        assert all_cps["search"].status == "completed"

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        assert load_all_checkpoints(tmp_path) == {}

    def test_skips_corrupt_files(self, tmp_path: Path) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        (cp_dir / "good.checkpoint.json").write_text(
            json.dumps({"stage": "plan", "status": "completed"}),
            encoding="utf-8",
        )
        (cp_dir / "bad.checkpoint.json").write_text(
            "NOT VALID JSON",
            encoding="utf-8",
        )

        all_cps = load_all_checkpoints(tmp_path)

        assert "plan" in all_cps
        assert len(all_cps) == 1
