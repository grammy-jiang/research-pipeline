"""Tests for artifact registration and hashing utilities."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.models.manifest import ArtifactRecord
from research_pipeline.storage.artifacts import register_artifact


class TestRegisterArtifact:
    """Tests for register_artifact()."""

    def test_returns_artifact_record(self, tmp_path: Path) -> None:
        """Result is an ArtifactRecord."""
        f = tmp_path / "data.jsonl"
        f.write_text("{}")
        rec = register_artifact(f, "metadata_jsonl", "search", tmp_path)
        assert isinstance(rec, ArtifactRecord)

    def test_correct_fields(self, tmp_path: Path) -> None:
        """Returned record has correct artifact_type and producer."""
        f = tmp_path / "data.jsonl"
        f.write_text("{}")
        rec = register_artifact(f, "metadata_jsonl", "search", tmp_path)
        assert rec.artifact_type == "metadata_jsonl"
        assert rec.producer == "search"
        assert rec.artifact_id == "search:data.jsonl"

    def test_path_inside_run_root_is_relative(self, tmp_path: Path) -> None:
        """Path inside run_root produces a relative path string."""
        sub = tmp_path / "convert" / "out.md"
        sub.parent.mkdir(parents=True, exist_ok=True)
        sub.write_text("content")
        rec = register_artifact(sub, "markdown", "convert", tmp_path)
        assert rec.path == "convert/out.md"

    def test_path_outside_run_root_is_absolute(self, tmp_path: Path) -> None:
        """Path outside run_root falls back to absolute path string."""
        outside = tmp_path / "other" / "file.pdf"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("pdf")
        run_root = tmp_path / "runs" / "abc"
        run_root.mkdir(parents=True, exist_ok=True)
        rec = register_artifact(outside, "pdf", "download", run_root)
        assert rec.path == str(outside)

    def test_missing_file_sha256_empty(self, tmp_path: Path) -> None:
        """Non-existent file produces empty sha256."""
        f = tmp_path / "does_not_exist.txt"
        rec = register_artifact(f, "text", "extract", tmp_path)
        assert rec.sha256 == ""

    def test_existing_file_sha256_nonempty(self, tmp_path: Path) -> None:
        """Existing file produces a non-empty sha256 hash."""
        f = tmp_path / "data.txt"
        f.write_text("hello world")
        rec = register_artifact(f, "text", "extract", tmp_path)
        assert rec.sha256 != ""
        assert len(rec.sha256) == 64  # SHA-256 hex digest length

    def test_inputs_default_to_empty_list(self, tmp_path: Path) -> None:
        """Omitting inputs produces an empty list."""
        f = tmp_path / "x.txt"
        f.write_text("x")
        rec = register_artifact(f, "text", "stage", tmp_path)
        assert rec.inputs == []

    def test_inputs_passed_through(self, tmp_path: Path) -> None:
        """Explicit inputs are preserved."""
        f = tmp_path / "x.txt"
        f.write_text("x")
        rec = register_artifact(f, "text", "stage", tmp_path, inputs=["a:1", "b:2"])
        assert rec.inputs == ["a:1", "b:2"]

    def test_tool_fingerprint_none_by_default(self, tmp_path: Path) -> None:
        """Omitting tool_fingerprint gives None."""
        f = tmp_path / "x.txt"
        f.write_text("x")
        rec = register_artifact(f, "text", "stage", tmp_path)
        assert rec.tool_fingerprint is None

    def test_tool_fingerprint_set(self, tmp_path: Path) -> None:
        """Explicit tool_fingerprint is preserved."""
        f = tmp_path / "x.txt"
        f.write_text("x")
        rec = register_artifact(
            f, "text", "stage", tmp_path, tool_fingerprint="docling-2.0"
        )
        assert rec.tool_fingerprint == "docling-2.0"

    def test_created_at_populated(self, tmp_path: Path) -> None:
        """created_at is a datetime object."""
        f = tmp_path / "x.txt"
        f.write_text("x")
        rec = register_artifact(f, "text", "stage", tmp_path)
        assert rec.created_at is not None
