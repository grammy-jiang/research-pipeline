"""Unit tests for storage.manifests module."""

import json
from datetime import UTC, datetime
from pathlib import Path

from arxiv_paper_pipeline.models.manifest import RunManifest, StageRecord
from arxiv_paper_pipeline.storage.manifests import (
    load_manifest,
    read_jsonl,
    save_manifest,
    update_stage,
    write_jsonl,
)


def _make_manifest(run_id: str = "test_run") -> RunManifest:
    return RunManifest(
        run_id=run_id,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        package_version="0.1.0",
        topic_input="neural search",
    )


class TestManifestRoundTrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        save_manifest(tmp_path, manifest)
        loaded = load_manifest(tmp_path)
        assert loaded is not None
        assert loaded.run_id == "test_run"
        assert loaded.topic_input == "neural search"

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        result = load_manifest(tmp_path)
        assert result is None

    def test_manifest_file_is_json(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        save_manifest(tmp_path, manifest)
        path = tmp_path / "run_manifest.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["run_id"] == "test_run"

    def test_manifest_with_stages(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        stage = StageRecord(
            stage_name="search",
            status="completed",
            started_at=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            ended_at=datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
            duration_ms=300000,
        )
        manifest = update_stage(manifest, stage)
        save_manifest(tmp_path, manifest)
        loaded = load_manifest(tmp_path)
        assert loaded is not None
        assert "search" in loaded.stages
        assert loaded.stages["search"].status == "completed"


class TestUpdateStage:
    def test_add_new_stage(self) -> None:
        manifest = _make_manifest()
        stage = StageRecord(stage_name="search", status="pending")
        updated = update_stage(manifest, stage)
        assert "search" in updated.stages

    def test_update_existing_stage(self) -> None:
        manifest = _make_manifest()
        stage1 = StageRecord(stage_name="search", status="pending")
        manifest = update_stage(manifest, stage1)
        stage2 = StageRecord(stage_name="search", status="completed")
        manifest = update_stage(manifest, stage2)
        assert manifest.stages["search"].status == "completed"


class TestJsonl:
    def test_write_and_read(self, tmp_path: Path) -> None:
        records = [
            {"id": "1", "value": "a"},
            {"id": "2", "value": "b"},
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(path, records)
        loaded = read_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["id"] == "1"
        assert loaded[1]["value"] == "b"

    def test_read_nonexistent(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.jsonl"
        result = read_jsonl(path)
        assert result == []

    def test_empty_write(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        write_jsonl(path, [])
        result = read_jsonl(path)
        assert result == []

    def test_preserves_types(self, tmp_path: Path) -> None:
        records = [
            {"num": 42, "flag": True, "items": [1, 2, 3]},
        ]
        path = tmp_path / "typed.jsonl"
        write_jsonl(path, records)
        loaded = read_jsonl(path)
        assert loaded[0]["num"] == 42
        assert loaded[0]["flag"] is True
        assert loaded[0]["items"] == [1, 2, 3]
