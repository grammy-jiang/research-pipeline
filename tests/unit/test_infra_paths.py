"""Unit tests for infra.paths module."""

import os
from pathlib import Path

from research_pipeline.infra.paths import (
    STAGE_NAMES,
    generate_run_id,
    latest_run_id,
    logs_dir,
    run_dir,
    stage_dir,
)


class TestGenerateRunId:
    def test_returns_string(self) -> None:
        result = generate_run_id()
        assert isinstance(result, str)

    def test_correct_length(self) -> None:
        result = generate_run_id()
        assert len(result) == 12

    def test_hex_characters(self) -> None:
        result = generate_run_id()
        int(result, 16)  # Should not raise

    def test_unique(self) -> None:
        ids = {generate_run_id() for _ in range(100)}
        assert len(ids) == 100


class TestRunDir:
    def test_basic_path(self) -> None:
        result = run_dir(Path("runs"), "abc123")
        assert result == Path("runs/abc123")


class TestStageDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        result = stage_dir(tmp_path, "run1", "search")
        assert result.exists()
        assert result == tmp_path / "run1" / "search"

    def test_idempotent(self, tmp_path: Path) -> None:
        d1 = stage_dir(tmp_path, "run1", "search")
        d2 = stage_dir(tmp_path, "run1", "search")
        assert d1 == d2


class TestLogsDir:
    def test_creates_logs_directory(self, tmp_path: Path) -> None:
        result = logs_dir(tmp_path, "run1")
        assert result.exists()
        assert result == tmp_path / "run1" / "logs"


class TestStageNames:
    def test_all_stages_present(self) -> None:
        expected = [
            "plan",
            "search",
            "screen",
            "download",
            "convert",
            "extract",
            "summarize",
        ]
        assert expected == STAGE_NAMES


class TestLatestRunId:
    """latest_run_id resolves run_id="" to the newest real run, not the root (#110)."""

    def _make_run(self, ws: Path, name: str, mtime: float) -> None:
        run = ws / name
        run.mkdir(parents=True)
        manifest = run / "run_manifest.json"
        manifest.write_text("{}")
        os.utime(manifest, (mtime, mtime))

    def test_empty_workspace_returns_empty(self, tmp_path: Path) -> None:
        assert latest_run_id(tmp_path) == ""

    def test_missing_workspace_returns_empty(self, tmp_path: Path) -> None:
        assert latest_run_id(tmp_path / "nope") == ""

    def test_picks_newest_by_manifest_mtime(self, tmp_path: Path) -> None:
        self._make_run(tmp_path, "run-old", mtime=1000)
        self._make_run(tmp_path, "run-new", mtime=2000)
        assert latest_run_id(tmp_path) == "run-new"

    def test_ignores_dirs_without_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-run").mkdir()
        self._make_run(tmp_path, "real-run", mtime=1500)
        assert latest_run_id(tmp_path) == "real-run"
