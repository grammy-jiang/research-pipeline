"""Unit tests for infra.paths module."""

from pathlib import Path

from research_pipeline.infra.paths import (
    STAGE_NAMES,
    generate_run_id,
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
