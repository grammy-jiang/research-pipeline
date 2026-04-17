"""Tests for failure taxonomy logging (v0.13.23).

Covers:
- FailureCategory and FailureSeverity enums
- FailureRecord dataclass creation
- FailureTaxonomyLogger: log, log_failure, stage-specific helpers
- summary() aggregation by category, severity, stage
- JSONL persistence: append + load_from_disk roundtrip
- Disabled logger no-op behaviour
"""

from __future__ import annotations

from pathlib import Path

from research_pipeline.infra.failure_taxonomy import (
    FailureCategory,
    FailureRecord,
    FailureSeverity,
    FailureTaxonomyLogger,
)

# ── Enums ───────────────────────────────────────────────────────────


class TestEnums:
    """FailureCategory and FailureSeverity enum values."""

    def test_all_categories_are_strings(self) -> None:
        for cat in FailureCategory:
            assert isinstance(cat.value, str)

    def test_category_count(self) -> None:
        assert len(FailureCategory) == 12

    def test_severity_levels(self) -> None:
        values = [s.value for s in FailureSeverity]
        assert values == ["low", "medium", "high", "critical"]


# ── FailureRecord ───────────────────────────────────────────────────


class TestFailureRecord:
    """FailureRecord dataclass."""

    def test_default_construction(self) -> None:
        rec = FailureRecord(category=FailureCategory.UNKNOWN)
        assert rec.category == FailureCategory.UNKNOWN
        assert rec.subcategory == ""
        assert rec.stage == ""
        assert rec.severity == FailureSeverity.MEDIUM
        assert rec.message == ""
        assert rec.details == {}
        assert rec.paper_id == ""
        assert rec.timestamp  # non-empty

    def test_full_construction(self) -> None:
        rec = FailureRecord(
            category=FailureCategory.DOWNLOAD_FAILURE,
            subcategory="http_403",
            stage="download",
            severity=FailureSeverity.HIGH,
            message="Forbidden",
            details={"url": "https://example.com/paper.pdf"},
            paper_id="arxiv:2401.01234",
        )
        assert rec.category == FailureCategory.DOWNLOAD_FAILURE
        assert rec.subcategory == "http_403"
        assert rec.details["url"] == "https://example.com/paper.pdf"


# ── FailureTaxonomyLogger basics ────────────────────────────────────


class TestLoggerBasics:
    """Basic logger operations."""

    def test_log_adds_record(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        rec = FailureRecord(
            category=FailureCategory.LLM_ERROR,
            message="Model timeout",
        )
        fl.log(rec)
        assert len(fl.records) == 1
        assert fl.records[0].category == FailureCategory.LLM_ERROR

    def test_log_failure_convenience(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        rec = fl.log_failure(
            FailureCategory.TIMEOUT,
            "Request timed out",
            stage="download",
            paper_id="arxiv:2401.99999",
        )
        assert rec.category == FailureCategory.TIMEOUT
        assert rec.stage == "download"
        assert rec.paper_id == "arxiv:2401.99999"
        assert len(fl.records) == 1

    def test_disabled_logger_noop(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path, enabled=False)
        fl.log_failure(FailureCategory.UNKNOWN, "should be ignored")
        assert len(fl.records) == 0
        assert not (tmp_path / "logs" / "failures.jsonl").exists()


# ── Stage-specific helpers ──────────────────────────────────────────


class TestStageHelpers:
    """Stage-specific convenience methods."""

    def test_retrieval_miss(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        rec = fl.retrieval_miss("Paper not found", paper_id="arxiv:2401.00001")
        assert rec.category == FailureCategory.RETRIEVAL_MISS
        assert rec.stage == "search"

    def test_conversion_error(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        rec = fl.conversion_error("Corrupt PDF", paper_id="arxiv:2401.00002")
        assert rec.category == FailureCategory.CONVERSION_ERROR
        assert rec.stage == "convert"
        assert rec.severity == FailureSeverity.HIGH

    def test_download_failure(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        rec = fl.download_failure("HTTP 404", paper_id="arxiv:2401.00003")
        assert rec.category == FailureCategory.DOWNLOAD_FAILURE
        assert rec.stage == "download"

    def test_screening_miss(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        rec = fl.screening_miss("Relevant paper excluded", paper_id="p1")
        assert rec.category == FailureCategory.SCREENING_MISS
        assert rec.stage == "screen"

    def test_synthesis_gap(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        rec = fl.synthesis_gap("Memory management not covered")
        assert rec.category == FailureCategory.SYNTHESIS_GAP
        assert rec.stage == "summarize"


# ── Summary aggregation ─────────────────────────────────────────────


class TestSummary:
    """summary() aggregation method."""

    def test_empty_summary(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        s = fl.summary()
        assert s["total"] == 0
        assert s["by_category"] == {}
        assert s["by_severity"] == {}
        assert s["by_stage"] == {}

    def test_mixed_failures_summary(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        fl.retrieval_miss("Miss 1")
        fl.retrieval_miss("Miss 2")
        fl.download_failure("DL fail")
        fl.synthesis_gap("Gap")

        s = fl.summary()
        assert s["total"] == 4
        assert s["by_category"]["retrieval_miss"] == 2
        assert s["by_category"]["download_failure"] == 1
        assert s["by_category"]["synthesis_gap"] == 1
        assert s["by_stage"]["search"] == 2
        assert s["by_stage"]["download"] == 1


# ── JSONL persistence ───────────────────────────────────────────────


class TestPersistence:
    """JSONL write + load_from_disk roundtrip."""

    def test_jsonl_file_created(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        fl.log_failure(FailureCategory.CONFIG_ERROR, "Bad config")
        log_path = tmp_path / "logs" / "failures.jsonl"
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_roundtrip(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        fl.retrieval_miss("Miss", paper_id="p1")
        fl.download_failure("DL", paper_id="p2", details={"code": 404})
        fl.conversion_error("Corrupt", paper_id="p3")

        # Load from fresh logger
        fl2 = FailureTaxonomyLogger(tmp_path)
        loaded = fl2.load_from_disk()
        assert len(loaded) == 3
        assert loaded[0].category == FailureCategory.RETRIEVAL_MISS
        assert loaded[0].paper_id == "p1"
        assert loaded[1].details == {"code": 404}
        assert loaded[2].severity == FailureSeverity.HIGH

    def test_load_empty_file(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        loaded = fl.load_from_disk()
        assert loaded == []

    def test_multiple_appends(self, tmp_path: Path) -> None:
        fl = FailureTaxonomyLogger(tmp_path)
        fl.log_failure(FailureCategory.LLM_ERROR, "Error 1")
        fl.log_failure(FailureCategory.TIMEOUT, "Error 2")
        fl.log_failure(FailureCategory.RATE_LIMIT, "Error 3")

        log_path = tmp_path / "logs" / "failures.jsonl"
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
