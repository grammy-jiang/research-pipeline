"""Unit tests for arxiv.dedup module."""

from datetime import UTC, datetime

from arxiv_paper_pipeline.arxiv.dedup import dedup_across_queries, dedup_by_version
from arxiv_paper_pipeline.models.candidate import CandidateRecord


def _make_candidate(
    arxiv_id: str = "2401.12345",
    version: str = "v1",
    title: str = "Test Paper",
) -> CandidateRecord:
    """Helper to create a CandidateRecord with minimal fields."""
    return CandidateRecord(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        authors=["Author"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
        categories=["cs.IR"],
        primary_category="cs.IR",
        abstract="Test abstract",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}{version}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}{version}",
    )


class TestDedupByVersion:
    def test_no_duplicates(self) -> None:
        candidates = [
            _make_candidate("2401.11111"),
            _make_candidate("2401.22222"),
        ]
        result = dedup_by_version(candidates)
        assert len(result) == 2

    def test_keeps_latest_version(self) -> None:
        candidates = [
            _make_candidate("2401.12345", "v1"),
            _make_candidate("2401.12345", "v3"),
            _make_candidate("2401.12345", "v2"),
        ]
        result = dedup_by_version(candidates)
        assert len(result) == 1
        assert result[0].version == "v3"

    def test_empty_list(self) -> None:
        result = dedup_by_version([])
        assert result == []

    def test_single_entry(self) -> None:
        candidates = [_make_candidate("2401.12345")]
        result = dedup_by_version(candidates)
        assert len(result) == 1

    def test_mixed_ids(self) -> None:
        candidates = [
            _make_candidate("2401.11111", "v1"),
            _make_candidate("2401.11111", "v2"),
            _make_candidate("2401.22222", "v1"),
            _make_candidate("2401.22222", "v3"),
        ]
        result = dedup_by_version(candidates)
        assert len(result) == 2
        versions = {c.arxiv_id: c.version for c in result}
        assert versions["2401.11111"] == "v2"
        assert versions["2401.22222"] == "v3"


class TestDedupAcrossQueries:
    def test_merge_two_lists(self) -> None:
        list1 = [_make_candidate("2401.11111")]
        list2 = [_make_candidate("2401.22222")]
        result = dedup_across_queries([list1, list2])
        assert len(result) == 2

    def test_dedup_across_lists(self) -> None:
        list1 = [_make_candidate("2401.11111", "v1")]
        list2 = [_make_candidate("2401.11111", "v2")]
        result = dedup_across_queries([list1, list2])
        assert len(result) == 1
        assert result[0].version == "v2"

    def test_empty_lists(self) -> None:
        result = dedup_across_queries([[], []])
        assert result == []

    def test_single_list(self) -> None:
        list1 = [_make_candidate("2401.11111")]
        result = dedup_across_queries([list1])
        assert len(result) == 1
