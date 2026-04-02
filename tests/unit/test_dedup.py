"""Unit tests for arxiv.dedup module."""

from datetime import UTC, datetime

from research_pipeline.arxiv.dedup import dedup_across_queries, dedup_by_version
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.sources.base import dedup_cross_source


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


class TestDedupCrossSource:
    def test_dedup_by_arxiv_id(self) -> None:
        candidates = [
            _make_candidate("2401.11111", title="Paper A"),
            _make_candidate("2401.11111", title="Paper A (arxiv)"),
        ]
        result = dedup_cross_source(candidates)
        assert len(result) == 1

    def test_dedup_by_title(self) -> None:
        c1 = _make_candidate("2401.11111", title="Same Title")
        c2 = _make_candidate("2401.22222", title="same title")
        result = dedup_cross_source([c1, c2])
        assert len(result) == 1

    def test_scholar_placeholder_ids_not_deduped_by_id(self) -> None:
        c1 = _make_candidate("scholar-abc123", title="Paper One")
        c2 = _make_candidate("scholar-def456", title="Paper Two")
        result = dedup_cross_source([c1, c2])
        assert len(result) == 2

    def test_scholar_and_arxiv_dedup_by_title(self) -> None:
        c1 = _make_candidate("2401.11111", title="AI Memory Systems")
        c2 = _make_candidate("scholar-abc", title="AI Memory Systems")
        result = dedup_cross_source([c1, c2])
        assert len(result) == 1

    def test_empty_list(self) -> None:
        result = dedup_cross_source([])
        assert result == []

    def test_preserves_order(self) -> None:
        c1 = _make_candidate("2401.11111", title="First")
        c2 = _make_candidate("2401.22222", title="Second")
        c3 = _make_candidate("2401.33333", title="Third")
        result = dedup_cross_source([c1, c2, c3])
        assert [c.title for c in result] == ["First", "Second", "Third"]

    def test_no_duplicates(self) -> None:
        candidates = [
            _make_candidate("2401.11111", title="Paper A"),
            _make_candidate("2401.22222", title="Paper B"),
        ]
        result = dedup_cross_source(candidates)
        assert len(result) == 2
