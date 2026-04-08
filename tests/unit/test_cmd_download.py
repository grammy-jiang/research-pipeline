"""Tests for lenient shortlist parsing in download stage."""

from research_pipeline.models.screening import (
    RelevanceDecision,
    parse_shortlist_lenient,
)


def _make_full_decision_dict() -> dict:
    """Create a fully-populated RelevanceDecision dict."""
    return {
        "paper": {
            "arxiv_id": "2401.12345",
            "version": "v1",
            "title": "Test Paper",
            "authors": ["Author One"],
            "published": "2024-01-01T00:00:00Z",
            "updated": "2024-01-01T00:00:00Z",
            "categories": ["cs.AI"],
            "primary_category": "cs.AI",
            "abstract": "Test abstract.",
            "abs_url": "https://arxiv.org/abs/2401.12345",
            "pdf_url": "https://arxiv.org/pdf/2401.12345",
        },
        "cheap": {
            "bm25_title": 0.5,
            "bm25_abstract": 0.7,
            "cat_match": 1.0,
            "negative_penalty": 0.0,
            "recency_bonus": 0.8,
            "cheap_score": 0.65,
        },
        "llm": None,
        "final_score": 0.65,
        "download": True,
        "download_reason": "manual_override",
    }


def _make_minimal_decision_dict() -> dict:
    """Create a minimal/simplified shortlist entry (as a user might write)."""
    return {
        "paper": {
            "arxiv_id": "2401.12345",
            "version": "v1",
            "title": "Test Paper",
            "authors": ["Author One"],
            "published": "2024-01-01T00:00:00Z",
            "updated": "2024-01-01T00:00:00Z",
            "categories": ["cs.AI"],
            "primary_category": "cs.AI",
            "abstract": "Test abstract.",
            "abs_url": "https://arxiv.org/abs/2401.12345",
            "pdf_url": "https://arxiv.org/pdf/2401.12345",
        },
        "final_score": 0.8,
        "download": True,
    }


class TestParseShortlistLenient:
    """Test lenient shortlist parsing for manually-edited entries."""

    def test_full_entry_parses(self) -> None:
        """A fully-populated entry should parse without changes."""
        entry = _make_full_decision_dict()
        result = parse_shortlist_lenient(entry)
        assert isinstance(result, RelevanceDecision)
        assert result.paper.arxiv_id == "2401.12345"
        assert result.download is True

    def test_missing_cheap_gets_defaults(self) -> None:
        """Missing 'cheap' field should get default zero breakdown."""
        entry = _make_minimal_decision_dict()
        result = parse_shortlist_lenient(entry)
        assert isinstance(result, RelevanceDecision)
        assert result.cheap.cheap_score == 0.0
        assert result.cheap.bm25_title == 0.0

    def test_cheap_as_float_converts(self) -> None:
        """A user might write cheap as a simple float value."""
        entry = _make_minimal_decision_dict()
        entry["cheap"] = 0.75
        result = parse_shortlist_lenient(entry)
        assert isinstance(result, RelevanceDecision)
        assert result.cheap.cheap_score == 0.75

    def test_missing_download_reason_defaults(self) -> None:
        """Missing download_reason should default to manual_override."""
        entry = _make_minimal_decision_dict()
        assert "download_reason" not in entry
        result = parse_shortlist_lenient(entry)
        assert result.download_reason == "manual_override"

    def test_invalid_download_reason_coerced(self) -> None:
        """An unrecognized download_reason should be coerced to manual_override."""
        entry = _make_full_decision_dict()
        entry["download_reason"] = "agent_selected"
        result = parse_shortlist_lenient(entry)
        assert result.download_reason == "manual_override"

    def test_missing_final_score_defaults(self) -> None:
        """Missing final_score should default to 0.0."""
        entry = _make_minimal_decision_dict()
        del entry["final_score"]
        result = parse_shortlist_lenient(entry)
        assert result.final_score == 0.0

    def test_batch_parsing(self) -> None:
        """Multiple entries should all parse correctly."""
        entries = [_make_full_decision_dict(), _make_minimal_decision_dict()]
        results = [parse_shortlist_lenient(e) for e in entries]
        assert len(results) == 2
        assert all(isinstance(r, RelevanceDecision) for r in results)
