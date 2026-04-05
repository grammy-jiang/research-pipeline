"""Unit tests for models.quality — QualityScore model."""

import pytest
from pydantic import ValidationError

from research_pipeline.models.quality import QualityScore


class TestQualityScore:
    """Tests for the QualityScore model."""

    def test_valid_full(self) -> None:
        score = QualityScore(
            paper_id="2401.12345",
            citation_impact=0.75,
            venue_tier="A*",
            venue_score=1.0,
            author_credibility=0.6,
            reproducibility=0.0,
            composite_score=0.72,
            details={"citation_count": 150, "max_h_index": 45},
        )
        assert score.paper_id == "2401.12345"
        assert score.citation_impact == 0.75
        assert score.venue_tier == "A*"
        assert score.venue_score == 1.0
        assert score.author_credibility == 0.6
        assert score.composite_score == 0.72
        assert score.details["citation_count"] == 150

    def test_minimal_valid(self) -> None:
        score = QualityScore(
            paper_id="2401.12345",
            citation_impact=0.5,
            venue_score=0.1,
            author_credibility=0.3,
            composite_score=0.35,
        )
        assert score.venue_tier is None
        assert score.reproducibility == 0.0
        assert score.details == {}

    def test_roundtrip_serialization(self) -> None:
        original = QualityScore(
            paper_id="2401.99999",
            citation_impact=0.9,
            venue_tier="B",
            venue_score=0.5,
            author_credibility=0.8,
            reproducibility=0.0,
            composite_score=0.7,
            details={"venue_name": "ICML", "h_index": 60},
        )
        data = original.model_dump()
        restored = QualityScore.model_validate(data)
        assert restored == original

    def test_json_roundtrip(self) -> None:
        original = QualityScore(
            paper_id="2401.11111",
            citation_impact=0.4,
            venue_score=0.3,
            author_credibility=0.2,
            composite_score=0.3,
        )
        json_str = original.model_dump_json()
        restored = QualityScore.model_validate_json(json_str)
        assert restored == original

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            QualityScore(
                paper_id="2401.12345",
                citation_impact=0.5,
                # missing venue_score, author_credibility, composite_score
            )  # type: ignore[call-arg]
