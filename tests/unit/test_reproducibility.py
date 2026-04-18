"""Tests for reproducibility scoring (quality/composite.py)."""

from datetime import UTC, datetime

import pytest

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.quality.composite import (
    DEFAULT_WEIGHTS,
    compute_quality_score,
    reproducibility_score,
)


def _make_candidate(
    abstract: str = "A paper about AI systems.",
    citation_count: int | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        arxiv_id="2401.00001",
        version="v1",
        title="Test Paper",
        authors=["Author A"],
        published=datetime(2024, 1, 15, tzinfo=UTC),
        updated=datetime(2024, 1, 15, tzinfo=UTC),
        categories=["cs.AI"],
        primary_category="cs.AI",
        abstract=abstract,
        abs_url="https://arxiv.org/abs/2401.00001",
        pdf_url="https://arxiv.org/pdf/2401.00001",
        citation_count=citation_count,
    )


class TestReproducibilityScore:
    def test_code_url_gives_high_score(self) -> None:
        c = _make_candidate()
        score = reproducibility_score(c, code_url="https://github.com/test/repo")
        assert score >= 0.4

    def test_no_signals_gives_low_score(self) -> None:
        c = _make_candidate(abstract="Short abstract.")
        score = reproducibility_score(c)
        assert score < 0.3

    def test_abstract_mentions_github(self) -> None:
        c = _make_candidate(
            abstract=(
                "Our code is available at github.com/test/repo for reproducibility."
            )
        )
        score = reproducibility_score(c)
        assert score >= 0.3

    def test_abstract_mentions_dataset(self) -> None:
        c = _make_candidate(
            abstract="We evaluate on a new benchmark dataset with 10k examples."
        )
        score = reproducibility_score(c)
        assert score >= 0.2

    def test_has_data_flag(self) -> None:
        c = _make_candidate()
        score = reproducibility_score(c, has_data=True)
        assert score >= 0.3

    def test_long_abstract_bonus(self) -> None:
        long_abstract = " ".join(["word"] * 250)
        c = _make_candidate(abstract=long_abstract)
        score_long = reproducibility_score(c)

        short_abstract = " ".join(["word"] * 50)
        c_short = _make_candidate(abstract=short_abstract)
        score_short = reproducibility_score(c_short)

        assert score_long > score_short

    def test_citation_bonus(self) -> None:
        c_cited = _make_candidate(citation_count=50)
        c_uncited = _make_candidate(citation_count=0)

        score_cited = reproducibility_score(c_cited)
        score_uncited = reproducibility_score(c_uncited)

        assert score_cited > score_uncited

    def test_all_signals_present(self) -> None:
        long_abstract = " ".join(["word"] * 250)
        c = _make_candidate(abstract=long_abstract, citation_count=100)
        score = reproducibility_score(
            c, code_url="https://github.com/test/repo", has_data=True
        )
        assert score == 1.0  # Capped at 1.0

    def test_score_capped_at_one(self) -> None:
        c = _make_candidate(citation_count=1000)
        score = reproducibility_score(
            c, code_url="https://github.com/test/repo", has_data=True
        )
        assert score <= 1.0


class TestCompositeWithReproducibility:
    def test_default_weights_include_reproducibility(self) -> None:
        assert "reproducibility_weight" in DEFAULT_WEIGHTS
        assert DEFAULT_WEIGHTS["reproducibility_weight"] > 0

    def test_weights_sum_to_one(self) -> None:
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_reproducibility_in_quality_score(self) -> None:
        c = _make_candidate(citation_count=10)
        qs = compute_quality_score(
            c,
            code_url="https://github.com/test/repo",
            has_data=True,
        )
        assert qs.reproducibility > 0.0
        assert qs.details.get("reproducibility_score") is not None

    def test_reproducibility_zero_without_signals(self) -> None:
        c = _make_candidate(abstract="Short.")
        qs = compute_quality_score(c)
        # Some signal might come from other proxies, but should be low
        assert qs.reproducibility < 0.5

    def test_custom_weights_without_repro_still_work(self) -> None:
        c = _make_candidate(citation_count=100)
        w = {
            "citation_weight": 1.0,
            "venue_weight": 0.0,
            "author_weight": 0.0,
            "recency_weight": 0.0,
        }
        qs = compute_quality_score(c, weights=w)
        # Should work without reproducibility_weight key
        assert qs.composite_score == pytest.approx(qs.citation_impact, abs=0.01)
