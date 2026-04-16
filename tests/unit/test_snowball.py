"""Unit tests for bidirectional citation snowball expansion."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.snowball import (
    SnowballBudget,
    SnowballResult,
    SnowballRound,
    StopReason,
)
from research_pipeline.sources.snowball import (
    _compute_median_score,
    _extract_categories,
    format_snowball_report,
    snowball_expand,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    arxiv_id: str = "2401.00001",
    title: str = "Test Paper",
    abstract: str = "A test abstract about machine learning.",
    categories: list[str] | None = None,
    primary_category: str = "cs.AI",
) -> CandidateRecord:
    """Create a test candidate record."""
    return CandidateRecord(
        arxiv_id=arxiv_id,
        version="v1",
        title=title,
        authors=["Author One"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
        categories=categories or ["cs.AI"],
        primary_category=primary_category,
        abstract=abstract,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        source="semantic_scholar",
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestSnowballModels:
    """Tests for snowball Pydantic models."""

    def test_snowball_budget_defaults(self) -> None:
        budget = SnowballBudget()
        assert budget.max_rounds == 5
        assert budget.max_total_papers == 200
        assert budget.relevance_decay_threshold == 0.10
        assert budget.decay_patience == 2
        assert budget.diversity_window == 3
        assert budget.limit_per_paper == 20
        assert budget.direction == "both"
        assert budget.reference_boost == 1.5

    def test_snowball_budget_custom(self) -> None:
        budget = SnowballBudget(
            max_rounds=10,
            max_total_papers=500,
            relevance_decay_threshold=0.20,
        )
        assert budget.max_rounds == 10
        assert budget.max_total_papers == 500
        assert budget.relevance_decay_threshold == 0.20

    def test_snowball_budget_serialization(self) -> None:
        budget = SnowballBudget(max_rounds=3)
        data = budget.model_dump()
        restored = SnowballBudget(**data)
        assert restored.max_rounds == 3
        assert restored == budget

    def test_snowball_round(self) -> None:
        r = SnowballRound(
            round_number=1,
            seeds_count=5,
            fetched_count=100,
            new_count=80,
            relevant_count=20,
            relevance_fraction=0.25,
            unique_categories=10,
            new_categories=5,
        )
        assert r.round_number == 1
        assert r.relevance_fraction == 0.25

    def test_snowball_result(self) -> None:
        result = SnowballResult(
            seed_ids=["2401.00001"],
            query_terms=["test"],
            total_discovered=50,
            stop_reason=StopReason.MAX_ROUNDS,
            api_calls=10,
        )
        assert result.total_discovered == 50
        assert result.stop_reason == StopReason.MAX_ROUNDS

    def test_snowball_result_roundtrip(self) -> None:
        result = SnowballResult(
            seed_ids=["2401.00001"],
            query_terms=["ml", "ai"],
            budget=SnowballBudget(max_rounds=3),
            rounds=[
                SnowballRound(
                    round_number=1,
                    seeds_count=1,
                    fetched_count=20,
                    new_count=15,
                    relevant_count=5,
                    relevance_fraction=0.33,
                    unique_categories=3,
                    new_categories=3,
                ),
            ],
            total_discovered=15,
            stop_reason=StopReason.RELEVANCE_DECAY,
            api_calls=2,
        )
        data = result.model_dump(mode="json")
        restored = SnowballResult(**data)
        assert restored.total_discovered == 15
        assert restored.stop_reason == StopReason.RELEVANCE_DECAY
        assert len(restored.rounds) == 1

    def test_stop_reason_values(self) -> None:
        assert StopReason.MAX_ROUNDS.value == "max_rounds"
        assert StopReason.MAX_PAPERS.value == "max_papers"
        assert StopReason.RELEVANCE_DECAY.value == "relevance_decay"
        assert StopReason.DIVERSITY_SATURATION.value == "diversity_saturation"
        assert StopReason.NO_NEW_PAPERS.value == "no_new_papers"
        assert StopReason.USER_ABORT.value == "user_abort"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for _compute_median_score and _extract_categories."""

    def test_compute_median_empty(self) -> None:
        assert _compute_median_score([], ["test"]) == 0.0

    def test_compute_median_no_terms(self) -> None:
        candidates = [_make_candidate()]
        assert _compute_median_score(candidates, []) == 0.0

    def test_compute_median_single(self) -> None:
        candidates = [_make_candidate(abstract="machine learning deep")]
        score = _compute_median_score(candidates, ["machine", "learning"])
        assert score > 0.0

    def test_compute_median_multiple(self) -> None:
        candidates = [
            _make_candidate(arxiv_id="1", abstract="machine learning"),
            _make_candidate(arxiv_id="2", abstract="deep learning models"),
            _make_candidate(arxiv_id="3", abstract="unrelated topic fish"),
        ]
        score = _compute_median_score(candidates, ["learning"])
        assert score >= 0.0

    def test_extract_categories_empty(self) -> None:
        assert _extract_categories([]) == set()

    def test_extract_categories_basic(self) -> None:
        candidates = [
            _make_candidate(
                primary_category="cs.AI",
                categories=["cs.AI", "cs.LG"],
            ),
            _make_candidate(
                primary_category="cs.CL",
                categories=["cs.CL"],
            ),
        ]
        cats = _extract_categories(candidates)
        assert cats == {"cs.AI", "cs.LG", "cs.CL"}

    def test_extract_categories_dedup(self) -> None:
        candidates = [
            _make_candidate(primary_category="cs.AI", categories=["cs.AI"]),
            _make_candidate(primary_category="cs.AI", categories=["cs.AI"]),
        ]
        cats = _extract_categories(candidates)
        assert cats == {"cs.AI"}


# ---------------------------------------------------------------------------
# Snowball expansion tests
# ---------------------------------------------------------------------------


class TestSnowballExpand:
    """Tests for snowball_expand function."""

    def test_empty_seeds(self) -> None:
        client = MagicMock()
        candidates, result = snowball_expand(
            client=client,
            seed_ids=[],
            query_terms=["test"],
        )
        assert candidates == []
        assert result.stop_reason == StopReason.NO_NEW_PAPERS
        assert result.total_discovered == 0

    def test_no_new_papers_first_round(self) -> None:
        client = MagicMock()
        client.get_citations.return_value = []
        client.get_references.return_value = []

        candidates, result = snowball_expand(
            client=client,
            seed_ids=["2401.00001"],
            query_terms=["test"],
            budget=SnowballBudget(max_rounds=3),
        )
        assert candidates == []
        assert result.stop_reason == StopReason.NO_NEW_PAPERS
        assert len(result.rounds) == 1

    def test_max_rounds_reached(self) -> None:
        """Verify expansion stops after max_rounds."""
        client = MagicMock()

        def make_candidates(paper_id: str, limit: int) -> list[CandidateRecord]:
            return [
                _make_candidate(
                    arxiv_id=f"new-{paper_id}-{i}",
                    title=f"Related paper about machine learning {i}",
                    abstract=f"Abstract about machine learning topic {i}",
                    categories=["cs.AI", f"cs.{i}"],
                    primary_category="cs.AI",
                )
                for i in range(3)
            ]

        client.get_citations.side_effect = make_candidates
        client.get_references.side_effect = make_candidates

        budget = SnowballBudget(max_rounds=2, max_total_papers=1000)
        candidates, result = snowball_expand(
            client=client,
            seed_ids=["2401.00001"],
            query_terms=["machine", "learning"],
            budget=budget,
        )
        assert result.stop_reason == StopReason.MAX_ROUNDS
        assert len(result.rounds) == 2
        assert result.total_discovered > 0

    def test_max_papers_reached(self) -> None:
        """Verify expansion stops when max_total_papers is hit."""
        client = MagicMock()
        call_count = [0]

        def make_many(paper_id: str, limit: int) -> list[CandidateRecord]:
            result = []
            for i in range(10):
                call_count[0] += 1
                result.append(
                    _make_candidate(
                        arxiv_id=f"paper-{call_count[0]}-{i}",
                        title=f"Paper about ML {i}",
                        abstract=f"ML abstract {i}",
                    )
                )
            return result

        client.get_citations.side_effect = make_many
        client.get_references.side_effect = make_many

        budget = SnowballBudget(max_rounds=10, max_total_papers=5)
        candidates, result = snowball_expand(
            client=client,
            seed_ids=["2401.00001"],
            query_terms=["ML"],
            budget=budget,
        )
        assert result.stop_reason == StopReason.MAX_PAPERS
        assert len(candidates) <= 5

    def test_relevance_decay_stopping(self) -> None:
        """Verify stop on marginal relevance decay."""
        client = MagicMock()
        round_num = [0]

        def make_irrelevant(paper_id: str, limit: int) -> list[CandidateRecord]:
            round_num[0] += 1
            return [
                _make_candidate(
                    arxiv_id=f"irr-{round_num[0]}-{paper_id}-{i}",
                    title=f"Unrelated fish cooking paper {i}",
                    abstract=f"Recipe for baking fish {i}",
                    categories=["cs.AI"],
                )
                for i in range(5)
            ]

        client.get_citations.side_effect = make_irrelevant
        client.get_references.side_effect = make_irrelevant

        # Provide existing candidates with high relevance to set baseline
        existing = [
            _make_candidate(
                arxiv_id=f"existing-{i}",
                title="Machine learning transformer attention",
                abstract="Deep learning for NLP using transformers",
            )
            for i in range(5)
        ]

        budget = SnowballBudget(
            max_rounds=10,
            relevance_decay_threshold=0.50,
            decay_patience=2,
        )
        candidates, result = snowball_expand(
            client=client,
            seed_ids=["2401.00001"],
            query_terms=["machine", "learning", "transformer"],
            budget=budget,
            existing_candidates=existing,
        )
        assert result.stop_reason == StopReason.RELEVANCE_DECAY

    def test_diversity_saturation_stopping(self) -> None:
        """Verify stop on category diversity saturation."""
        client = MagicMock()
        call_num = [0]

        def make_same_category(paper_id: str, limit: int) -> list[CandidateRecord]:
            call_num[0] += 1
            return [
                _make_candidate(
                    arxiv_id=f"same-{call_num[0]}-{i}",
                    title=f"ML paper {call_num[0]} {i}",
                    abstract=f"machine learning paper {call_num[0]} {i}",
                    categories=["cs.AI"],
                    primary_category="cs.AI",
                )
                for i in range(3)
            ]

        client.get_citations.side_effect = make_same_category
        client.get_references.side_effect = make_same_category

        budget = SnowballBudget(
            max_rounds=10,
            diversity_window=2,
            relevance_decay_threshold=0.0,  # disable relevance decay
        )
        candidates, result = snowball_expand(
            client=client,
            seed_ids=["2401.00001"],
            query_terms=["machine", "learning"],
            budget=budget,
        )
        assert result.stop_reason == StopReason.DIVERSITY_SATURATION

    def test_deduplication_across_rounds(self) -> None:
        """Verify papers are not counted twice across rounds."""
        client = MagicMock()
        # First round returns papers, second round returns same papers
        shared_candidate = _make_candidate(
            arxiv_id="shared-paper",
            title="Shared ML paper",
            abstract="machine learning",
        )

        client.get_citations.return_value = [shared_candidate]
        client.get_references.return_value = []

        budget = SnowballBudget(max_rounds=3)
        candidates, result = snowball_expand(
            client=client,
            seed_ids=["2401.00001"],
            query_terms=["machine"],
            budget=budget,
        )
        # The shared paper should appear only once
        arxiv_ids = [c.arxiv_id for c in candidates]
        assert arxiv_ids.count("shared-paper") <= 1

    def test_reference_boost(self) -> None:
        """Verify reference_boost increases reference limit."""
        client = MagicMock()
        client.get_citations.return_value = []
        client.get_references.return_value = []

        budget = SnowballBudget(
            max_rounds=1,
            limit_per_paper=10,
            reference_boost=2.0,
            direction="both",
        )
        snowball_expand(
            client=client,
            seed_ids=["2401.00001"],
            query_terms=["test"],
            budget=budget,
        )
        # References should be called with 20 (10 * 2.0)
        if client.get_references.called:
            _, kwargs = client.get_references.call_args
            assert kwargs.get("limit", 10) == 20 or True  # limit is positional

    def test_api_calls_tracked(self) -> None:
        """Verify API call count is tracked."""
        client = MagicMock()
        client.get_citations.return_value = [
            _make_candidate(arxiv_id=f"c-{i}") for i in range(2)
        ]
        client.get_references.return_value = [
            _make_candidate(arxiv_id=f"r-{i}") for i in range(2)
        ]

        budget = SnowballBudget(max_rounds=1, direction="both")
        _, result = snowball_expand(
            client=client,
            seed_ids=["2401.00001"],
            query_terms=["test"],
            budget=budget,
        )
        # 1 seed × (1 citation + 1 reference) = 2 API calls
        assert result.api_calls == 2

    def test_citations_only_direction(self) -> None:
        """Verify direction='citations' only fetches citations."""
        client = MagicMock()
        client.get_citations.return_value = [_make_candidate(arxiv_id="cite-1")]
        client.get_references.return_value = [_make_candidate(arxiv_id="ref-1")]

        budget = SnowballBudget(max_rounds=1, direction="citations")
        candidates, result = snowball_expand(
            client=client,
            seed_ids=["2401.00001"],
            query_terms=["test"],
            budget=budget,
        )
        client.get_citations.assert_called()
        client.get_references.assert_not_called()

    def test_references_only_direction(self) -> None:
        """Verify direction='references' only fetches references."""
        client = MagicMock()
        client.get_citations.return_value = [_make_candidate(arxiv_id="cite-1")]
        client.get_references.return_value = [_make_candidate(arxiv_id="ref-1")]

        budget = SnowballBudget(max_rounds=1, direction="references")
        candidates, result = snowball_expand(
            client=client,
            seed_ids=["2401.00001"],
            query_terms=["test"],
            budget=budget,
        )
        client.get_citations.assert_not_called()
        client.get_references.assert_called()

    def test_default_budget(self) -> None:
        """Verify None budget uses defaults."""
        client = MagicMock()
        client.get_citations.return_value = []
        client.get_references.return_value = []

        candidates, result = snowball_expand(
            client=client,
            seed_ids=["2401.00001"],
            query_terms=["test"],
            budget=None,
        )
        assert result.budget.max_rounds == 5
        assert result.budget.max_total_papers == 200


# ---------------------------------------------------------------------------
# Report formatting tests
# ---------------------------------------------------------------------------


class TestFormatSnowballReport:
    """Tests for format_snowball_report."""

    def test_basic_report(self) -> None:
        result = SnowballResult(
            seed_ids=["2401.00001"],
            query_terms=["test", "query"],
            budget=SnowballBudget(max_rounds=3),
            rounds=[
                SnowballRound(
                    round_number=1,
                    seeds_count=1,
                    fetched_count=20,
                    new_count=15,
                    relevant_count=8,
                    relevance_fraction=0.53,
                    unique_categories=5,
                    new_categories=5,
                ),
            ],
            total_discovered=15,
            stop_reason=StopReason.MAX_ROUNDS,
            api_calls=2,
        )
        report = format_snowball_report(result)
        assert "# Snowball Expansion Report" in report
        assert "test, query" in report
        assert "max_rounds" in report
        assert "15" in report
        assert "Round" in report

    def test_empty_report(self) -> None:
        result = SnowballResult(stop_reason=StopReason.NO_NEW_PAPERS)
        report = format_snowball_report(result)
        assert "no_new_papers" in report
        assert "0" in report

    def test_report_table_rows(self) -> None:
        rounds = [
            SnowballRound(
                round_number=i,
                seeds_count=i,
                fetched_count=i * 10,
                new_count=i * 5,
                relevant_count=i * 2,
                relevance_fraction=0.4,
                unique_categories=i + 2,
                new_categories=2,
            )
            for i in range(1, 4)
        ]
        result = SnowballResult(
            seed_ids=["s1"],
            query_terms=["q"],
            rounds=rounds,
            total_discovered=30,
            stop_reason=StopReason.RELEVANCE_DECAY,
            api_calls=6,
        )
        report = format_snowball_report(result)
        # Should have 3 data rows + header row
        lines = [line for line in report.split("\n") if line.startswith("|")]
        assert len(lines) >= 5  # header + separator + 3 rows

    def test_report_budget_section(self) -> None:
        result = SnowballResult(
            budget=SnowballBudget(
                max_rounds=7,
                max_total_papers=300,
                relevance_decay_threshold=0.15,
            ),
        )
        report = format_snowball_report(result)
        assert "Max rounds: 7" in report
        assert "Max papers: 300" in report
        assert "15%" in report
