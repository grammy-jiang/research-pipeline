"""Unit tests for Pydantic domain models and schema validation."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.extraction import ChunkMetadata
from research_pipeline.models.manifest import (
    RunManifest,
    StageRecord,
)
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.models.screening import (
    CheapScoreBreakdown,
)


class TestQueryPlan:
    def test_minimal_valid(self) -> None:
        plan = QueryPlan(
            topic_raw="test",
            topic_normalized="test topic",
        )
        assert plan.must_terms == []
        assert plan.primary_months == 6

    def test_full(self) -> None:
        plan = QueryPlan(
            topic_raw="neural search",
            topic_normalized="neural information retrieval",
            must_terms=["neural", "retrieval"],
            nice_terms=["embedding"],
            negative_terms=["survey"],
            candidate_categories=["cs.IR"],
            query_variants=["ti:neural AND ti:retrieval"],
            primary_months=3,
            fallback_months=9,
        )
        assert len(plan.must_terms) == 2
        assert plan.primary_months == 3

    def test_defaults(self) -> None:
        plan = QueryPlan(
            topic_raw="t",
            topic_normalized="t",
        )
        assert plan.sparsity_thresholds.min_candidates == 40
        assert plan.fallback_months == 12


class TestCandidateRecord:
    def test_valid_record(self) -> None:
        record = CandidateRecord(
            arxiv_id="2401.12345",
            version="v1",
            title="Test Paper",
            authors=["Author One"],
            published=datetime(2024, 1, 1, tzinfo=UTC),
            updated=datetime(2024, 1, 1, tzinfo=UTC),
            categories=["cs.IR"],
            primary_category="cs.IR",
            abstract="Test abstract",
            abs_url="https://arxiv.org/abs/2401.12345",
            pdf_url="https://arxiv.org/pdf/2401.12345",
        )
        assert record.arxiv_id == "2401.12345"

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            CandidateRecord(
                arxiv_id="2401.12345",
                version="v1",
                # missing title
            )  # type: ignore[call-arg]


class TestCheapScoreBreakdown:
    def test_valid(self) -> None:
        score = CheapScoreBreakdown(
            bm25_title=0.5,
            bm25_abstract=0.7,
            cat_match=1.0,
            negative_penalty=0.0,
            recency_bonus=0.8,
            cheap_score=0.65,
        )
        assert score.cheap_score == 0.65


class TestRunManifest:
    def test_roundtrip_serialization(self) -> None:
        manifest = RunManifest(
            run_id="abc123",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            package_version="0.1.0",
            topic_input="neural search",
        )
        data = manifest.model_dump(mode="json")
        restored = RunManifest.model_validate(data)
        assert restored.run_id == manifest.run_id
        assert restored.topic_input == manifest.topic_input

    def test_with_stages(self) -> None:
        manifest = RunManifest(
            run_id="abc123",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            package_version="0.1.0",
            topic_input="test",
            stages={
                "search": StageRecord(
                    stage_name="search",
                    status="completed",
                )
            },
        )
        assert manifest.stages["search"].status == "completed"


class TestStageRecord:
    def test_minimal(self) -> None:
        record = StageRecord(stage_name="search", status="pending")
        assert record.started_at is None
        assert record.output_paths == []

    def test_full(self) -> None:
        record = StageRecord(
            stage_name="search",
            status="completed",
            started_at=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            ended_at=datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
            duration_ms=300000,
            input_hash="abc123",
            output_paths=["search/results.jsonl"],
            warnings=["Slow response"],
            errors=[],
        )
        assert record.duration_ms == 300000


class TestChunkMetadata:
    def test_valid(self) -> None:
        meta = ChunkMetadata(
            paper_id="2401.12345",
            section_path="Introduction",
            chunk_id="2401.12345_chunk_001",
            source_span="L1-L20",
            token_count=150,
        )
        assert meta.paper_id == "2401.12345"
