"""Property-based tests for Pydantic domain models using Hypothesis.

Each test generates random valid model instances and verifies:
  1. JSON roundtrip: model → model_dump_json() → model_validate_json() → equal
  2. Dict roundtrip: model → model_dump() → model_validate() → equal
  3. Validators accept all generated data without raising.
"""

from datetime import UTC, datetime

import hypothesis.strategies as st
from hypothesis import given, settings

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.conversion import ConvertManifestEntry
from research_pipeline.models.download import DownloadManifestEntry
from research_pipeline.models.extraction import (
    ChunkMetadata,
    ExtractedClaim,
    MarkdownExtraction,
)
from research_pipeline.models.quality import QualityScore
from research_pipeline.models.query_plan import QueryPlan, SparsityThresholds
from research_pipeline.models.screening import CheapScoreBreakdown, LLMJudgment
from research_pipeline.models.summary import (
    PaperSummary,
    SummaryEvidence,
    SynthesisReport,
)

# ---------------------------------------------------------------------------
# Reusable strategies
# ---------------------------------------------------------------------------

_nonempty_text = st.text(
    min_size=1, max_size=80, alphabet=st.characters(categories=("L", "N", "P", "Z"))
)
_short_text = st.text(
    min_size=1, max_size=40, alphabet=st.characters(categories=("L", "N", "P", "Z"))
)
_optional_text = st.none() | _short_text
_url = st.from_regex(r"https://[a-z]{3,12}\.[a-z]{2,4}/[a-z0-9/]{1,30}", fullmatch=True)
_sha256 = st.from_regex(r"[0-9a-f]{64}", fullmatch=True)
_score_0_1 = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
_nonneg_float = st.floats(
    min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
)
_nonneg_int = st.integers(min_value=0, max_value=10**7)
_pos_int = st.integers(min_value=1, max_value=10**7)
_small_list = lambda s: st.lists(s, min_size=0, max_size=5)  # noqa: E731
_nonempty_list = lambda s: st.lists(s, min_size=1, max_size=5)  # noqa: E731
_aware_datetimes = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2099, 12, 31),
    timezones=st.just(UTC),
)

# ---------------------------------------------------------------------------
# Model strategies
# ---------------------------------------------------------------------------

st_sparsity = st.builds(
    SparsityThresholds,
    min_candidates=_pos_int,
    min_highscore=_pos_int,
    min_downloads=_pos_int,
)

st_query_plan = st.builds(
    QueryPlan,
    topic_raw=_nonempty_text,
    topic_normalized=_nonempty_text,
    must_terms=_small_list(_short_text),
    nice_terms=_small_list(_short_text),
    negative_terms=_small_list(_short_text),
    candidate_categories=_small_list(_short_text),
    query_variants=_small_list(_short_text),
    primary_months=_pos_int,
    fallback_months=_pos_int,
    sparsity_thresholds=st_sparsity,
)

st_candidate = st.builds(
    CandidateRecord,
    arxiv_id=_short_text,
    version=_short_text,
    title=_nonempty_text,
    authors=_nonempty_list(_short_text),
    published=_aware_datetimes,
    updated=_aware_datetimes,
    categories=_nonempty_list(_short_text),
    primary_category=_short_text,
    abstract=_nonempty_text,
    abs_url=_url,
    pdf_url=_url,
    source=st.sampled_from(["arxiv", "semantic_scholar", "openalex", "dblp"]),
    doi=_optional_text,
    semantic_scholar_id=_optional_text,
    openalex_id=_optional_text,
    citation_count=st.none() | _nonneg_int,
    influential_citation_count=st.none() | _nonneg_int,
    venue=_optional_text,
    year=st.none() | st.integers(min_value=1900, max_value=2099),
)

st_cheap_score = st.builds(
    CheapScoreBreakdown,
    bm25_title=_nonneg_float,
    bm25_abstract=_nonneg_float,
    cat_match=_nonneg_float,
    negative_penalty=_nonneg_float,
    recency_bonus=_nonneg_float,
    semantic_score=st.none() | _score_0_1,
    cheap_score=_nonneg_float,
)

st_llm_judgment = st.builds(
    LLMJudgment,
    llm_score=_score_0_1,
    label=st.sampled_from(["high", "medium", "low", "off_topic"]),
    rationale=_small_list(_short_text),
    evidence_quotes=st.just([]),
    uncertainties=_small_list(_short_text),
    needs_fulltext_validation=_small_list(_short_text),
)

st_download_entry = st.builds(
    DownloadManifestEntry,
    arxiv_id=_short_text,
    version=_short_text,
    pdf_url=_url,
    local_path=_short_text,
    sha256=_sha256,
    size_bytes=_nonneg_int,
    downloaded_at=_aware_datetimes,
    status=st.sampled_from(["downloaded", "skipped_exists", "failed"]),
    error=_optional_text,
    retry_count=st.integers(min_value=0, max_value=100),
    last_error=_optional_text,
)

st_convert_entry = st.builds(
    ConvertManifestEntry,
    arxiv_id=_short_text,
    version=_short_text,
    pdf_path=_short_text,
    pdf_sha256=_sha256,
    markdown_path=_short_text,
    converter_name=_short_text,
    converter_version=_short_text,
    converter_config_hash=_sha256,
    converted_at=_aware_datetimes,
    warnings=_small_list(_short_text),
    status=st.sampled_from(["converted", "skipped_exists", "failed"]),
    tier=st.sampled_from(["rough", "fine"]),
    error=_optional_text,
    retry_count=st.integers(min_value=0, max_value=100),
    last_error=_optional_text,
)

st_quality_score = st.builds(
    QualityScore,
    paper_id=_short_text,
    citation_impact=_score_0_1,
    venue_tier=st.none() | st.sampled_from(["A*", "A", "B", "C"]),
    venue_score=_score_0_1,
    author_credibility=_score_0_1,
    reproducibility=_score_0_1,
    composite_score=_score_0_1,
    safety_flag=st.none() | st.sampled_from(["retracted", "fabricated"]),
    details=st.just({}),
)

st_summary_evidence = st.builds(
    SummaryEvidence,
    chunk_id=_short_text,
    line_range=_short_text,
    quote=_short_text,
)

st_paper_summary = st.builds(
    PaperSummary,
    arxiv_id=_short_text,
    version=_short_text,
    title=_nonempty_text,
    objective=_nonempty_text,
    methodology=_nonempty_text,
    findings=_small_list(_short_text),
    limitations=_small_list(_short_text),
    evidence=_small_list(st_summary_evidence),
    uncertainties=_small_list(_short_text),
)

st_chunk_metadata = st.builds(
    ChunkMetadata,
    paper_id=_short_text,
    section_path=_short_text,
    chunk_id=_short_text,
    source_span=_short_text,
    token_count=_nonneg_int,
)

st_extracted_claim = st.builds(
    ExtractedClaim,
    claim=_nonempty_text,
    chunk_ids=_nonempty_list(_short_text),
    confidence=_score_0_1,
)

st_markdown_extraction = st.builds(
    MarkdownExtraction,
    arxiv_id=_short_text,
    version=_short_text,
    chunks=_small_list(st_chunk_metadata),
    claims=_small_list(st_extracted_claim),
    sections=_small_list(_short_text),
)

# ---------------------------------------------------------------------------
# Roundtrip helpers
# ---------------------------------------------------------------------------


def _assert_json_roundtrip(model_cls, instance):
    """Verify JSON serialization roundtrip produces an equal object."""
    json_bytes = instance.model_dump_json()
    rebuilt = model_cls.model_validate_json(json_bytes)
    assert rebuilt == instance


def _assert_dict_roundtrip(model_cls, instance):
    """Verify dict serialization roundtrip produces an equal object."""
    data = instance.model_dump()
    rebuilt = model_cls.model_validate(data)
    assert rebuilt == instance


# ---------------------------------------------------------------------------
# Tests — QueryPlan & SparsityThresholds
# ---------------------------------------------------------------------------


class TestQueryPlanProperties:
    @given(instance=st_sparsity)
    @settings(max_examples=50)
    def test_sparsity_thresholds_json_roundtrip(self, instance):
        _assert_json_roundtrip(SparsityThresholds, instance)

    @given(instance=st_query_plan)
    @settings(max_examples=50)
    def test_query_plan_json_roundtrip(self, instance):
        _assert_json_roundtrip(QueryPlan, instance)

    @given(instance=st_query_plan)
    @settings(max_examples=50)
    def test_query_plan_dict_roundtrip(self, instance):
        _assert_dict_roundtrip(QueryPlan, instance)


# ---------------------------------------------------------------------------
# Tests — CandidateRecord
# ---------------------------------------------------------------------------


class TestCandidateRecordProperties:
    @given(instance=st_candidate)
    @settings(max_examples=50)
    def test_json_roundtrip(self, instance):
        _assert_json_roundtrip(CandidateRecord, instance)

    @given(instance=st_candidate)
    @settings(max_examples=50)
    def test_dict_roundtrip(self, instance):
        _assert_dict_roundtrip(CandidateRecord, instance)

    @given(instance=st_candidate)
    @settings(max_examples=50)
    def test_required_fields_present(self, instance):
        data = instance.model_dump()
        for field in ("arxiv_id", "title", "authors", "abstract"):
            assert field in data


# ---------------------------------------------------------------------------
# Tests — CheapScoreBreakdown & LLMJudgment
# ---------------------------------------------------------------------------


class TestScreeningProperties:
    @given(instance=st_cheap_score)
    @settings(max_examples=50)
    def test_cheap_score_json_roundtrip(self, instance):
        _assert_json_roundtrip(CheapScoreBreakdown, instance)

    @given(instance=st_cheap_score)
    @settings(max_examples=50)
    def test_cheap_score_dict_roundtrip(self, instance):
        _assert_dict_roundtrip(CheapScoreBreakdown, instance)

    @given(instance=st_llm_judgment)
    @settings(max_examples=50)
    def test_llm_judgment_json_roundtrip(self, instance):
        _assert_json_roundtrip(LLMJudgment, instance)

    @given(instance=st_llm_judgment)
    @settings(max_examples=50)
    def test_llm_score_in_range(self, instance):
        assert 0.0 <= instance.llm_score <= 1.0


# ---------------------------------------------------------------------------
# Tests — DownloadManifestEntry
# ---------------------------------------------------------------------------


class TestDownloadEntryProperties:
    @given(instance=st_download_entry)
    @settings(max_examples=50)
    def test_json_roundtrip(self, instance):
        _assert_json_roundtrip(DownloadManifestEntry, instance)

    @given(instance=st_download_entry)
    @settings(max_examples=50)
    def test_dict_roundtrip(self, instance):
        _assert_dict_roundtrip(DownloadManifestEntry, instance)

    @given(instance=st_download_entry)
    @settings(max_examples=50)
    def test_status_is_valid_literal(self, instance):
        assert instance.status in {"downloaded", "skipped_exists", "failed"}


# ---------------------------------------------------------------------------
# Tests — ConvertManifestEntry
# ---------------------------------------------------------------------------


class TestConvertEntryProperties:
    @given(instance=st_convert_entry)
    @settings(max_examples=50)
    def test_json_roundtrip(self, instance):
        _assert_json_roundtrip(ConvertManifestEntry, instance)

    @given(instance=st_convert_entry)
    @settings(max_examples=50)
    def test_dict_roundtrip(self, instance):
        _assert_dict_roundtrip(ConvertManifestEntry, instance)

    @given(instance=st_convert_entry)
    @settings(max_examples=50)
    def test_tier_is_valid(self, instance):
        assert instance.tier in {"rough", "fine"}


# ---------------------------------------------------------------------------
# Tests — QualityScore
# ---------------------------------------------------------------------------


class TestQualityScoreProperties:
    @given(instance=st_quality_score)
    @settings(max_examples=50)
    def test_json_roundtrip(self, instance):
        _assert_json_roundtrip(QualityScore, instance)

    @given(instance=st_quality_score)
    @settings(max_examples=50)
    def test_dict_roundtrip(self, instance):
        _assert_dict_roundtrip(QualityScore, instance)

    @given(instance=st_quality_score)
    @settings(max_examples=50)
    def test_scores_in_unit_range(self, instance):
        for score in (
            instance.citation_impact,
            instance.venue_score,
            instance.author_credibility,
            instance.reproducibility,
            instance.composite_score,
        ):
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests — PaperSummary
# ---------------------------------------------------------------------------


class TestPaperSummaryProperties:
    @given(instance=st_paper_summary)
    @settings(max_examples=50)
    def test_json_roundtrip(self, instance):
        _assert_json_roundtrip(PaperSummary, instance)

    @given(instance=st_paper_summary)
    @settings(max_examples=50)
    def test_dict_roundtrip(self, instance):
        _assert_dict_roundtrip(PaperSummary, instance)


# ---------------------------------------------------------------------------
# Tests — Extraction models
# ---------------------------------------------------------------------------


class TestExtractionProperties:
    @given(instance=st_chunk_metadata)
    @settings(max_examples=50)
    def test_chunk_metadata_json_roundtrip(self, instance):
        _assert_json_roundtrip(ChunkMetadata, instance)

    @given(instance=st_extracted_claim)
    @settings(max_examples=50)
    def test_extracted_claim_json_roundtrip(self, instance):
        _assert_json_roundtrip(ExtractedClaim, instance)

    @given(instance=st_extracted_claim)
    @settings(max_examples=50)
    def test_claim_confidence_in_range(self, instance):
        assert 0.0 <= instance.confidence <= 1.0

    @given(instance=st_markdown_extraction)
    @settings(max_examples=50)
    def test_markdown_extraction_json_roundtrip(self, instance):
        _assert_json_roundtrip(MarkdownExtraction, instance)

    @given(instance=st_markdown_extraction)
    @settings(max_examples=50)
    def test_markdown_extraction_dict_roundtrip(self, instance):
        _assert_dict_roundtrip(MarkdownExtraction, instance)


# ---------------------------------------------------------------------------
# Tests — SynthesisReport (composite)
# ---------------------------------------------------------------------------


class TestSynthesisReportProperties:
    @given(
        instance=st.builds(
            SynthesisReport,
            topic=_nonempty_text,
            paper_count=_nonneg_int,
            agreements=st.just([]),
            disagreements=st.just([]),
            open_questions=_small_list(_short_text),
            paper_summaries=_small_list(st_paper_summary),
        )
    )
    @settings(max_examples=50)
    def test_synthesis_report_json_roundtrip(self, instance):
        _assert_json_roundtrip(SynthesisReport, instance)
