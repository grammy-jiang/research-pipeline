"""Unit tests for arxiv.query_builder module."""

from arxiv_paper_pipeline.arxiv.query_builder import (
    build_api_url,
    build_category_filter,
    build_field_query,
    build_negative_filter,
    build_query_from_plan,
    canonical_cache_key,
)
from arxiv_paper_pipeline.models.query_plan import QueryPlan


class TestBuildFieldQuery:
    def test_single_term(self) -> None:
        result = build_field_query("ti", ["neural networks"])
        assert result == 'ti:"neural networks"'

    def test_multiple_terms_and(self) -> None:
        result = build_field_query("abs", ["neural", "network"], "AND")
        assert result == 'abs:"neural" AND abs:"network"'

    def test_multiple_terms_or(self) -> None:
        result = build_field_query("all", ["neural", "network"], "OR")
        assert result == 'all:"neural" OR all:"network"'

    def test_empty_terms(self) -> None:
        result = build_field_query("ti", [])
        assert result == ""

    def test_blank_terms_filtered(self) -> None:
        result = build_field_query("ti", ["neural", "  ", ""])
        assert result == 'ti:"neural"'

    def test_quotes_stripped(self) -> None:
        result = build_field_query("ti", ['"quoted term"'])
        assert result == 'ti:"quoted term"'


class TestBuildCategoryFilter:
    def test_single_category(self) -> None:
        result = build_category_filter(["cs.IR"])
        assert result == "cat:cs.IR"

    def test_multiple_categories(self) -> None:
        result = build_category_filter(["cs.IR", "cs.CL", "cs.AI"])
        assert result == "cat:cs.IR OR cat:cs.CL OR cat:cs.AI"

    def test_empty_categories(self) -> None:
        result = build_category_filter([])
        assert result == ""


class TestBuildNegativeFilter:
    def test_single_negative(self) -> None:
        result = build_negative_filter(["survey"])
        assert result == 'ANDNOT all:"survey"'

    def test_multiple_negatives(self) -> None:
        result = build_negative_filter(["survey", "tutorial"])
        assert result == 'ANDNOT all:"survey" ANDNOT all:"tutorial"'

    def test_empty_negatives(self) -> None:
        result = build_negative_filter([])
        assert result == ""

    def test_blank_negatives_filtered(self) -> None:
        result = build_negative_filter(["survey", "  "])
        assert result == 'ANDNOT all:"survey"'


class TestBuildQueryFromPlan:
    def test_pre_defined_variants(self) -> None:
        plan = QueryPlan(
            topic_raw="test topic",
            topic_normalized="test topic",
            query_variants=["ti:test", "abs:test"],
        )
        result = build_query_from_plan(plan)
        assert result == ["ti:test", "abs:test"]

    def test_must_terms_only(self) -> None:
        plan = QueryPlan(
            topic_raw="neural search",
            topic_normalized="neural search",
            must_terms=["neural", "search"],
        )
        result = build_query_from_plan(plan)
        assert len(result) >= 1
        # First variant should use title + abstract
        assert "ti:" in result[0]
        assert "abs:" in result[0]

    def test_must_and_nice_terms(self) -> None:
        plan = QueryPlan(
            topic_raw="neural search",
            topic_normalized="neural search",
            must_terms=["neural"],
            nice_terms=["embedding", "vector"],
        )
        result = build_query_from_plan(plan)
        assert len(result) >= 2  # At least must-only and must+nice variants

    def test_with_categories(self) -> None:
        plan = QueryPlan(
            topic_raw="neural search",
            topic_normalized="neural search",
            must_terms=["neural"],
            candidate_categories=["cs.IR"],
        )
        result = build_query_from_plan(plan)
        assert any("cat:cs.IR" in q for q in result)

    def test_with_negative_terms(self) -> None:
        plan = QueryPlan(
            topic_raw="neural search",
            topic_normalized="neural search",
            must_terms=["neural"],
            negative_terms=["survey"],
        )
        result = build_query_from_plan(plan)
        assert any("ANDNOT" in q for q in result)

    def test_fallback_on_empty(self) -> None:
        plan = QueryPlan(
            topic_raw="test",
            topic_normalized="test topic",
        )
        result = build_query_from_plan(plan)
        assert len(result) == 1
        assert "test topic" in result[0]

    def test_nice_terms_only(self) -> None:
        plan = QueryPlan(
            topic_raw="test",
            topic_normalized="test",
            nice_terms=["a", "b", "c", "d"],
        )
        result = build_query_from_plan(plan)
        assert len(result) >= 1


class TestBuildApiUrl:
    def test_basic_url(self) -> None:
        url = build_api_url("ti:test", start=0, max_results=10)
        assert "search_query=ti:test" in url
        assert "start=0" in url
        assert "max_results=10" in url
        assert url.startswith("https://export.arxiv.org/api/query")

    def test_with_date_window(self) -> None:
        url = build_api_url(
            "ti:test",
            date_from="202401010000",
            date_to="202407010000",
        )
        assert "submittedDate:" in url
        assert "202401010000" in url
        assert "202407010000" in url

    def test_sort_parameters(self) -> None:
        url = build_api_url(
            "ti:test",
            sort_by="relevance",
            sort_order="ascending",
        )
        assert "sortBy=relevance" in url
        assert "sortOrder=ascending" in url

    def test_custom_base_url(self) -> None:
        url = build_api_url(
            "ti:test",
            base_url="http://localhost:8080/api",
        )
        assert url.startswith("http://localhost:8080/api")


class TestCanonicalCacheKey:
    def test_deterministic(self) -> None:
        key1 = canonical_cache_key("q", 0, 100, "relevance", "desc", None, None)
        key2 = canonical_cache_key("q", 0, 100, "relevance", "desc", None, None)
        assert key1 == key2

    def test_different_queries(self) -> None:
        key1 = canonical_cache_key("q1", 0, 100, "relevance", "desc", None, None)
        key2 = canonical_cache_key("q2", 0, 100, "relevance", "desc", None, None)
        assert key1 != key2

    def test_with_dates(self) -> None:
        key = canonical_cache_key(
            "q",
            0,
            100,
            "relevance",
            "desc",
            "202401010000",
            "202407010000",
        )
        assert "202401010000" in key
        assert "202407010000" in key

    def test_without_dates(self) -> None:
        key = canonical_cache_key("q", 0, 100, "relevance", "desc", None, None)
        assert "df=" not in key
        assert "dt=" not in key
