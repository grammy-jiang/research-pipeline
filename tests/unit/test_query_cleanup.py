"""Tests for query noise removal (screening/query_cleanup.py)."""

from research_pipeline.screening.query_cleanup import (
    clean_query_terms,
    deduplicate_substrings,
    normalize_terms,
    remove_academic_boilerplate,
    remove_short_terms,
)


class TestRemoveAcademicBoilerplate:
    def test_removes_common_terms(self) -> None:
        terms = ["neural", "method", "retrieval", "novel", "approach"]
        result = remove_academic_boilerplate(terms)
        assert "neural" in result
        assert "retrieval" in result
        assert "method" not in result
        assert "novel" not in result
        assert "approach" not in result

    def test_preserves_domain_terms(self) -> None:
        terms = ["transformer", "attention", "embedding"]
        result = remove_academic_boilerplate(terms)
        assert result == terms

    def test_case_insensitive(self) -> None:
        terms = ["Method", "NOVEL", "transformer"]
        result = remove_academic_boilerplate(terms)
        assert "transformer" in result
        assert len(result) == 1

    def test_empty_list(self) -> None:
        assert remove_academic_boilerplate([]) == []


class TestRemoveShortTerms:
    def test_removes_single_char(self) -> None:
        result = remove_short_terms(["a", "neural", "b"])
        assert "neural" in result
        assert "a" not in result
        assert "b" not in result

    def test_custom_min_length(self) -> None:
        result = remove_short_terms(["ab", "abc", "abcd"], min_length=3)
        assert "ab" not in result
        assert "abc" in result
        assert "abcd" in result

    def test_empty_list(self) -> None:
        assert remove_short_terms([]) == []


class TestDeduplicateSubstrings:
    def test_removes_substring(self) -> None:
        terms = ["neural", "neural network"]
        result = deduplicate_substrings(terms)
        assert "neural network" in result
        assert "neural" not in result

    def test_keeps_independent_terms(self) -> None:
        terms = ["transformer", "attention", "embedding"]
        result = deduplicate_substrings(terms)
        assert len(result) == 3

    def test_single_term(self) -> None:
        assert deduplicate_substrings(["hello"]) == ["hello"]

    def test_empty_list(self) -> None:
        assert deduplicate_substrings([]) == []

    def test_preserves_order(self) -> None:
        terms = ["attention", "self-attention", "transformer"]
        result = deduplicate_substrings(terms)
        # "attention" is substring of "self-attention", should be removed
        assert result == ["self-attention", "transformer"]

    def test_exact_duplicates_kept(self) -> None:
        terms = ["neural", "neural"]
        result = deduplicate_substrings(terms)
        # Both are same, not substring relationship
        assert len(result) == 2


class TestNormalizeTerms:
    def test_lowercases(self) -> None:
        result = normalize_terms(["Neural", "RETRIEVAL"])
        assert result == ["neural", "retrieval"]

    def test_strips_whitespace(self) -> None:
        result = normalize_terms(["  neural  ", "retrieval "])
        assert result == ["neural", "retrieval"]

    def test_deduplicates(self) -> None:
        result = normalize_terms(["neural", "Neural", "NEURAL"])
        assert result == ["neural"]

    def test_removes_empty(self) -> None:
        result = normalize_terms(["", "neural", "  "])
        assert result == ["neural"]

    def test_collapses_whitespace(self) -> None:
        result = normalize_terms(["neural  network"])
        assert result == ["neural network"]


class TestCleanQueryTerms:
    def test_full_pipeline(self) -> None:
        terms = ["novel", "neural", "Method", "retrieval", "neural network"]
        result = clean_query_terms(terms)
        # "novel" and "method" are boilerplate
        # "neural" is substring of "neural network"
        assert "novel" not in result
        assert "method" not in result
        assert "retrieval" in result
        assert "neural network" in result

    def test_no_boilerplate_removal(self) -> None:
        terms = ["model", "training"]
        result = clean_query_terms(terms, remove_boilerplate=False)
        assert "model" in result
        assert "training" in result

    def test_no_dedup(self) -> None:
        terms = ["neural", "neural network"]
        result = clean_query_terms(terms, deduplicate=False)
        assert len(result) == 2

    def test_safety_returns_originals_when_all_removed(self) -> None:
        # All boilerplate → should return originals instead of empty
        terms = ["model", "method", "approach"]
        result = clean_query_terms(terms)
        assert len(result) > 0

    def test_empty_input(self) -> None:
        assert clean_query_terms([]) == []

    def test_preserves_technical_terms(self) -> None:
        terms = ["transformer", "attention", "bert", "gpt"]
        result = clean_query_terms(terms)
        assert result == ["transformer", "attention", "bert", "gpt"]
