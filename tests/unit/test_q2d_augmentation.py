"""Unit tests for screening.q2d_augmentation."""

from research_pipeline.screening.q2d_augmentation import (
    augment_query_plan,
    expand_domain_synonyms,
    generate_q2d_queries,
)


class TestExpandDomainSynonyms:
    """Tests for expand_domain_synonyms."""

    def test_known_acronym_expanded(self) -> None:
        result = expand_domain_synonyms(["llm", "agents"])
        assert "llm" in result
        assert "large language model" in result
        assert "agents" in result

    def test_multiple_acronyms(self) -> None:
        result = expand_domain_synonyms(["rag", "llm"])
        assert "retrieval augmented generation" in result
        assert "large language model" in result

    def test_unknown_terms_preserved(self) -> None:
        result = expand_domain_synonyms(["transformer", "harness"])
        assert "transformer" in result
        assert "harness" in result

    def test_empty_input(self) -> None:
        result = expand_domain_synonyms([])
        assert result == []

    def test_no_duplicates(self) -> None:
        result = expand_domain_synonyms(["llm", "llm"])
        assert result.count("llm") == 1

    def test_case_insensitive_matching(self) -> None:
        result = expand_domain_synonyms(["LLM"])
        assert "LLM" in result
        assert "large language model" in result

    def test_synonym_with_multiple_expansions(self) -> None:
        result = expand_domain_synonyms(["bm25"])
        assert "bm25" in result
        assert "best matching 25" in result
        assert "okapi bm25" in result

    def test_preserves_order(self) -> None:
        result = expand_domain_synonyms(["nlp", "cv", "ml"])
        # Original terms should appear before their expansions
        nlp_idx = result.index("nlp")
        nlp_exp_idx = result.index("natural language processing")
        assert nlp_idx < nlp_exp_idx

    def test_whitespace_terms_ignored(self) -> None:
        result = expand_domain_synonyms(["", "  ", "llm"])
        assert "" not in result
        assert "  " not in result
        assert "llm" in result


class TestGenerateQ2dQueries:
    """Tests for generate_q2d_queries."""

    def test_basic_generation(self) -> None:
        result = generate_q2d_queries(["transformer"], ["attention"], max_queries=3)
        assert len(result) <= 3
        assert len(result) > 0
        assert any("transformer" in q for q in result)

    def test_uses_templates(self) -> None:
        result = generate_q2d_queries(["memory", "systems"], [], max_queries=5)
        assert any("this paper presents" in q for q in result)
        assert any("we propose" in q for q in result)

    def test_max_queries_respected(self) -> None:
        result = generate_q2d_queries(
            ["llm", "agents"], ["memory", "retrieval"], max_queries=2
        )
        assert len(result) <= 2

    def test_empty_terms_returns_empty(self) -> None:
        result = generate_q2d_queries([], [], max_queries=5)
        assert result == []

    def test_must_only(self) -> None:
        result = generate_q2d_queries(["neural", "networks"], [], max_queries=3)
        assert len(result) > 0
        assert all("neural" in q or "networks" in q for q in result)

    def test_nice_only(self) -> None:
        result = generate_q2d_queries([], ["transformer", "attention"], max_queries=3)
        assert len(result) > 0

    def test_no_duplicates(self) -> None:
        result = generate_q2d_queries(["memory"], ["agents"], max_queries=10)
        assert len(result) == len(set(result))

    def test_focused_must_variants_included(self) -> None:
        """When nice_terms exist, should also generate must-only Q2D."""
        result = generate_q2d_queries(["rag"], ["performance"], max_queries=15)
        # Should have variants with just "rag" (no "performance")
        must_only = [q for q in result if "performance" not in q]
        assert len(must_only) >= 1


class TestAugmentQueryPlan:
    """Tests for augment_query_plan."""

    def test_preserves_existing_variants(self) -> None:
        existing = ["query one", "query two"]
        result = augment_query_plan(
            must_terms=["llm"],
            nice_terms=["agents"],
            existing_variants=existing,
            max_total_variants=10,
        )
        assert result[0] == "query one"
        assert result[1] == "query two"

    def test_adds_synonym_expanded_query(self) -> None:
        result = augment_query_plan(
            must_terms=["llm"],
            nice_terms=[],
            existing_variants=["llm search"],
            max_total_variants=10,
        )
        # Should have a variant with "large language model"
        assert any("large language model" in v for v in result)

    def test_adds_q2d_queries(self) -> None:
        result = augment_query_plan(
            must_terms=["transformer"],
            nice_terms=["attention"],
            existing_variants=["transformer attention"],
            max_total_variants=10,
        )
        # Should have Q2D-style variants
        assert any("this paper presents" in v for v in result)

    def test_respects_max_total(self) -> None:
        result = augment_query_plan(
            must_terms=["rag", "llm"],
            nice_terms=["memory", "retrieval", "agent"],
            existing_variants=["rag llm", "rag"],
            max_total_variants=4,
        )
        assert len(result) <= 4

    def test_no_duplicates(self) -> None:
        result = augment_query_plan(
            must_terms=["memory"],
            nice_terms=["agent"],
            existing_variants=["memory agent"],
            max_total_variants=10,
        )
        assert len(result) == len(set(result))

    def test_empty_terms_returns_existing(self) -> None:
        result = augment_query_plan(
            must_terms=[],
            nice_terms=[],
            existing_variants=["existing query"],
            max_total_variants=10,
        )
        assert "existing query" in result

    def test_empty_everything(self) -> None:
        result = augment_query_plan(
            must_terms=[],
            nice_terms=[],
            existing_variants=[],
            max_total_variants=10,
        )
        assert result == []

    def test_integration_full_pipeline(self) -> None:
        """Simulate a realistic query plan augmentation."""
        result = augment_query_plan(
            must_terms=["harness", "engineering"],
            nice_terms=["llm", "agents", "prompting"],
            existing_variants=[
                "harness engineering llm agents prompting",
                "harness engineering",
            ],
            max_total_variants=8,
        )
        assert len(result) >= 2
        assert len(result) <= 8
        # Should contain original + expanded + Q2D
        assert result[0] == "harness engineering llm agents prompting"
