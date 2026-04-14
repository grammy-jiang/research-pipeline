"""Unit tests for Q2D query augmentation in cmd_plan."""

from research_pipeline.cli.cmd_plan import _generate_query_variants


class TestQ2DQueryAugmentation:
    """Tests for document-style query variant generation."""

    def test_q2d_variants_included_when_room(self) -> None:
        """Q2D templates appear when max_variants is high enough."""
        variants = _generate_query_variants(
            must_terms=["memory", "agents"],
            nice_terms=["local"],
            max_variants=10,
        )
        q2d_found = [v for v in variants if v.startswith("this paper presents")]
        assert len(q2d_found) >= 1

    def test_q2d_contains_topic_terms(self) -> None:
        """Q2D variants include the original topic terms."""
        variants = _generate_query_variants(
            must_terms=["transformer", "attention"],
            nice_terms=["efficient"],
            max_variants=10,
        )
        q2d = [v for v in variants if "propose" in v or "paper presents" in v]
        for v in q2d:
            assert "transformer" in v
            assert "attention" in v

    def test_q2d_not_included_when_max_low(self) -> None:
        """When max_variants is small, Q2D may not appear."""
        variants = _generate_query_variants(
            must_terms=["a", "b", "c"],
            nice_terms=["d", "e", "f"],
            max_variants=3,
        )
        assert len(variants) <= 3

    def test_q2d_survey_variant(self) -> None:
        """The 'a survey of' Q2D template is generated."""
        variants = _generate_query_variants(
            must_terms=["rag"],
            nice_terms=["knowledge"],
            max_variants=10,
        )
        survey = [v for v in variants if v.startswith("a survey of")]
        assert len(survey) >= 1

    def test_original_variants_preserved(self) -> None:
        """Existing keyword variants still appear first."""
        variants = _generate_query_variants(
            must_terms=["neural", "retrieval"],
            nice_terms=["augmented"],
            max_variants=10,
        )
        # First variant should be all terms joined
        assert variants[0] == "neural retrieval augmented"

    def test_no_duplicates(self) -> None:
        """No duplicate variants in output."""
        variants = _generate_query_variants(
            must_terms=["memory"],
            nice_terms=[],
            max_variants=10,
        )
        assert len(variants) == len(set(variants))
