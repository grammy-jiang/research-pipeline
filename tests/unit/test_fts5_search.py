"""Unit tests for FTS5 global index full-text search."""

import pytest

from research_pipeline.storage.global_index import GlobalPaperIndex


@pytest.fixture()
def fts_index(tmp_path: object) -> GlobalPaperIndex:
    """Create a GlobalPaperIndex instance in a temp directory."""
    from pathlib import Path

    idx = GlobalPaperIndex(Path(str(tmp_path)) / "papers.db")
    return idx


class TestFTS5Search:
    """Tests for full-text search via FTS5."""

    def test_register_with_abstract(self, fts_index: GlobalPaperIndex) -> None:
        """register_paper() accepts and stores an abstract."""
        fts_index.register_paper(
            arxiv_id="2401.00001",
            title="Attention Is All You Need",
            run_id="run-001",
            stage="search",
            abstract="We propose the transformer architecture.",
        )
        results = fts_index.search_fulltext("transformer")
        assert len(results) == 1
        assert results[0]["arxiv_id"] == "2401.00001"
        assert results[0]["abstract"] == "We propose the transformer architecture."

    def test_search_by_title(self, fts_index: GlobalPaperIndex) -> None:
        """FTS matches on title content."""
        fts_index.register_paper(
            arxiv_id="2401.00002",
            title="BERT pre-training of deep bidirectional transformers",
            run_id="run-001",
            stage="search",
        )
        results = fts_index.search_fulltext("BERT")
        assert len(results) == 1
        assert results[0]["arxiv_id"] == "2401.00002"

    def test_search_by_abstract(self, fts_index: GlobalPaperIndex) -> None:
        """FTS matches on abstract content."""
        fts_index.register_paper(
            arxiv_id="2401.00003",
            title="A Simple Method",
            run_id="run-001",
            stage="search",
            abstract="We introduce a novel contrastive learning approach.",
        )
        results = fts_index.search_fulltext("contrastive learning")
        assert len(results) == 1

    def test_empty_query_returns_empty(self, fts_index: GlobalPaperIndex) -> None:
        """Empty or blank queries return an empty list."""
        fts_index.register_paper(
            arxiv_id="2401.00004",
            title="Some Paper",
            run_id="run-001",
            stage="search",
        )
        assert fts_index.search_fulltext("") == []
        assert fts_index.search_fulltext("   ") == []

    def test_no_match_returns_empty(self, fts_index: GlobalPaperIndex) -> None:
        """Query that matches nothing returns empty list."""
        fts_index.register_paper(
            arxiv_id="2401.00005",
            title="Neural Networks",
            run_id="run-001",
            stage="search",
        )
        results = fts_index.search_fulltext("quantum computing")
        assert results == []

    def test_limit_parameter(self, fts_index: GlobalPaperIndex) -> None:
        """Limit parameter controls maximum results."""
        for i in range(5):
            fts_index.register_paper(
                arxiv_id=f"2401.0000{i}",
                title=f"Attention paper variant {i}",
                run_id="run-001",
                stage="search",
                abstract=f"A study of attention mechanisms version {i}.",
            )
        results = fts_index.search_fulltext("attention", limit=3)
        assert len(results) == 3

    def test_register_without_abstract_still_searchable(
        self, fts_index: GlobalPaperIndex
    ) -> None:
        """Papers registered without abstract are still searchable by title."""
        fts_index.register_paper(
            arxiv_id="2401.00010",
            title="Diffusion Models Beat GANs",
            run_id="run-001",
            stage="search",
        )
        results = fts_index.search_fulltext("diffusion")
        assert len(results) == 1
        assert results[0]["abstract"] is None or results[0]["abstract"] == ""

    def test_multiple_papers_ranked(self, fts_index: GlobalPaperIndex) -> None:
        """Multiple matching papers are returned sorted by relevance."""
        fts_index.register_paper(
            arxiv_id="2401.00020",
            title="Memory Systems for LLMs",
            run_id="run-001",
            stage="search",
            abstract="A long-term memory architecture for language models.",
        )
        fts_index.register_paper(
            arxiv_id="2401.00021",
            title="Some Unrelated Paper",
            run_id="run-001",
            stage="search",
            abstract="Memory is also used in computer hardware.",
        )
        results = fts_index.search_fulltext("memory language models")
        assert len(results) >= 1
        # The first result should be the more relevant paper
        ids = [r["arxiv_id"] for r in results]
        assert "2401.00020" in ids
