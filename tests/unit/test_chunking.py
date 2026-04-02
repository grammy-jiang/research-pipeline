"""Unit tests for extraction.chunking module."""

from arxiv_paper_pipeline.extraction.chunking import chunk_markdown


class TestChunkMarkdown:
    def test_single_section(self) -> None:
        md = "# Introduction\n\nThis is the introduction paragraph."
        chunks = chunk_markdown(md, "2401.12345")
        assert len(chunks) >= 1
        meta, text = chunks[0]
        assert meta.paper_id == "2401.12345"
        assert "Introduction" in meta.section_path

    def test_multiple_sections(self) -> None:
        md = (
            "# Introduction\n\nIntro text.\n\n"
            "# Methods\n\nMethods text.\n\n"
            "# Results\n\nResults text.\n"
        )
        chunks = chunk_markdown(md, "2401.12345")
        assert len(chunks) == 3
        sections = [meta.section_path for meta, _ in chunks]
        assert "Introduction" in sections
        assert "Methods" in sections
        assert "Results" in sections

    def test_no_headings(self) -> None:
        md = "This is a document without any headings.\n\nJust plain text."
        chunks = chunk_markdown(md, "2401.12345")
        assert len(chunks) >= 1

    def test_chunk_ids_unique(self) -> None:
        md = (
            "# Section A\n\nText A.\n\n"
            "# Section B\n\nText B.\n\n"
            "# Section C\n\nText C.\n"
        )
        chunks = chunk_markdown(md, "paper1")
        chunk_ids = [meta.chunk_id for meta, _ in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_token_count_populated(self) -> None:
        md = "# Test\n\nSome content here with multiple words."
        chunks = chunk_markdown(md, "2401.12345")
        for meta, _ in chunks:
            assert meta.token_count > 0

    def test_large_section_splits(self) -> None:
        # Create a section larger than max_tokens
        large_text = "\n\n".join(
            [f"Paragraph {i}: " + "word " * 200 for i in range(20)]
        )
        md = f"# Large Section\n\n{large_text}"
        chunks = chunk_markdown(md, "2401.12345", max_tokens=100)
        assert len(chunks) > 1

    def test_empty_document(self) -> None:
        chunks = chunk_markdown("", "2401.12345")
        # Should handle gracefully
        assert isinstance(chunks, list)

    def test_nested_headings(self) -> None:
        md = (
            "# Main\n\nMain text.\n\n"
            "## Sub 1\n\nSub 1 text.\n\n"
            "## Sub 2\n\nSub 2 text.\n"
        )
        chunks = chunk_markdown(md, "2401.12345")
        assert len(chunks) == 3
