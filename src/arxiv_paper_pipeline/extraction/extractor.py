"""LLM-assisted structured content extraction from Markdown papers."""

import logging
from pathlib import Path

from arxiv_paper_pipeline.extraction.chunking import chunk_markdown
from arxiv_paper_pipeline.models.extraction import MarkdownExtraction

logger = logging.getLogger(__name__)


def extract_from_markdown(
    markdown_path: Path,
    arxiv_id: str,
    version: str,
    max_chunk_tokens: int = 1500,
) -> MarkdownExtraction:
    """Extract structured content from a converted Markdown file.

    In lite mode (no LLM), performs section slicing only.

    Args:
        markdown_path: Path to the Markdown file.
        arxiv_id: Paper arXiv ID.
        version: Paper version.
        max_chunk_tokens: Maximum tokens per chunk.

    Returns:
        MarkdownExtraction with chunks and sections.
    """
    text = markdown_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(text, arxiv_id, max_tokens=max_chunk_tokens)

    sections = list(dict.fromkeys(meta.section_path for meta, _ in chunks))

    extraction = MarkdownExtraction(
        arxiv_id=arxiv_id,
        version=version,
        chunks=[meta for meta, _ in chunks],
        claims=[],  # LLM-based extraction to be added later
        sections=sections,
    )

    logger.info(
        "Extracted %d chunks, %d sections from %s",
        len(extraction.chunks),
        len(extraction.sections),
        markdown_path.name,
    )
    return extraction
