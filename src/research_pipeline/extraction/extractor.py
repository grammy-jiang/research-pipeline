"""Structured content extraction from Markdown papers.

Performs section slicing and heuristic claim extraction from the paper
content. LLM-based extraction (higher precision) is not yet implemented;
when an LLM provider is configured this module should be extended to call
it and populate the claims field with higher-confidence results.
"""

import logging
from pathlib import Path

from research_pipeline.extraction.chunking import chunk_markdown
from research_pipeline.extraction.claims import extract_claims_heuristic
from research_pipeline.models.extraction import MarkdownExtraction

logger = logging.getLogger(__name__)


def extract_from_markdown(
    markdown_path: Path,
    arxiv_id: str,
    version: str,
    max_chunk_tokens: int = 1500,
    max_claims: int = 20,
) -> MarkdownExtraction:
    """Extract structured content from a converted Markdown file.

    Performs section slicing and heuristic claim extraction. Claims are
    extracted from high-signal sections (Abstract, Conclusion, Results) using
    BM25-based retrieval for evidence grounding. LLM-based extraction will
    replace this when a provider is configured.

    Args:
        markdown_path: Path to the Markdown file.
        arxiv_id: Paper arXiv ID.
        version: Paper version.
        max_chunk_tokens: Maximum tokens per chunk.
        max_claims: Maximum number of heuristic claims to extract.

    Returns:
        MarkdownExtraction with chunks, sections, and heuristic claims.
    """
    text = markdown_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(text, arxiv_id, max_tokens=max_chunk_tokens)

    sections = list(dict.fromkeys(meta.section_path for meta, _ in chunks))
    claims = extract_claims_heuristic(chunks, max_claims=max_claims)

    extraction = MarkdownExtraction(
        arxiv_id=arxiv_id,
        version=version,
        chunks=[meta for meta, _ in chunks],
        claims=claims,
        sections=sections,
    )

    logger.info(
        "Extracted %d chunks, %d sections, %d heuristic claims from %s",
        len(extraction.chunks),
        len(extraction.sections),
        len(extraction.claims),
        markdown_path.name,
    )
    return extraction
