"""Per-paper evidence-driven summarization."""

import logging
from pathlib import Path

from research_pipeline.extraction.chunking import chunk_markdown
from research_pipeline.extraction.retrieval import retrieve_relevant_chunks
from research_pipeline.models.summary import PaperSummary, SummaryEvidence

logger = logging.getLogger(__name__)


def summarize_paper(
    markdown_path: Path,
    arxiv_id: str,
    version: str,
    title: str,
    topic_terms: list[str],
    max_chunk_tokens: int = 1500,
    top_k_chunks: int = 10,
) -> PaperSummary:
    """Generate a summary of a single paper from its Markdown.

    In LLM-disabled mode, produces a template-based summary from
    the most relevant chunks.

    Args:
        markdown_path: Path to the converted Markdown.
        arxiv_id: Paper arXiv ID.
        version: Paper version.
        title: Paper title.
        topic_terms: Terms for relevance-based chunk retrieval.
        max_chunk_tokens: Maximum tokens per chunk.
        top_k_chunks: Number of top chunks to use.

    Returns:
        PaperSummary with evidence references.
    """
    text = markdown_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(text, arxiv_id, max_tokens=max_chunk_tokens)
    relevant = retrieve_relevant_chunks(chunks, topic_terms, top_k=top_k_chunks)

    evidence = [
        SummaryEvidence(
            chunk_id=meta.chunk_id,
            line_range=meta.source_span,
            quote=chunk_text[:200],
        )
        for meta, chunk_text, _score in relevant
    ]

    # Template-based summary (no LLM)
    summary = PaperSummary(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        objective=f"See paper: {title}",
        methodology="(LLM summarization not enabled — template mode)",
        findings=[
            f"Relevant section: {meta.section_path}" for meta, _, _ in relevant[:5]
        ],
        limitations=["Template-based summary; enable LLM for detailed analysis."],
        evidence=evidence,
        uncertainties=["Full LLM-based extraction not enabled."],
    )

    logger.info(
        "Generated template summary for %s with %d evidence refs",
        arxiv_id,
        len(evidence),
    )
    return summary
