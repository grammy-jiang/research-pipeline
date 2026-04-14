"""Per-paper evidence-driven summarization."""

from __future__ import annotations

import logging
from pathlib import Path

from research_pipeline.extraction.chunking import chunk_markdown
from research_pipeline.extraction.retrieval import retrieve_relevant_chunks
from research_pipeline.llm.base import LLMProvider
from research_pipeline.models.summary import PaperSummary, SummaryEvidence

logger = logging.getLogger(__name__)

_PAPER_SUMMARY_PROMPT = """\
You are an expert academic paper analyst. Summarize the following paper.

Paper title: {title}
Research topic: {topic}

Below are the most relevant excerpts from the paper:

{chunks_text}

Respond with a JSON object containing:
- "objective": a concise statement of the paper's main objective
- "methodology": a description of the key methodology or approach
- "findings": a list of key findings (strings)
- "limitations": a list of limitations (strings)
- "uncertainties": a list of uncertain or unresolved items (strings)
"""


def _build_paper_prompt(
    title: str,
    topic_terms: list[str],
    relevant: list[tuple[object, str, float]],
) -> str:
    """Build the LLM prompt for per-paper summarization.

    Args:
        title: Paper title.
        topic_terms: Research topic terms.
        relevant: Retrieved relevant chunks (meta, text, score).

    Returns:
        Formatted prompt string.
    """
    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk_text}"
        for i, (_, chunk_text, _) in enumerate(relevant)
    )
    return _PAPER_SUMMARY_PROMPT.format(
        title=title,
        topic=", ".join(topic_terms),
        chunks_text=chunks_text,
    )


def _parse_llm_paper_response(
    response: dict,  # type: ignore[type-arg]
    arxiv_id: str,
    version: str,
    title: str,
    evidence: list[SummaryEvidence],
) -> PaperSummary:
    """Parse LLM response dict into a PaperSummary.

    Args:
        response: Raw LLM response dict.
        arxiv_id: Paper arXiv ID.
        version: Paper version string.
        title: Paper title.
        evidence: Pre-built evidence references.

    Returns:
        PaperSummary constructed from LLM output.

    Raises:
        KeyError: If required fields are missing.
        TypeError: If field types are wrong.
    """
    objective = str(response["objective"])
    methodology = str(response["methodology"])
    findings = [str(f) for f in response["findings"]]
    limitations = [str(lim) for lim in response["limitations"]]
    uncertainties = [str(u) for u in response["uncertainties"]]

    return PaperSummary(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        objective=objective,
        methodology=f"[LLM] {methodology}",
        findings=findings,
        limitations=limitations,
        evidence=evidence,
        uncertainties=uncertainties,
    )


def summarize_paper(
    markdown_path: Path,
    arxiv_id: str,
    version: str,
    title: str,
    topic_terms: list[str],
    max_chunk_tokens: int = 1500,
    top_k_chunks: int = 10,
    llm_provider: LLMProvider | None = None,
) -> PaperSummary:
    """Generate a summary of a single paper from its Markdown.

    When *llm_provider* is given, the top relevant chunks are sent to the
    LLM for structured summarization.  Falls back to template mode if the
    LLM call fails or if no provider is supplied.

    Args:
        markdown_path: Path to the converted Markdown.
        arxiv_id: Paper arXiv ID.
        version: Paper version.
        title: Paper title.
        topic_terms: Terms for relevance-based chunk retrieval.
        max_chunk_tokens: Maximum tokens per chunk.
        top_k_chunks: Number of top chunks to use.
        llm_provider: Optional LLM provider for summarization.

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

    # --- LLM mode ---
    if llm_provider is not None:
        try:
            prompt = _build_paper_prompt(title, topic_terms, relevant)
            response = llm_provider.call(
                prompt, schema_id="paper_summary", temperature=0.0
            )
            summary = _parse_llm_paper_response(
                response, arxiv_id, version, title, evidence
            )
            logger.info(
                "Generated LLM summary for %s with %d evidence refs",
                arxiv_id,
                len(evidence),
            )
            return summary
        except Exception as exc:
            logger.warning(
                "LLM summarization failed for %s, falling back to template: %s",
                arxiv_id,
                exc,
            )

    # --- Template mode (fallback) ---
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
