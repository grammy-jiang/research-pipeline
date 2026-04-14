"""Claim decomposition: break paper summaries into atomic claims."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from research_pipeline.extraction.chunking import chunk_markdown
from research_pipeline.extraction.retrieval import retrieve_relevant_chunks
from research_pipeline.llm.base import LLMProvider
from research_pipeline.models.claim import (
    AtomicClaim,
    ClaimDecomposition,
    ClaimEvidence,
    EvidenceClass,
)
from research_pipeline.models.extraction import ChunkMetadata
from research_pipeline.models.summary import PaperSummary

logger = logging.getLogger(__name__)


def _split_into_atomic(text: str) -> list[str]:
    """Split a compound statement into atomic claims using heuristics.

    Splits on:
    - Semicolons
    - "and" connecting independent clauses (followed by uppercase)
    - "while"/"whereas" introducing contrast
    - Numbered/bulleted sub-items

    Returns:
        List of trimmed, non-empty strings longer than 10 characters.
    """
    # Split on semicolons first
    parts = re.split(r";\s*", text)

    result: list[str] = []
    for part in parts:
        # Split on ", and " or " and " when followed by a subject
        sub_parts = re.split(r",?\s+and\s+(?=[A-Z])", part)
        for sp in sub_parts:
            # Split on "while"/"whereas" contrast
            contrast = re.split(r",?\s+(?:while|whereas)\s+", sp)
            result.extend(contrast)

    return [s.strip() for s in result if s.strip() and len(s.strip()) > 10]


def _classify_evidence_heuristic(
    claim_text: str,
    chunks: list[tuple[ChunkMetadata, str, float]],
    threshold_supported: float = 0.3,
    threshold_partial: float = 0.15,
) -> tuple[EvidenceClass, list[ClaimEvidence], float]:
    """Classify evidence support using BM25 retrieval scores.

    Args:
        claim_text: The atomic claim text.
        chunks: Retrieved chunks as (metadata, text, score) tuples.
        threshold_supported: Score threshold for SUPPORTED.
        threshold_partial: Score threshold for PARTIAL.

    Returns:
        Tuple of (evidence_class, evidence_list, confidence_score).
    """
    if not chunks:
        return EvidenceClass.UNSUPPORTED, [], 0.0

    max_score = max(score for _, _, score in chunks)
    evidence_list: list[ClaimEvidence] = []

    for meta, text, score in chunks[:5]:
        chunk_id = meta.chunk_id if isinstance(meta, ChunkMetadata) else str(meta)
        quote = text[:200].strip()
        evidence_list.append(
            ClaimEvidence(
                chunk_id=chunk_id,
                relevance_score=round(score, 4),
                quote=quote,
            )
        )

    # Check for conflicting signals (high-scoring chunks with negation)
    negation_patterns = re.compile(
        r"\b(however|but|contrary|not|fail|unable|incorrect|disprove"
        r"|refute|contradict)\b",
        re.IGNORECASE,
    )
    has_conflict = False
    for _, text, score in chunks[:3]:
        if score > threshold_partial and negation_patterns.search(text):
            has_conflict = True
            break

    if has_conflict and max_score > threshold_partial:
        confidence = min(max_score, 0.8)
        return EvidenceClass.CONFLICTING, evidence_list, round(confidence, 4)

    if max_score >= threshold_supported:
        confidence = min(max_score * 1.2, 1.0)
        return EvidenceClass.SUPPORTED, evidence_list, round(confidence, 4)
    elif max_score >= threshold_partial:
        confidence = max_score
        return EvidenceClass.PARTIAL, evidence_list, round(confidence, 4)
    elif max_score > 0.05:
        return EvidenceClass.INCONCLUSIVE, evidence_list, round(max_score * 0.5, 4)
    else:
        return EvidenceClass.UNSUPPORTED, evidence_list, 0.0


def decompose_paper(
    summary: PaperSummary,
    markdown_path: str | None = None,
    chunks: list[tuple[ChunkMetadata, str]] | None = None,
    llm_provider: LLMProvider | None = None,
) -> ClaimDecomposition:
    """Decompose a paper summary into atomic claims with evidence classification.

    Args:
        summary: The paper summary to decompose.
        markdown_path: Path to the paper's markdown file for chunk retrieval.
        chunks: Pre-computed chunks (metadata, text) pairs.
            If None, reads from markdown_path.
        llm_provider: Optional LLM for enhanced decomposition (future use).

    Returns:
        ClaimDecomposition with classified atomic claims.
    """
    all_claims: list[AtomicClaim] = []
    claim_counter = 0

    # Collect source items to decompose
    source_items: list[tuple[str, str]] = []
    source_items.append(("objective", summary.objective))
    source_items.append(("methodology", summary.methodology))
    for finding in summary.findings:
        source_items.append(("finding", finding))
    for limitation in summary.limitations:
        source_items.append(("limitation", limitation))

    # Load chunks from markdown if available
    loaded_chunks: list[tuple[ChunkMetadata, str]] | None = None
    if markdown_path:
        try:
            md_text = Path(markdown_path).read_text(encoding="utf-8")
            loaded_chunks = chunk_markdown(md_text, paper_id=summary.arxiv_id)
        except Exception as exc:
            logger.warning("Could not load markdown for %s: %s", summary.arxiv_id, exc)
    elif chunks:
        loaded_chunks = chunks

    for source_type, text in source_items:
        if not text or not text.strip():
            continue

        # Decompose into atomic claims
        atomic_texts = _split_into_atomic(text)
        if not atomic_texts:
            atomic_texts = [text]  # Keep original if can't split

        for claim_text in atomic_texts:
            claim_counter += 1
            claim_id = f"CL-{claim_counter:03d}"

            # Classify evidence
            evidence_class = EvidenceClass.UNSUPPORTED
            evidence_list: list[ClaimEvidence] = []
            confidence = 0.0

            if loaded_chunks:
                try:
                    query_terms = claim_text.lower().split()
                    relevant = retrieve_relevant_chunks(
                        chunks=loaded_chunks,
                        query_terms=query_terms,
                        top_k=5,
                    )
                    evidence_class, evidence_list, confidence = (
                        _classify_evidence_heuristic(claim_text, relevant)
                    )
                except Exception as exc:
                    logger.warning(
                        "Evidence classification failed for %s: %s", claim_id, exc
                    )

            all_claims.append(
                AtomicClaim(
                    claim_id=claim_id,
                    paper_id=summary.arxiv_id,
                    source_type=source_type,
                    statement=claim_text,
                    evidence_class=evidence_class,
                    evidence=evidence_list,
                    confidence_score=confidence,
                )
            )

    # Build evidence summary counts
    evidence_summary: dict[str, int] = {}
    for claim in all_claims:
        key = claim.evidence_class.value
        evidence_summary[key] = evidence_summary.get(key, 0) + 1

    return ClaimDecomposition(
        paper_id=summary.arxiv_id,
        title=summary.title,
        claims=all_claims,
        total_claims=len(all_claims),
        evidence_summary=evidence_summary,
    )
