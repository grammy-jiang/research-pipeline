"""Structured JSON evidence output for synthesis reports.

Converts a ``SynthesisReport`` into a machine-consumable JSON format with
explicit evidence pointers (claim → paper → chunk → quote).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from research_pipeline import __version__
from research_pipeline.models.summary import (
    SummaryEvidence,
    SynthesisReport,
)

logger = logging.getLogger(__name__)


class EvidenceLink(BaseModel):
    """Single link in an evidence chain pointing to a specific paper chunk."""

    paper_id: str = Field(description="arXiv ID of the source paper.")
    paper_title: str = Field(description="Title of the source paper.")
    chunk_id: str = Field(description="Chunk ID within the paper.")
    quote: str = Field(description="Supporting quote from the chunk.")


class EvidencedClaim(BaseModel):
    """A claim backed by an explicit evidence chain."""

    claim_id: str = Field(description="Unique claim identifier (e.g. C001).")
    claim_type: str = Field(
        description="Type of claim: agreement, disagreement, or finding."
    )
    statement: str = Field(description="The claim text.")
    confidence: str = Field(
        description="Confidence level: high (3+), medium (2), or low (1 paper)."
    )
    supporting_papers: list[str] = Field(
        default_factory=list,
        description="arXiv IDs of papers supporting this claim.",
    )
    evidence_chain: list[EvidenceLink] = Field(
        default_factory=list,
        description="Explicit evidence pointers for this claim.",
    )


class StructuredEvidence(BaseModel):
    """Top-level structured evidence output for a synthesis report."""

    topic: str = Field(description="Research topic.")
    paper_count: int = Field(description="Number of papers synthesized.")
    claims: list[EvidencedClaim] = Field(
        default_factory=list,
        description="All evidenced claims extracted from the report.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved questions from the synthesis.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Generation metadata (timestamp, version).",
    )


def _confidence_level(paper_count: int) -> str:
    """Determine confidence level based on supporting paper count.

    Args:
        paper_count: Number of papers supporting the claim.

    Returns:
        ``"high"`` for 3+, ``"medium"`` for 2, ``"low"`` for 0–1.
    """
    if paper_count >= 3:
        return "high"
    if paper_count == 2:
        return "medium"
    return "low"


def _build_title_lookup(report: SynthesisReport) -> dict[str, str]:
    """Build a mapping from arXiv ID to paper title.

    Args:
        report: The synthesis report containing paper summaries.

    Returns:
        Dictionary mapping arXiv IDs to their titles.
    """
    return {ps.arxiv_id: ps.title for ps in report.paper_summaries}


def _resolve_evidence_chain(
    evidence_items: list[SummaryEvidence],
    paper_id: str,
    title_lookup: dict[str, str],
) -> list[EvidenceLink]:
    """Convert ``SummaryEvidence`` items into ``EvidenceLink`` objects.

    Args:
        evidence_items: Raw evidence references from the report.
        paper_id: arXiv ID to associate with these evidence items.
        title_lookup: Mapping from arXiv ID to paper title.

    Returns:
        List of resolved evidence links.
    """
    return [
        EvidenceLink(
            paper_id=paper_id,
            paper_title=title_lookup.get(paper_id, ""),
            chunk_id=ev.chunk_id,
            quote=ev.quote,
        )
        for ev in evidence_items
    ]


def build_structured_output(report: SynthesisReport) -> StructuredEvidence:
    """Convert a ``SynthesisReport`` into structured evidence JSON.

    Walks agreements, disagreements, and per-paper findings to produce a
    flat list of ``EvidencedClaim`` objects with explicit evidence chains.

    Args:
        report: The synthesis report to convert.

    Returns:
        A ``StructuredEvidence`` instance ready for JSON serialization.
    """
    title_lookup = _build_title_lookup(report)
    claims: list[EvidencedClaim] = []
    claim_counter = 0

    # --- agreements ---
    for agreement in report.agreements:
        claim_counter += 1
        chain: list[EvidenceLink] = []
        for ev in agreement.evidence:
            for paper_id in agreement.supporting_papers:
                chain.append(
                    EvidenceLink(
                        paper_id=paper_id,
                        paper_title=title_lookup.get(paper_id, ""),
                        chunk_id=ev.chunk_id,
                        quote=ev.quote,
                    )
                )
        claims.append(
            EvidencedClaim(
                claim_id=f"C{claim_counter:03d}",
                claim_type="agreement",
                statement=agreement.claim,
                confidence=_confidence_level(len(agreement.supporting_papers)),
                supporting_papers=list(agreement.supporting_papers),
                evidence_chain=chain,
            )
        )

    # --- disagreements ---
    for disagreement in report.disagreements:
        claim_counter += 1
        paper_ids = list(disagreement.positions.keys())
        chain = []
        for ev in disagreement.evidence:
            for paper_id in paper_ids:
                chain.append(
                    EvidenceLink(
                        paper_id=paper_id,
                        paper_title=title_lookup.get(paper_id, ""),
                        chunk_id=ev.chunk_id,
                        quote=ev.quote,
                    )
                )
        positions_text = "; ".join(
            f"{pid}: {pos}" for pid, pos in disagreement.positions.items()
        )
        claims.append(
            EvidencedClaim(
                claim_id=f"C{claim_counter:03d}",
                claim_type="disagreement",
                statement=f"{disagreement.topic} — {positions_text}",
                confidence=_confidence_level(len(paper_ids)),
                supporting_papers=paper_ids,
                evidence_chain=chain,
            )
        )

    # --- per-paper findings ---
    for paper in report.paper_summaries:
        for finding in paper.findings:
            claim_counter += 1
            chain = _resolve_evidence_chain(
                paper.evidence, paper.arxiv_id, title_lookup
            )
            claims.append(
                EvidencedClaim(
                    claim_id=f"C{claim_counter:03d}",
                    claim_type="finding",
                    statement=finding,
                    confidence=_confidence_level(1),
                    supporting_papers=[paper.arxiv_id],
                    evidence_chain=chain,
                )
            )

    metadata = {
        "generation_timestamp": datetime.now(tz=UTC).isoformat(),
        "pipeline_version": __version__,
    }

    result = StructuredEvidence(
        topic=report.topic,
        paper_count=report.paper_count,
        claims=claims,
        open_questions=list(report.open_questions),
        metadata=metadata,
    )
    logger.info(
        "Built structured evidence: %d claims from %d papers",
        len(claims),
        report.paper_count,
    )
    return result


def export_structured_json(
    report: SynthesisReport,
    output_path: Path,
) -> None:
    """Build structured evidence and write it to a JSON file.

    Args:
        report: The synthesis report to convert.
        output_path: Destination file path for the JSON output.
    """
    structured = build_structured_output(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(structured.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Exported structured evidence to %s", output_path)
