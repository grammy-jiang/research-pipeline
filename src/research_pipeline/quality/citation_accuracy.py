"""Claim-level citation accuracy scoring.

Measures the percentage of claims in a synthesis report that have
valid source paper citations.  Parses the structured evidence output
and checks each claim's ``supporting_papers`` against the set of
known paper IDs from the report.

Metrics produced:
- **citation_rate**: fraction of claims with ≥1 valid citation
- **avg_citations_per_claim**: mean number of valid citations
- **orphan_claims**: claims with zero valid citations
- **phantom_citations**: references to paper IDs not in the report

References:
    Deep-research report Theme 13 (Evidence-Based Reporting) and
    Engineering Gap 9 (Citation Verification).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from research_pipeline.models.summary import SynthesisReport

logger = logging.getLogger(__name__)


@dataclass
class CitationAccuracyResult:
    """Result of citation accuracy analysis.

    Attributes:
        total_claims: Number of claims analyzed.
        cited_claims: Number of claims with ≥1 valid citation.
        citation_rate: Fraction of claims with valid citations (0–1).
        avg_citations: Mean number of valid citations per claim.
        orphan_claims: List of claim statements with zero citations.
        phantom_refs: Set of paper IDs referenced but not in the report.
        per_claim: Per-claim detail dicts.
    """

    total_claims: int = 0
    cited_claims: int = 0
    citation_rate: float = 0.0
    avg_citations: float = 0.0
    orphan_claims: list[str] = field(default_factory=list)
    phantom_refs: set[str] = field(default_factory=set)
    per_claim: list[dict[str, Any]] = field(default_factory=list)


def score_citation_accuracy(report: SynthesisReport) -> CitationAccuracyResult:
    """Score citation accuracy for a synthesis report.

    Examines agreements, disagreements, and per-paper findings to
    determine how many claims are properly backed by citations.

    Args:
        report: The synthesis report to analyze.

    Returns:
        :class:`CitationAccuracyResult` with metrics.
    """
    known_ids = {ps.arxiv_id for ps in report.paper_summaries}
    claims: list[dict[str, Any]] = []

    # Agreements
    for ag in report.agreements:
        claims.append(
            {
                "type": "agreement",
                "statement": ag.claim,
                "paper_ids": list(ag.supporting_papers),
            }
        )

    # Disagreements
    for dg in report.disagreements:
        claims.append(
            {
                "type": "disagreement",
                "statement": f"Disagreement on: {dg.topic}",
                "paper_ids": list(dg.positions.keys()),
            }
        )

    # Per-paper findings
    for ps in report.paper_summaries:
        for finding in ps.findings:
            claims.append(
                {
                    "type": "finding",
                    "statement": finding,
                    "paper_ids": [ps.arxiv_id],
                }
            )

    if not claims:
        return CitationAccuracyResult()

    total = len(claims)
    cited = 0
    total_valid_refs = 0
    orphans: list[str] = []
    phantoms: set[str] = set()
    per_claim: list[dict[str, Any]] = []

    for claim in claims:
        refs = claim["paper_ids"]
        valid_refs = [r for r in refs if r in known_ids]
        invalid_refs = [r for r in refs if r not in known_ids]

        phantoms.update(invalid_refs)

        has_citation = len(valid_refs) > 0
        if has_citation:
            cited += 1
        else:
            orphans.append(claim["statement"])

        total_valid_refs += len(valid_refs)

        per_claim.append(
            {
                "type": claim["type"],
                "statement": claim["statement"],
                "valid_refs": valid_refs,
                "invalid_refs": invalid_refs,
                "has_citation": has_citation,
            }
        )

    rate = cited / total if total > 0 else 0.0
    avg = total_valid_refs / total if total > 0 else 0.0

    result = CitationAccuracyResult(
        total_claims=total,
        cited_claims=cited,
        citation_rate=rate,
        avg_citations=avg,
        orphan_claims=orphans,
        phantom_refs=phantoms,
        per_claim=per_claim,
    )

    logger.info(
        "Citation accuracy: %.1f%% (%d/%d claims cited, %.1f avg refs)",
        rate * 100,
        cited,
        total,
        avg,
    )

    return result


def score_text_citations(
    text: str,
    known_ids: set[str],
) -> CitationAccuracyResult:
    """Score citation accuracy in free-text (markdown report).

    Extracts arXiv-style IDs from the text and checks them against
    the set of known paper IDs.

    Args:
        text: The report text to analyze.
        known_ids: Set of valid paper identifiers.

    Returns:
        :class:`CitationAccuracyResult` with metrics.
    """
    # Extract sentences as claims (split on sentence boundaries)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if len(s) > 20]  # skip short fragments

    if not sentences:
        return CitationAccuracyResult()

    # Pattern for arXiv IDs: 2401.01234, arxiv:2401.01234
    arxiv_pattern = re.compile(r"(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)

    total = len(sentences)
    cited = 0
    total_valid_refs = 0
    orphans: list[str] = []
    phantoms: set[str] = set()
    per_claim: list[dict[str, Any]] = []

    for sentence in sentences:
        refs = arxiv_pattern.findall(sentence)
        valid_refs = [r for r in refs if r in known_ids]
        invalid_refs = [r for r in refs if r not in known_ids]

        phantoms.update(invalid_refs)

        has_citation = len(valid_refs) > 0
        if has_citation:
            cited += 1
        else:
            orphans.append(sentence[:100])

        total_valid_refs += len(valid_refs)

        per_claim.append(
            {
                "type": "sentence",
                "statement": sentence[:100],
                "valid_refs": valid_refs,
                "invalid_refs": invalid_refs,
                "has_citation": has_citation,
            }
        )

    rate = cited / total if total > 0 else 0.0
    avg = total_valid_refs / total if total > 0 else 0.0

    return CitationAccuracyResult(
        total_claims=total,
        cited_claims=cited,
        citation_rate=rate,
        avg_citations=avg,
        orphan_claims=orphans,
        phantom_refs=phantoms,
        per_claim=per_claim,
    )
