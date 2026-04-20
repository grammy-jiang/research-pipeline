"""Recall / Reasoning / Presentation (RRP) diagnostic — Theme 16 finding.

Operationalizes the DeepResearch Bench II finding (Deep Research Report
§Theme 16): deep-research agent quality decomposes into three orthogonal
axes — Information Recall, Reasoning, and Presentation — and Information
Recall is the dominant bottleneck (<50% of expert rubrics satisfied).

The diagnostic accepts a *synthesis payload* — the structured synthesis
report plus the shortlist of papers that were supposed to inform it — and
returns three scores in ``[0, 1]``:

- **Recall**:       fraction of shortlisted papers referenced in the report
                    (coverage) × citation density (uniqueness).
- **Reasoning**:    presence of contradiction handling, gap classification,
                    and confidence-graded claims.
- **Presentation**: structural completeness (required sections present,
                    adequate length, citation markers).

Returning all three lets users quickly see where to invest effort. Per the
report, presentation is usually near-saturated while recall lags; a run
with R < 0.5 should trigger retrieval-side improvements before prose work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RRPDiagnostic",
    "compute_rrp_diagnostic",
]


_REQUIRED_SECTIONS = (
    "executive summary",
    "themes",
    "contradictions",
    "gaps",
    "confidence",
)

_CONTRADICTION_HINTS = ("contradict", "disagree", "conflict", "inconsistent")
_GAP_HINTS = ("gap", "unknown", "future work", "open question", "unaddressed")
_CONFIDENCE_HINTS = ("high confidence", "medium confidence", "low confidence")


@dataclass
class RRPDiagnostic:
    """Recall / Reasoning / Presentation scores + overall composite."""

    recall: float
    reasoning: float
    presentation: float
    overall: float
    bottleneck: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "recall": self.recall,
            "reasoning": self.reasoning,
            "presentation": self.presentation,
            "overall": self.overall,
            "bottleneck": self.bottleneck,
            "details": dict(self.details),
        }


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _score_recall(
    report_text: str, shortlist_ids: list[str]
) -> tuple[float, dict[str, Any]]:
    """Fraction of shortlisted papers whose IDs appear in the report."""
    if not shortlist_ids:
        return 0.0, {"shortlist_size": 0, "referenced": 0}
    referenced = sum(1 for pid in shortlist_ids if pid and pid in report_text)
    coverage = referenced / len(shortlist_ids)
    # Citation density: unique [n] markers / 500 words, capped at 1.0
    citation_markers = len(set(re.findall(r"\[\d+\]", report_text)))
    words = max(len(report_text.split()), 1)
    density = min(1.0, citation_markers / max(words / 500, 1.0))
    score = _clamp(0.7 * coverage + 0.3 * density)
    return score, {
        "shortlist_size": len(shortlist_ids),
        "referenced": referenced,
        "coverage": coverage,
        "citation_density": density,
        "unique_citation_markers": citation_markers,
    }


def _score_reasoning(report_text: str) -> tuple[float, dict[str, Any]]:
    """Check for contradiction handling, gap classification, confidence grading."""
    lower = report_text.lower()
    has_contradictions = any(h in lower for h in _CONTRADICTION_HINTS)
    has_gaps = any(h in lower for h in _GAP_HINTS)
    has_confidence = any(h in lower for h in _CONFIDENCE_HINTS)
    score = _clamp(
        0.35 * float(has_contradictions)
        + 0.35 * float(has_gaps)
        + 0.30 * float(has_confidence)
    )
    return score, {
        "contradictions_handled": has_contradictions,
        "gaps_classified": has_gaps,
        "confidence_graded": has_confidence,
    }


def _score_presentation(report_text: str) -> tuple[float, dict[str, Any]]:
    """Structural completeness: required sections + adequate length."""
    lower = report_text.lower()
    sections_present = [s for s in _REQUIRED_SECTIONS if s in lower]
    section_ratio = len(sections_present) / len(_REQUIRED_SECTIONS)
    # Length adequacy: 500 words minimum, 2000 saturates
    words = len(report_text.split())
    length_score = _clamp((words - 500) / 1500) if words >= 500 else 0.0
    score = _clamp(0.7 * section_ratio + 0.3 * length_score)
    return score, {
        "sections_present": sections_present,
        "sections_required": list(_REQUIRED_SECTIONS),
        "word_count": words,
        "length_score": length_score,
    }


def compute_rrp_diagnostic(
    report_text: str,
    shortlist_ids: list[str],
) -> RRPDiagnostic:
    """Compute Recall / Reasoning / Presentation diagnostic for a run.

    Args:
        report_text: The rendered synthesis report (markdown or plain text).
        shortlist_ids: IDs (arXiv IDs or paper IDs) of papers that were
            supposed to inform the synthesis.

    Returns:
        ``RRPDiagnostic`` with scores, a bottleneck label, and details.
    """
    recall, recall_detail = _score_recall(report_text, shortlist_ids)
    reasoning, reasoning_detail = _score_reasoning(report_text)
    presentation, presentation_detail = _score_presentation(report_text)

    overall = _clamp((recall + reasoning + presentation) / 3.0)

    # Identify bottleneck (lowest axis), consistent with Theme 16 guidance
    axes = {"recall": recall, "reasoning": reasoning, "presentation": presentation}
    bottleneck = min(axes, key=lambda k: axes[k])

    return RRPDiagnostic(
        recall=recall,
        reasoning=reasoning,
        presentation=presentation,
        overall=overall,
        bottleneck=bottleneck,
        details={
            "recall": recall_detail,
            "reasoning": reasoning_detail,
            "presentation": presentation_detail,
        },
    )
