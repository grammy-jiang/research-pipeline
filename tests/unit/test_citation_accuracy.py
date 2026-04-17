"""Tests for claim-level citation accuracy scoring (v0.13.25).

Covers:
- score_citation_accuracy() with empty, fully-cited, and partially-cited reports
- Orphan claim detection (claims with no valid citations)
- Phantom reference detection (references to unknown paper IDs)
- Per-paper findings included in claim count
- score_text_citations() with markdown text
- CitationAccuracyResult defaults
"""

from __future__ import annotations

from research_pipeline.models.summary import (
    PaperSummary,
    SynthesisAgreement,
    SynthesisDisagreement,
    SynthesisReport,
)
from research_pipeline.quality.citation_accuracy import (
    CitationAccuracyResult,
    score_citation_accuracy,
    score_text_citations,
)


def _paper(
    arxiv_id: str = "2401.00001",
    title: str = "Paper A",
    findings: list[str] | None = None,
) -> PaperSummary:
    return PaperSummary(
        arxiv_id=arxiv_id,
        version="v1",
        title=title,
        objective="Obj",
        methodology="Method",
        findings=findings or [],
        limitations=[],
        evidence=[],
        uncertainties=[],
    )


def _report(
    papers: list[PaperSummary] | None = None,
    agreements: list[SynthesisAgreement] | None = None,
    disagreements: list[SynthesisDisagreement] | None = None,
) -> SynthesisReport:
    ps = papers or []
    return SynthesisReport(
        topic="test",
        paper_count=len(ps),
        agreements=agreements or [],
        disagreements=disagreements or [],
        open_questions=[],
        paper_summaries=ps,
    )


# ── Empty report ────────────────────────────────────────────────────


class TestEmptyReport:
    """Empty report produces zero-count result."""

    def test_empty(self) -> None:
        result = score_citation_accuracy(_report())
        assert result.total_claims == 0
        assert result.citation_rate == 0.0
        assert result.orphan_claims == []


# ── Fully cited ─────────────────────────────────────────────────────


class TestFullyCited:
    """All claims have valid citations."""

    def test_all_agreements_cited(self) -> None:
        papers = [_paper("2401.00001"), _paper("2401.00002")]
        ag = SynthesisAgreement(
            claim="Claim A",
            supporting_papers=["2401.00001", "2401.00002"],
            evidence=[],
        )
        result = score_citation_accuracy(_report(papers, [ag]))
        assert result.total_claims == 1
        assert result.cited_claims == 1
        assert result.citation_rate == 1.0
        assert result.orphan_claims == []

    def test_findings_are_self_cited(self) -> None:
        papers = [_paper("2401.00001", findings=["F1", "F2"])]
        result = score_citation_accuracy(_report(papers))
        assert result.total_claims == 2
        assert result.cited_claims == 2
        assert result.citation_rate == 1.0


# ── Partially cited ─────────────────────────────────────────────────


class TestPartiallyCited:
    """Mix of cited and orphan claims."""

    def test_mixed(self) -> None:
        papers = [_paper("2401.00001")]
        ag_good = SynthesisAgreement(
            claim="Good claim",
            supporting_papers=["2401.00001"],
            evidence=[],
        )
        ag_bad = SynthesisAgreement(
            claim="Orphan claim",
            supporting_papers=["9999.99999"],  # not in papers
            evidence=[],
        )
        result = score_citation_accuracy(_report(papers, [ag_good, ag_bad]))
        assert result.total_claims == 2
        assert result.cited_claims == 1
        assert result.citation_rate == 0.5
        assert "Orphan claim" in result.orphan_claims


# ── Phantom references ──────────────────────────────────────────────


class TestPhantomRefs:
    """References to paper IDs not in the report."""

    def test_phantom_detected(self) -> None:
        papers = [_paper("2401.00001")]
        ag = SynthesisAgreement(
            claim="Claim",
            supporting_papers=["2401.00001", "9999.00001"],
            evidence=[],
        )
        result = score_citation_accuracy(_report(papers, [ag]))
        assert "9999.00001" in result.phantom_refs
        # Claim still counted as cited (has ≥1 valid ref)
        assert result.cited_claims == 1


# ── Disagreements ───────────────────────────────────────────────────


class TestDisagreements:
    """Disagreement positions become claims."""

    def test_disagreement_cited(self) -> None:
        papers = [_paper("2401.00001"), _paper("2401.00002")]
        dg = SynthesisDisagreement(
            topic="Scaling",
            positions={"2401.00001": "linear", "2401.00002": "log"},
            evidence=[],
        )
        result = score_citation_accuracy(_report(papers, disagreements=[dg]))
        assert result.total_claims == 1
        assert result.cited_claims == 1
        assert result.avg_citations == 2.0


# ── Average citations ──────────────────────────────────────────────


class TestAvgCitations:
    """Average citations per claim."""

    def test_avg_multi_ref(self) -> None:
        papers = [_paper("2401.00001"), _paper("2401.00002")]
        ag1 = SynthesisAgreement(
            claim="C1",
            supporting_papers=["2401.00001", "2401.00002"],
            evidence=[],
        )
        ag2 = SynthesisAgreement(
            claim="C2",
            supporting_papers=["2401.00001"],
            evidence=[],
        )
        result = score_citation_accuracy(_report(papers, [ag1, ag2]))
        assert result.avg_citations == 1.5  # (2+1)/2


# ── Per-claim detail ───────────────────────────────────────────────


class TestPerClaim:
    """per_claim list contains detail dicts."""

    def test_per_claim_populated(self) -> None:
        papers = [_paper("2401.00001")]
        ag = SynthesisAgreement(
            claim="C1",
            supporting_papers=["2401.00001"],
            evidence=[],
        )
        result = score_citation_accuracy(_report(papers, [ag]))
        assert len(result.per_claim) == 1
        assert result.per_claim[0]["has_citation"] is True
        assert result.per_claim[0]["type"] == "agreement"


# ── score_text_citations ───────────────────────────────────────────


class TestTextCitations:
    """score_text_citations() for markdown text."""

    def test_text_with_arxiv_ids(self) -> None:
        text = (
            "Transformers achieve SOTA results (2401.00001). "
            "Memory systems are improving rapidly. "
            "Recent work (2401.00002) shows promising directions."
        )
        known = {"2401.00001", "2401.00002"}
        result = score_text_citations(text, known)
        assert result.cited_claims >= 2

    def test_empty_text(self) -> None:
        result = score_text_citations("", set())
        assert result.total_claims == 0

    def test_phantom_in_text(self) -> None:
        text = "Paper 9999.99999 claims something extraordinary here."
        known = {"2401.00001"}
        result = score_text_citations(text, known)
        assert "9999.99999" in result.phantom_refs

    def test_no_citations_in_text(self) -> None:
        text = "This sentence has no paper references at all in it."
        result = score_text_citations(text, {"2401.00001"})
        assert result.citation_rate == 0.0


# ── CitationAccuracyResult defaults ────────────────────────────────


class TestResultDefaults:
    """CitationAccuracyResult default construction."""

    def test_defaults(self) -> None:
        r = CitationAccuracyResult()
        assert r.total_claims == 0
        assert r.citation_rate == 0.0
        assert r.phantom_refs == set()
