"""Tests for summarization.structured_output module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_pipeline import __version__
from research_pipeline.models.summary import (
    PaperSummary,
    SummaryEvidence,
    SynthesisAgreement,
    SynthesisDisagreement,
    SynthesisReport,
)
from research_pipeline.summarization.structured_output import (
    StructuredEvidence,
    build_structured_output,
    export_structured_json,
)


def _make_evidence(
    chunk_id: str = "chunk-1", quote: str = "some quote"
) -> SummaryEvidence:
    return SummaryEvidence(chunk_id=chunk_id, quote=quote)


def _make_paper(
    arxiv_id: str = "2301.00001",
    title: str = "Paper A",
    findings: list[str] | None = None,
    evidence: list[SummaryEvidence] | None = None,
) -> PaperSummary:
    return PaperSummary(
        arxiv_id=arxiv_id,
        version="v1",
        title=title,
        objective="Objective",
        methodology="Method",
        findings=findings or [],
        limitations=[],
        evidence=evidence or [],
        uncertainties=[],
    )


def _make_report(
    agreements: list[SynthesisAgreement] | None = None,
    disagreements: list[SynthesisDisagreement] | None = None,
    paper_summaries: list[PaperSummary] | None = None,
    open_questions: list[str] | None = None,
) -> SynthesisReport:
    papers = paper_summaries or []
    return SynthesisReport(
        topic="test topic",
        paper_count=len(papers),
        agreements=agreements or [],
        disagreements=disagreements or [],
        open_questions=open_questions or [],
        paper_summaries=papers,
    )


class TestEmptyReport:
    """An empty report should produce valid output with zero claims."""

    def test_empty_report_zero_claims(self) -> None:
        result = build_structured_output(_make_report())
        assert result.claims == []
        assert result.paper_count == 0
        assert result.topic == "test topic"

    def test_empty_report_valid_metadata(self) -> None:
        result = build_structured_output(_make_report())
        assert "generation_timestamp" in result.metadata
        assert "pipeline_version" in result.metadata


class TestAgreementConversion:
    """Agreements are converted to claims with type='agreement'."""

    def test_agreement_becomes_claim(self) -> None:
        papers = [
            _make_paper("2301.00001", "Paper A"),
            _make_paper("2301.00002", "Paper B"),
        ]
        agreement = SynthesisAgreement(
            claim="Transformers are effective",
            supporting_papers=["2301.00001", "2301.00002"],
            evidence=[_make_evidence("c1", "evidence quote")],
        )
        result = build_structured_output(
            _make_report(agreements=[agreement], paper_summaries=papers)
        )
        assert len(result.claims) == 1
        claim = result.claims[0]
        assert claim.claim_type == "agreement"
        assert claim.statement == "Transformers are effective"
        assert set(claim.supporting_papers) == {"2301.00001", "2301.00002"}

    def test_agreement_evidence_chain_enriched_with_titles(self) -> None:
        papers = [_make_paper("2301.00001", "Paper A")]
        agreement = SynthesisAgreement(
            claim="Claim X",
            supporting_papers=["2301.00001"],
            evidence=[_make_evidence("c1", "quote text")],
        )
        result = build_structured_output(
            _make_report(agreements=[agreement], paper_summaries=papers)
        )
        chain = result.claims[0].evidence_chain
        assert len(chain) == 1
        assert chain[0].paper_title == "Paper A"
        assert chain[0].quote == "quote text"


class TestDisagreementConversion:
    """Disagreements are converted to claims with type='disagreement'."""

    def test_disagreement_becomes_claim(self) -> None:
        papers = [
            _make_paper("2301.00001", "Paper A"),
            _make_paper("2301.00002", "Paper B"),
        ]
        disagreement = SynthesisDisagreement(
            topic="Scaling laws",
            positions={"2301.00001": "linear", "2301.00002": "logarithmic"},
            evidence=[_make_evidence("c2", "disagree quote")],
        )
        result = build_structured_output(
            _make_report(disagreements=[disagreement], paper_summaries=papers)
        )
        assert len(result.claims) == 1
        claim = result.claims[0]
        assert claim.claim_type == "disagreement"
        assert "Scaling laws" in claim.statement
        assert set(claim.supporting_papers) == {"2301.00001", "2301.00002"}


class TestFindingConversion:
    """Per-paper findings are converted to claims with type='finding'."""

    def test_findings_become_claims(self) -> None:
        paper = _make_paper(
            "2301.00001",
            "Paper A",
            findings=["Finding 1", "Finding 2"],
            evidence=[_make_evidence("c3", "finding quote")],
        )
        result = build_structured_output(_make_report(paper_summaries=[paper]))
        finding_claims = [c for c in result.claims if c.claim_type == "finding"]
        assert len(finding_claims) == 2
        assert finding_claims[0].statement == "Finding 1"
        assert finding_claims[1].statement == "Finding 2"

    def test_finding_evidence_chain_uses_paper_evidence(self) -> None:
        paper = _make_paper(
            "2301.00001",
            "Paper A",
            findings=["Finding 1"],
            evidence=[_make_evidence("c3", "paper quote")],
        )
        result = build_structured_output(_make_report(paper_summaries=[paper]))
        chain = result.claims[0].evidence_chain
        assert len(chain) == 1
        assert chain[0].paper_id == "2301.00001"
        assert chain[0].paper_title == "Paper A"
        assert chain[0].chunk_id == "c3"


class TestConfidenceLevels:
    """Confidence is determined by supporting paper count."""

    @pytest.mark.parametrize(
        ("paper_ids", "expected"),
        [
            (["p1"], "low"),
            (["p1", "p2"], "medium"),
            (["p1", "p2", "p3"], "high"),
            (["p1", "p2", "p3", "p4"], "high"),
        ],
    )
    def test_confidence_from_paper_count(
        self, paper_ids: list[str], expected: str
    ) -> None:
        papers = [_make_paper(pid, f"Paper {pid}") for pid in paper_ids]
        agreement = SynthesisAgreement(
            claim="Some claim",
            supporting_papers=paper_ids,
            evidence=[],
        )
        result = build_structured_output(
            _make_report(agreements=[agreement], paper_summaries=papers)
        )
        assert result.claims[0].confidence == expected


class TestClaimIds:
    """Claim IDs must be unique and sequential."""

    def test_ids_are_sequential(self) -> None:
        papers = [_make_paper("2301.00001", "A"), _make_paper("2301.00002", "B")]
        agreement = SynthesisAgreement(
            claim="Agree",
            supporting_papers=["2301.00001"],
            evidence=[],
        )
        disagreement = SynthesisDisagreement(
            topic="Topic",
            positions={"2301.00002": "pos"},
            evidence=[],
        )
        paper_with_finding = _make_paper("2301.00003", "C", findings=["F1"])
        result = build_structured_output(
            _make_report(
                agreements=[agreement],
                disagreements=[disagreement],
                paper_summaries=[*papers, paper_with_finding],
            )
        )
        ids = [c.claim_id for c in result.claims]
        assert ids == ["C001", "C002", "C003"]

    def test_ids_are_unique(self) -> None:
        papers = [
            _make_paper("p1", "A", findings=["F1", "F2"]),
            _make_paper("p2", "B", findings=["F3"]),
        ]
        result = build_structured_output(_make_report(paper_summaries=papers))
        ids = [c.claim_id for c in result.claims]
        assert len(ids) == len(set(ids))


class TestMetadata:
    """Metadata includes timestamp and pipeline version."""

    def test_metadata_has_version(self) -> None:
        result = build_structured_output(_make_report())
        assert result.metadata["pipeline_version"] == __version__

    def test_metadata_has_timestamp(self) -> None:
        result = build_structured_output(_make_report())
        ts = result.metadata["generation_timestamp"]
        assert "T" in ts  # ISO format


class TestJsonRoundtrip:
    """Serialization and deserialization preserve data."""

    def test_roundtrip(self) -> None:
        paper = _make_paper(
            "2301.00001", "Paper A", findings=["F1"], evidence=[_make_evidence()]
        )
        agreement = SynthesisAgreement(
            claim="Agreed",
            supporting_papers=["2301.00001"],
            evidence=[_make_evidence("c2", "agree quote")],
        )
        report = _make_report(agreements=[agreement], paper_summaries=[paper])
        original = build_structured_output(report)
        dumped = original.model_dump_json()
        restored = StructuredEvidence.model_validate_json(dumped)
        assert restored.topic == original.topic
        assert len(restored.claims) == len(original.claims)
        assert restored.claims[0].claim_id == original.claims[0].claim_id


class TestOpenQuestions:
    """Open questions are passed through."""

    def test_open_questions_preserved(self) -> None:
        result = build_structured_output(_make_report(open_questions=["Q1?", "Q2?"]))
        assert result.open_questions == ["Q1?", "Q2?"]


class TestExportJson:
    """export_structured_json writes valid JSON to disk."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        report = _make_report(paper_summaries=[_make_paper(findings=["F1"])])
        out = tmp_path / "sub" / "evidence.json"
        export_structured_json(report, out)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["topic"] == "test topic"
        assert len(data["claims"]) == 1
