"""Tests for structured Step 1 extraction and Step 2 synthesis."""

from __future__ import annotations

import textwrap
from pathlib import Path

from research_pipeline.models.summary import (
    ConfidenceLevel,
    EvidenceSnippet,
    ExtractedStatement,
    PaperExtractionRecord,
    StatementType,
)
from research_pipeline.summarization.per_paper import (
    extract_paper,
    project_extraction_to_summary,
    render_extraction_markdown,
    score_extraction_quality,
)
from research_pipeline.summarization.synthesis import (
    project_structured_synthesis_to_report,
    render_structured_synthesis_markdown,
    synthesize_extractions,
)


def _write_markdown(tmp_path: Path) -> Path:
    path = tmp_path / "paper.md"
    path.write_text(
        textwrap.dedent(
            """\
            # Introduction

            This paper studies retrieval memory for AI agents.

            # Method

            The system uses a hybrid retrieval index with reranking.

            # Results

            The hybrid retrieval method improves answer accuracy by 12 percent.

            # Limitations

            The evaluation uses a small benchmark and does not test production load.
            """
        ),
        encoding="utf-8",
    )
    return path


def _statement(
    paper_id: str,
    category: str,
    text: str,
    evidence_id: str = "E001",
) -> ExtractedStatement:
    return ExtractedStatement(
        statement_id=f"{paper_id}:{category}:001",
        statement=text,
        category=category,
        statement_type=StatementType.AUTHOR_CLAIM,
        confidence=ConfidenceLevel.MEDIUM,
        evidence_ids=[evidence_id],
    )


def _record(paper_id: str, title: str, result: str) -> PaperExtractionRecord:
    evidence = EvidenceSnippet(
        evidence_id="E001",
        paper_id=paper_id,
        chunk_id=f"{paper_id}_chunk_001",
        line_range="L1-L4",
        quote=result,
        confidence=ConfidenceLevel.HIGH,
    )
    record = PaperExtractionRecord(
        paper_id=paper_id,
        version="v1",
        title=title,
        problem=[_statement(paper_id, "problem", "Agents need reliable memory.")],
        contributions=[
            _statement(paper_id, "contributions", "Hybrid retrieval is proposed.")
        ],
        methods=[
            _statement(
                paper_id,
                "methods",
                "The method uses a hybrid retrieval index with reranking.",
            )
        ],
        results=[_statement(paper_id, "results", result)],
        assumptions=[
            _statement(paper_id, "assumptions", "The benchmark fits in memory.")
        ],
        limitations=[
            _statement(paper_id, "limitations", "Production load is not tested.")
        ],
        reusable_mechanisms=[
            _statement(paper_id, "reusable_mechanisms", "Hybrid retrieval index")
        ],
        evidence=[evidence],
    )
    return record.model_copy(update={"quality": score_extraction_quality(record)})


def test_extract_paper_fallback_produces_valid_record(tmp_path: Path) -> None:
    markdown = _write_markdown(tmp_path)
    record = extract_paper(
        markdown,
        "2401.00001",
        "v1",
        "Hybrid Retrieval Memory",
        ["retrieval", "memory"],
    )

    assert record.paper_id == "2401.00001"
    assert record.extraction_metadata.mode == "template_fallback"
    assert record.evidence
    assert record.quality.completeness_score > 0
    assert render_extraction_markdown(record).startswith("# Paper Extraction")


def test_project_extraction_to_legacy_summary() -> None:
    record = _record(
        "2401.00001",
        "Hybrid Retrieval Memory",
        "Hybrid retrieval improves answer accuracy.",
    )
    summary = project_extraction_to_summary(record)

    assert summary.arxiv_id == record.paper_id
    assert "Hybrid retrieval" in " ".join(summary.findings)
    assert summary.evidence[0].chunk_id == "2401.00001_chunk_001"


def test_synthesize_extractions_builds_structured_sections() -> None:
    records = [
        _record(
            "2401.00001",
            "Hybrid Retrieval A",
            "Hybrid retrieval improves answer accuracy.",
        ),
        _record(
            "2401.00002",
            "Hybrid Retrieval B",
            "Hybrid retrieval improves answer accuracy on agent tasks.",
        ),
    ]

    synthesis = synthesize_extractions(records, "agent memory")

    assert synthesis.corpus
    assert synthesis.evidence_matrix
    assert synthesis.assumption_map
    assert synthesis.evidence_strength_map
    assert synthesis.traceability_appendix
    assert synthesis.quality.coverage_score == 1.0


def test_structured_synthesis_projects_to_legacy_report() -> None:
    records = [
        _record(
            "2401.00001",
            "Hybrid Retrieval A",
            "Hybrid retrieval improves answer accuracy.",
        )
    ]
    synthesis = synthesize_extractions(records, "agent memory")
    summaries = [project_extraction_to_summary(records[0])]
    legacy = project_structured_synthesis_to_report(synthesis, summaries)
    markdown = render_structured_synthesis_markdown(synthesis)

    assert legacy.paper_count == 1
    assert "Assumption Map" in markdown
    assert "Traceability Appendix" in markdown
