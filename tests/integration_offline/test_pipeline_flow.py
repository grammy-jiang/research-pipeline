"""Integration tests for end-to-end pipeline flows (offline, no network).

Tests exercise the plan → screen → cluster → export-bibtex → report → cite-context
flow by creating mock candidate data and exercising CLI functions directly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from research_pipeline.models.candidate import CandidateRecord


def _make_candidate(idx: int, title: str, abstract: str) -> CandidateRecord:
    """Create a minimal CandidateRecord for testing."""
    return CandidateRecord(
        arxiv_id=f"2401.{10000 + idx:05d}",
        version="v1",
        title=title,
        authors=[f"Author {idx}"],
        published=datetime(2024, 1, idx + 1, tzinfo=UTC),
        updated=datetime(2024, 1, idx + 1, tzinfo=UTC),
        categories=["cs.CL"],
        primary_category="cs.CL",
        abstract=abstract,
        abs_url=f"https://arxiv.org/abs/2401.{10000 + idx:05d}",
        pdf_url=f"https://arxiv.org/pdf/2401.{10000 + idx:05d}",
        year=2024,
    )


CANDIDATES = [
    _make_candidate(
        0,
        "Attention Is All You Need",
        "We propose a new architecture based entirely on attention mechanisms.",
    ),
    _make_candidate(
        1,
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "We introduce BERT, a bidirectional transformer for language understanding.",
    ),
    _make_candidate(
        2,
        "GPT-4 Technical Report",
        "We report on GPT-4, a large multimodal model with human-level performance.",
    ),
    _make_candidate(
        3,
        "Retrieval-Augmented Generation for Knowledge Tasks",
        "RAG combines retrieval with generation for improved factual accuracy.",
    ),
    _make_candidate(
        4,
        "Scaling Laws for Neural Language Models",
        "We study empirical scaling laws for language model performance.",
    ),
]


def _write_candidates_jsonl(path: Path, candidates: list[CandidateRecord]) -> None:
    """Write candidates to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in candidates:
            f.write(c.model_dump_json() + "\n")


class TestPlanStage:
    """Test the plan stage produces valid output."""

    def test_plan_creates_query_plan(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_plan import run_plan

        run_plan(
            topic="transformer attention mechanisms",
            workspace=tmp_path,
        )

        # Should create a run directory with plan output
        runs = list(tmp_path.iterdir())
        assert len(runs) == 1
        run_dir = runs[0]

        plan_dir = run_dir / "plan"
        assert plan_dir.exists()

        plan_file = plan_dir / "query_plan.json"
        assert plan_file.exists()

        plan_data = json.loads(plan_file.read_text())
        assert "must_terms" in plan_data
        assert "query_variants" in plan_data


class TestClusterStage:
    """Test paper clustering from candidate data."""

    def test_cluster_groups_similar_papers(self, tmp_path: Path) -> None:
        from research_pipeline.screening.clustering import cluster_candidates

        clusters = cluster_candidates(CANDIDATES, threshold=0.15)

        assert isinstance(clusters, list)
        assert len(clusters) > 0

        # Each cluster has at least one paper
        all_ids = set()
        for cluster in clusters:
            assert len(cluster.paper_ids) > 0
            for pid in cluster.paper_ids:
                all_ids.add(pid)

        # All papers accounted for
        assert all_ids == {c.arxiv_id for c in CANDIDATES}


class TestExportBibtexStage:
    """Test BibTeX export from candidate data."""

    def test_bibtex_export_from_screen(self, tmp_path: Path) -> None:
        from research_pipeline.summarization.bibtex_export import (
            export_candidates_bibtex,
        )

        out_path = tmp_path / "refs.bib"
        count = export_candidates_bibtex(CANDIDATES, out_path)

        assert count == 5
        bib_str = out_path.read_text()
        assert "@article{" in bib_str or "@misc{" in bib_str
        assert "Attention Is All You Need" in bib_str
        assert "2401.10000" in bib_str

    def test_bibtex_roundtrip_via_file(self, tmp_path: Path) -> None:
        from research_pipeline.summarization.bibtex_export import (
            export_candidates_bibtex,
        )

        out_path = tmp_path / "refs.bib"
        export_candidates_bibtex(CANDIDATES, out_path)

        content = out_path.read_text()
        assert len(content) > 100
        # All 5 papers present
        for c in CANDIDATES:
            assert c.arxiv_id in content


class TestReportTemplateStage:
    """Test report template rendering."""

    def test_survey_template_renders(self, tmp_path: Path) -> None:
        from research_pipeline.models.summary import SynthesisReport
        from research_pipeline.summarization.report_templates import render_report

        report = SynthesisReport(
            topic="transformer attention",
            paper_count=5,
            agreements=[],
            disagreements=[],
            open_questions=["How do transformers scale?"],
        )

        rendered = render_report(report, template_name="survey")

        assert "transformer attention" in rendered.lower()
        assert len(rendered) > 50

    def test_all_builtin_templates_render(self, tmp_path: Path) -> None:
        from research_pipeline.models.summary import SynthesisReport
        from research_pipeline.summarization.report_templates import render_report

        report = SynthesisReport(
            topic="test topic",
            paper_count=3,
        )

        for tmpl in ["survey", "gap_analysis", "lit_review", "executive"]:
            rendered = render_report(report, template_name=tmpl)
            assert len(rendered) > 20, f"Template {tmpl} produced empty output"


class TestCitationContextStage:
    """Test citation context extraction from markdown."""

    def test_extract_numeric_citations(self) -> None:
        from research_pipeline.extraction.citation_context import (
            extract_citation_contexts,
        )

        md_text = (
            "# Introduction\n\n"
            "Transformers were introduced in [1]. They use attention mechanisms.\n"
            "BERT [2] improved upon this with bidirectional training.\n\n"
            "# Methods\n\n"
            "We build on RAG [3] for retrieval-augmented generation.\n"
        )

        contexts = extract_citation_contexts(md_text, context_window=0)

        assert len(contexts) >= 3
        # Check citation markers are detected
        markers = {c.marker for c in contexts}
        assert "[1]" in markers
        assert "[2]" in markers
        assert "[3]" in markers

    def test_extract_author_year_citations(self) -> None:
        from research_pipeline.extraction.citation_context import (
            extract_citation_contexts,
        )

        md_text = (
            "# Related Work\n\n"
            "The transformer (Vaswani, 2017) revolutionized NLP.\n"
            "Later, (Devlin, 2019) proposed BERT for pre-training.\n"
        )

        contexts = extract_citation_contexts(md_text, context_window=0)

        assert len(contexts) >= 2


class TestEndToEndPlanToCluster:
    """Test plan → (mock candidates) → cluster → export flow."""

    def test_plan_then_cluster_then_export(self, tmp_path: Path) -> None:
        """End-to-end: plan a topic, inject mock candidates, cluster, export."""
        from research_pipeline.cli.cmd_plan import run_plan
        from research_pipeline.screening.clustering import cluster_candidates
        from research_pipeline.summarization.bibtex_export import (
            export_candidates_bibtex,
        )

        # Step 1: Plan
        run_plan(topic="attention mechanisms", workspace=tmp_path)
        runs = list(tmp_path.iterdir())
        assert len(runs) == 1
        run_dir = runs[0]

        # Step 2: Inject mock candidates (simulates search + screen)
        screen_dir = run_dir / "screen"
        _write_candidates_jsonl(screen_dir / "shortlist.jsonl", CANDIDATES)

        # Step 3: Cluster
        clusters = cluster_candidates(CANDIDATES, threshold=0.15)
        cluster_out = run_dir / "clustering" / "clusters.json"
        cluster_out.parent.mkdir(parents=True, exist_ok=True)
        cluster_data = [
            {"cluster_id": cl.cluster_id, "label": cl.label, "paper_ids": cl.paper_ids}
            for cl in clusters
        ]
        cluster_out.write_text(json.dumps(cluster_data, indent=2))
        assert cluster_out.exists()

        # Step 4: BibTeX export
        bib_out = run_dir / "bibtex" / "references.bib"
        bib_out.parent.mkdir(parents=True, exist_ok=True)
        export_candidates_bibtex(CANDIDATES, bib_out)
        assert bib_out.exists()
        assert "Attention Is All You Need" in bib_out.read_text()


class TestWatchCommandParsing:
    """Test watch command query file parsing."""

    def test_watch_loads_queries_from_json(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_watch import _load_queries

        queries_file = tmp_path / "queries.json"
        queries_file.write_text(
            json.dumps(
                [
                    {"query": "transformer attention", "categories": ["cs.CL"]},
                    {"query": "retrieval augmented generation"},
                ]
            )
        )

        queries = _load_queries(queries_file)
        assert len(queries) == 2
        assert queries[0]["query"] == "transformer attention"
