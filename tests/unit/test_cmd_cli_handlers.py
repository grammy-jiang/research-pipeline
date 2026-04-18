"""Tests for CLI handler modules — exercises command logic via mocked dependencies.

Targets the 0%-coverage CLI handlers to boost overall project coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import click.exceptions
import pytest


# ---------------------------------------------------------------------------
# cmd_coherence
# ---------------------------------------------------------------------------
class TestCmdCoherence:
    """Tests for the coherence CLI handler."""

    def test_requires_at_least_two_run_ids(self) -> None:
        from research_pipeline.cli.cmd_coherence import run_coherence_cmd

        with pytest.raises(click.exceptions.Exit):
            run_coherence_cmd(run_ids=["only-one"])

    @patch("research_pipeline.cli.cmd_coherence.run_coherence")
    @patch("research_pipeline.cli.cmd_coherence.load_config")
    def test_calls_run_coherence_with_correct_args(
        self, mock_config: MagicMock, mock_coherence: MagicMock, tmp_path: Path
    ) -> None:
        from research_pipeline.cli.cmd_coherence import run_coherence_cmd

        mock_config.return_value = MagicMock(workspace=str(tmp_path))
        score = MagicMock(
            overall=0.9,
            factual_consistency=0.8,
            temporal_ordering=0.85,
            knowledge_update_fidelity=0.9,
            contradiction_rate=0.05,
        )
        mock_coherence.return_value = MagicMock(
            score=score,
            topic_overlap=0.7,
            finding_count=10,
            common_finding_count=5,
            contradictions=[],
            knowledge_updates=[],
        )
        run_coherence_cmd(run_ids=["r1", "r2"], workspace=tmp_path)
        mock_coherence.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_adaptive_stopping
# ---------------------------------------------------------------------------
class TestCmdAdaptiveStopping:
    """Tests for the adaptive-stopping CLI handler."""

    def test_basic_invocation(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_adaptive_stopping import (
            adaptive_stopping_command,
        )

        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps([[0.8, 0.6], [0.3, 0.2]]))

        with patch(
            "research_pipeline.screening.adaptive_stopping.evaluate_stopping"
        ) as mock_eval:
            decision = MagicMock(
                should_stop=True,
                reason=MagicMock(value="converged"),
                details="scores converged",
                batches_processed=2,
                total_results=4,
                current_score=0.47,
            )
            mock_eval.return_value = decision

            adaptive_stopping_command(
                scores_file=scores_file,
                query="test query",
                query_type="recall",
                min_results=5,
                max_budget=500,
                relevance_threshold=0.5,
                output=None,
            )
            mock_eval.assert_called_once()

    def test_invalid_scores_file(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_adaptive_stopping import (
            adaptive_stopping_command,
        )

        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps({"not": "a list"}))

        with pytest.raises(click.exceptions.Exit):
            adaptive_stopping_command(
                scores_file=scores_file,
                query="",
                query_type="auto",
                min_results=5,
                max_budget=500,
                relevance_threshold=0.5,
                output=None,
            )

    def test_invalid_query_type(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_adaptive_stopping import (
            adaptive_stopping_command,
        )

        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps([[0.5]]))

        with pytest.raises(click.exceptions.Exit):
            adaptive_stopping_command(
                scores_file=scores_file,
                query="",
                query_type="bogus",
                min_results=5,
                max_budget=500,
                relevance_threshold=0.5,
                output=None,
            )

    def test_output_to_file(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_adaptive_stopping import (
            adaptive_stopping_command,
        )

        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps([[0.8, 0.6]]))
        out_file = tmp_path / "result.json"

        with patch(
            "research_pipeline.screening.adaptive_stopping.evaluate_stopping"
        ) as mock_eval:
            decision = MagicMock(
                should_stop=False,
                reason=MagicMock(value="continuing"),
                details="not enough data",
                batches_processed=1,
                total_results=2,
                current_score=0.7,
            )
            mock_eval.return_value = decision

            adaptive_stopping_command(
                scores_file=scores_file,
                query="test",
                query_type="precision",
                min_results=5,
                max_budget=500,
                relevance_threshold=0.5,
                output=out_file,
            )
            assert out_file.exists()
        mock_eval.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_aggregate
# ---------------------------------------------------------------------------
class TestCmdAggregate:
    """Tests for the aggregate CLI handler."""

    @patch("research_pipeline.cli.cmd_aggregate.aggregate_evidence")
    @patch("research_pipeline.cli.cmd_aggregate.load_config")
    @patch("research_pipeline.cli.cmd_aggregate.get_stage_dir")
    def test_missing_synthesis_report_exits(
        self,
        mock_stage_dir: MagicMock,
        mock_config: MagicMock,
        mock_agg: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_aggregate import aggregate_cmd

        mock_config.return_value = MagicMock(workspace=str(tmp_path))
        mock_stage_dir.return_value = tmp_path / "summarize"
        (tmp_path / "summarize").mkdir()
        # No synthesis.json → should exit
        with pytest.raises(click.exceptions.Exit):
            aggregate_cmd(run_id="test-run", config_path=str(tmp_path / "c.toml"))


# ---------------------------------------------------------------------------
# cmd_cbr
# ---------------------------------------------------------------------------
class TestCmdCbr:
    """Tests for the CBR CLI handlers."""

    @patch("research_pipeline.cli.cmd_cbr.cbr_lookup")
    def test_lookup_json_output(self, mock_lookup: MagicMock, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_cbr import handle_cbr_lookup

        rec = MagicMock()
        rec.to_dict.return_value = {"confidence": 0.8, "sources": ["arxiv"]}
        mock_lookup.return_value = rec

        handle_cbr_lookup(workspace=tmp_path, topic="AI memory", output_json=True)
        mock_lookup.assert_called_once()

    @patch("research_pipeline.cli.cmd_cbr.cbr_lookup")
    def test_lookup_text_output(self, mock_lookup: MagicMock, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_cbr import handle_cbr_lookup

        rec = MagicMock(
            confidence=0.8,
            recommended_sources=["arxiv"],
            recommended_profile="standard",
            recommended_query_terms=["memory", "llm"],
            basis_cases=["case1"],
            reasoning="good match",
        )
        mock_lookup.return_value = rec

        handle_cbr_lookup(workspace=tmp_path, topic="AI memory")
        mock_lookup.assert_called_once()

    @patch("research_pipeline.cli.cmd_cbr.cbr_retain")
    def test_retain_json_output(self, mock_retain: MagicMock, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_cbr import handle_cbr_retain

        case = MagicMock()
        case.to_dict.return_value = {"case_id": "c1"}
        mock_retain.return_value = case

        handle_cbr_retain(
            workspace=tmp_path,
            run_id="r1",
            topic="AI",
            output_json=True,
        )
        mock_retain.assert_called_once()

    @patch("research_pipeline.cli.cmd_cbr.cbr_retain")
    def test_retain_text_output(self, mock_retain: MagicMock, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_cbr import handle_cbr_retain

        case = MagicMock(
            case_id="c1",
            topic="AI",
            sources_used=["arxiv"],
            paper_count=10,
            shortlist_count=5,
            synthesis_quality=0.8,
            outcome="good",
        )
        mock_retain.return_value = case

        handle_cbr_retain(workspace=tmp_path, run_id="r1", topic="AI")
        mock_retain.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_memory
# ---------------------------------------------------------------------------
class TestCmdMemory:
    """Tests for the memory CLI handlers."""

    @patch("research_pipeline.memory.manager.MemoryManager")
    def test_stats_calls_summary(self, mock_mgr_cls: MagicMock) -> None:
        from research_pipeline.cli.cmd_memory import run_memory_stats

        mgr = MagicMock()
        mgr.summary.return_value = {"episodes": 5}
        mock_mgr_cls.return_value = mgr

        run_memory_stats()
        mgr.summary.assert_called_once()
        mgr.close.assert_called_once()

    @patch("research_pipeline.memory.episodic.EpisodicMemory")
    def test_episodes_empty(self, mock_ep_cls: MagicMock) -> None:
        from research_pipeline.cli.cmd_memory import run_memory_episodes

        mem = MagicMock()
        mem.recent_episodes.return_value = []
        mock_ep_cls.return_value = mem

        run_memory_episodes()
        mem.close.assert_called_once()

    @patch("research_pipeline.memory.episodic.EpisodicMemory")
    def test_episodes_with_data(self, mock_ep_cls: MagicMock) -> None:
        from research_pipeline.cli.cmd_memory import run_memory_episodes

        ep = MagicMock(
            run_id="r1",
            topic="AI",
            paper_count=10,
            shortlist_count=3,
            stages_completed=["plan", "search"],
            started_at="2024-01-01",
        )
        mem = MagicMock()
        mem.recent_episodes.return_value = [ep]
        mock_ep_cls.return_value = mem

        run_memory_episodes()
        mem.close.assert_called_once()

    @patch("research_pipeline.memory.episodic.EpisodicMemory")
    def test_search_empty(self, mock_ep_cls: MagicMock) -> None:
        from research_pipeline.cli.cmd_memory import run_memory_search

        mem = MagicMock()
        mem.search_by_topic.return_value = []
        mock_ep_cls.return_value = mem

        run_memory_search(topic="AI")
        mem.close.assert_called_once()

    @patch("research_pipeline.memory.episodic.EpisodicMemory")
    def test_search_with_results(self, mock_ep_cls: MagicMock) -> None:
        from research_pipeline.cli.cmd_memory import run_memory_search

        ep = MagicMock(
            run_id="r1",
            topic="AI",
            paper_count=10,
            shortlist_count=3,
            started_at="2024-01-01",
        )
        mem = MagicMock()
        mem.search_by_topic.return_value = [ep]
        mock_ep_cls.return_value = mem

        run_memory_search(topic="AI")
        mem.close.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_index
# ---------------------------------------------------------------------------
class TestCmdIndex:
    """Tests for the index CLI handler."""

    @patch("research_pipeline.cli.cmd_index.GlobalPaperIndex")
    def test_gc_mode(self, mock_idx_cls: MagicMock) -> None:
        from research_pipeline.cli.cmd_index import run_index

        idx = MagicMock()
        idx.garbage_collect.return_value = 5
        mock_idx_cls.return_value = idx

        run_index(gc=True)
        idx.garbage_collect.assert_called_once()
        idx.close.assert_called_once()

    @patch("research_pipeline.cli.cmd_index.GlobalPaperIndex")
    def test_search_mode_no_results(self, mock_idx_cls: MagicMock) -> None:
        from research_pipeline.cli.cmd_index import run_index

        idx = MagicMock()
        idx.search_fulltext.return_value = []
        mock_idx_cls.return_value = idx

        run_index(search="AI")
        idx.search_fulltext.assert_called_once_with("AI", limit=50)
        idx.close.assert_called_once()

    @patch("research_pipeline.cli.cmd_index.GlobalPaperIndex")
    def test_search_mode_with_results(self, mock_idx_cls: MagicMock) -> None:
        from research_pipeline.cli.cmd_index import run_index

        idx = MagicMock()
        idx.search_fulltext.return_value = [
            {"arxiv_id": "2401.00001", "title": "Test Paper", "run_id": "r1"}
        ]
        mock_idx_cls.return_value = idx

        run_index(search="AI")
        idx.close.assert_called_once()

    @patch("research_pipeline.cli.cmd_index.GlobalPaperIndex")
    def test_list_mode_empty(self, mock_idx_cls: MagicMock) -> None:
        from research_pipeline.cli.cmd_index import run_index

        idx = MagicMock()
        idx.list_papers.return_value = []
        mock_idx_cls.return_value = idx

        run_index(list_papers=True)
        idx.list_papers.assert_called_once()
        idx.close.assert_called_once()

    @patch("research_pipeline.cli.cmd_index.GlobalPaperIndex")
    def test_list_mode_with_papers(self, mock_idx_cls: MagicMock) -> None:
        from research_pipeline.cli.cmd_index import run_index

        idx = MagicMock()
        idx.list_papers.return_value = [
            {
                "arxiv_id": "2401.00001",
                "stage": "search",
                "run_id": "r1",
                "indexed_at": "2024-01-01",
            }
        ]
        mock_idx_cls.return_value = idx

        run_index(list_papers=True)
        idx.close.assert_called_once()

    @patch("research_pipeline.cli.cmd_index.GlobalPaperIndex")
    def test_default_mode_shows_usage(self, mock_idx_cls: MagicMock) -> None:
        from research_pipeline.cli.cmd_index import run_index

        idx = MagicMock()
        mock_idx_cls.return_value = idx

        run_index()
        idx.close.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_inspect
# ---------------------------------------------------------------------------
class TestCmdInspect:
    """Tests for the inspect CLI handler."""

    def test_inspect_nonexistent_run(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_inspect import run_inspect

        with pytest.raises(click.exceptions.Exit):
            run_inspect(workspace=tmp_path, run_id="nonexistent")

    @patch("research_pipeline.cli.cmd_inspect.load_manifest")
    def test_inspect_no_manifest(
        self, mock_manifest: MagicMock, tmp_path: Path
    ) -> None:
        from research_pipeline.cli.cmd_inspect import run_inspect

        run_dir = tmp_path / "r1"
        run_dir.mkdir()
        mock_manifest.return_value = None

        with pytest.raises(click.exceptions.Exit):
            run_inspect(workspace=tmp_path, run_id="r1")

    @patch("research_pipeline.cli.cmd_inspect.load_manifest")
    def test_inspect_with_manifest(
        self, mock_manifest: MagicMock, tmp_path: Path
    ) -> None:
        from research_pipeline.cli.cmd_inspect import run_inspect

        run_dir = tmp_path / "r1"
        run_dir.mkdir()

        stage_record = MagicMock(status="completed", duration_ms=100, errors=[])
        manifest = MagicMock(
            run_id="r1",
            created_at="2024-01-01",
            package_version="0.14.1",
            topic_input="test topic",
            stages={"plan": stage_record},
            artifacts=["a1"],
            llm_calls=[],
        )
        mock_manifest.return_value = manifest

        run_inspect(workspace=tmp_path, run_id="r1")

    def test_inspect_list_runs_no_workspace(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_inspect import run_inspect

        with pytest.raises(click.exceptions.Exit):
            run_inspect(workspace=tmp_path / "nonexistent")

    @patch("research_pipeline.cli.cmd_inspect.load_manifest")
    def test_inspect_list_runs_empty(
        self, mock_manifest: MagicMock, tmp_path: Path
    ) -> None:
        from research_pipeline.cli.cmd_inspect import run_inspect

        run_inspect(workspace=tmp_path)

    @patch("research_pipeline.cli.cmd_inspect.load_manifest")
    def test_inspect_list_runs_with_runs(
        self, mock_manifest: MagicMock, tmp_path: Path
    ) -> None:
        from research_pipeline.cli.cmd_inspect import run_inspect

        (tmp_path / "r1").mkdir()
        manifest = MagicMock(
            topic_input="some topic that is long enough",
            stages={"plan": MagicMock(status="completed")},
        )
        mock_manifest.return_value = manifest

        run_inspect(workspace=tmp_path)


# ---------------------------------------------------------------------------
# cmd_dual_metrics
# ---------------------------------------------------------------------------
class TestCmdDualMetrics:
    """Tests for the dual-metrics CLI handler."""

    @patch("research_pipeline.cli.cmd_dual_metrics.aggregate_metrics")
    @patch("research_pipeline.cli.cmd_dual_metrics.evaluate_runs")
    def test_no_runs_warns(
        self,
        mock_eval: MagicMock,
        mock_agg: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_dual_metrics import handle_dual_metrics

        handle_dual_metrics(workspace=tmp_path, query="test", run_ids=[])

    @patch("research_pipeline.cli.cmd_dual_metrics.aggregate_metrics")
    @patch("research_pipeline.cli.cmd_dual_metrics.evaluate_runs")
    def test_json_output(
        self,
        mock_eval: MagicMock,
        mock_agg: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_dual_metrics import handle_dual_metrics

        result = MagicMock()
        result.to_dict.return_value = {"pass_at_k": 0.9}
        mock_eval.return_value = result

        handle_dual_metrics(
            workspace=tmp_path,
            query="test",
            run_ids=["r1"],
            output_json=True,
        )
        mock_eval.assert_called_once()

    @patch("research_pipeline.cli.cmd_dual_metrics.aggregate_metrics")
    @patch("research_pipeline.cli.cmd_dual_metrics.evaluate_runs")
    def test_text_output(
        self,
        mock_eval: MagicMock,
        mock_agg: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_dual_metrics import handle_dual_metrics

        sample = MagicMock(
            run_id="r1",
            correct=True,
            quality_score=0.9,
            fabrication_detected=False,
        )
        result = MagicMock(
            query="test",
            n=1,
            c=1,
            k=5,
            pass_at_k=0.9,
            pass_bracket_k=0.8,
            safety_gate=1.0,
            gated_pass_at_k=0.9,
            gated_pass_bracket_k=0.8,
            fabrication_count=0,
            samples=[sample],
        )
        mock_eval.return_value = result
        mock_agg.return_value = MagicMock(total_queries=1, reliability_gap=0.1)

        handle_dual_metrics(workspace=tmp_path, query="test", run_ids=["r1"])


# ---------------------------------------------------------------------------
# cmd_blinding_audit (additional coverage)
# ---------------------------------------------------------------------------
class TestCmdBlindingAuditExtended:
    """Extended tests for the blinding-audit CLI handler."""

    @patch("research_pipeline.cli.cmd_blinding_audit.run_blinding_audit_for_workspace")
    def test_text_output_with_scores(
        self, mock_audit: MagicMock, tmp_path: Path
    ) -> None:
        from research_pipeline.cli.cmd_blinding_audit import handle_blinding_audit

        paper_score = MagicMock(
            paper_id="paper123456789",
            overall_score=0.9,
            identity_references=2,
            contaminated_claims=1,
            total_claims=5,
        )
        result = MagicMock(
            run_id="r1",
            timestamp="2024-01-01",
            paper_scores=[paper_score],
            aggregate_score=0.5,
            high_contamination_papers=["paper123456789"],
            recommendation="review",
        )
        mock_audit.return_value = result

        handle_blinding_audit(workspace=tmp_path)
        mock_audit.assert_called_once()

    @patch("research_pipeline.cli.cmd_blinding_audit.run_blinding_audit_for_workspace")
    def test_json_output(self, mock_audit: MagicMock, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_blinding_audit import handle_blinding_audit

        result = MagicMock()
        result.to_dict.return_value = {"aggregate_score": 0.5}
        mock_audit.return_value = result

        handle_blinding_audit(workspace=tmp_path, output_json=True)


# ---------------------------------------------------------------------------
# cmd_export_html
# ---------------------------------------------------------------------------
class TestCmdExportHtml:
    """Tests for the export-html CLI handler."""

    def test_no_run_id_or_markdown_exits(self) -> None:
        from research_pipeline.cli.cmd_export_html import export_html_cmd

        with pytest.raises(click.exceptions.Exit):
            export_html_cmd(
                run_id="",
                markdown_file="",
                output="",
                title="Test",
                config_path="config.toml",
            )

    @patch("research_pipeline.cli.cmd_export_html.render_html_from_markdown")
    def test_markdown_mode(self, mock_render: MagicMock, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_export_html import export_html_cmd

        md_file = tmp_path / "report.md"
        md_file.write_text("# Report\nSome content")
        mock_render.return_value = "<html>test</html>"

        export_html_cmd(
            run_id="",
            markdown_file=str(md_file),
            output="",
            title="Test",
            config_path="config.toml",
        )
        mock_render.assert_called_once()

    def test_markdown_mode_missing_file(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_export_html import export_html_cmd

        with pytest.raises(click.exceptions.Exit):
            export_html_cmd(
                run_id="",
                markdown_file=str(tmp_path / "nonexistent.md"),
                output="",
                title="Test",
                config_path="config.toml",
            )


# ---------------------------------------------------------------------------
# cmd_report
# ---------------------------------------------------------------------------
class TestCmdReport:
    """Tests for the report CLI handler."""

    @patch("research_pipeline.cli.cmd_report.list_templates")
    def test_unknown_template_exits(self, mock_list: MagicMock) -> None:
        from research_pipeline.cli.cmd_report import report_cmd

        mock_list.return_value = ["survey", "gap_analysis"]
        with pytest.raises(click.exceptions.Exit):
            report_cmd(run_id="r1", template="nonexistent")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------
class TestMainModule:
    """Tests for __main__.py."""

    def test_main_has_app_reference(self) -> None:
        """Verify __main__ module source references the app."""
        from pathlib import Path

        main_src = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "research_pipeline"
            / "__main__.py"
        )
        content = main_src.read_text()
        assert "from research_pipeline.cli.app import app" in content
        assert "app()" in content


# ---------------------------------------------------------------------------
# cmd_export_bibtex
# ---------------------------------------------------------------------------


class TestCmdExportBibtex:
    """Tests for the export-bibtex CLI handler."""

    @patch("research_pipeline.cli.cmd_export_bibtex.export_candidates_bibtex")
    @patch("research_pipeline.cli.cmd_export_bibtex.load_candidates_from_jsonl")
    @patch("research_pipeline.cli.cmd_export_bibtex.init_run")
    @patch("research_pipeline.cli.cmd_export_bibtex.load_config")
    def test_export_bibtex_happy_path(
        self,
        mock_cfg: MagicMock,
        mock_init: MagicMock,
        mock_load_cands: MagicMock,
        mock_export: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_export_bibtex import export_bibtex_cmd

        stage_dir = tmp_path / "screen"
        stage_dir.mkdir()
        jsonl = stage_dir / "candidates.jsonl"
        jsonl.write_text('{"title":"p1"}\n')
        mock_cfg.return_value = MagicMock(workspace=str(tmp_path))
        mock_init.return_value = ("run1", tmp_path)
        mock_load_cands.return_value = [{"title": "p1"}]
        mock_export.return_value = 1

        with patch(
            "research_pipeline.cli.cmd_export_bibtex.get_stage_dir",
            return_value=stage_dir,
        ):
            export_bibtex_cmd(run_id="run1", stage="screen", output="")

        mock_export.assert_called_once()

    @patch("research_pipeline.cli.cmd_export_bibtex.init_run")
    @patch("research_pipeline.cli.cmd_export_bibtex.load_config")
    def test_export_bibtex_no_jsonl_exits(
        self,
        mock_cfg: MagicMock,
        mock_init: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_export_bibtex import export_bibtex_cmd

        stage_dir = tmp_path / "screen"
        stage_dir.mkdir()
        mock_cfg.return_value = MagicMock(workspace=str(tmp_path))
        mock_init.return_value = ("run1", tmp_path)

        with (
            patch(
                "research_pipeline.cli.cmd_export_bibtex.get_stage_dir",
                return_value=stage_dir,
            ),
            pytest.raises(click.exceptions.Exit),
        ):
            export_bibtex_cmd(run_id="run1", stage="screen", output="")


# ---------------------------------------------------------------------------
# cmd_cluster
# ---------------------------------------------------------------------------


class TestCmdCluster:
    """Tests for the cluster CLI handler."""

    @patch("research_pipeline.cli.cmd_cluster.cluster_candidates")
    @patch("research_pipeline.cli.cmd_cluster.load_candidates_from_jsonl")
    @patch("research_pipeline.cli.cmd_cluster.init_run")
    @patch("research_pipeline.cli.cmd_cluster.load_config")
    def test_cluster_happy_path(
        self,
        mock_cfg: MagicMock,
        mock_init: MagicMock,
        mock_load_cands: MagicMock,
        mock_cluster: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_cluster import cluster_cmd

        stage_dir = tmp_path / "screen"
        stage_dir.mkdir()
        jsonl = stage_dir / "candidates.jsonl"
        jsonl.write_text('{"title":"p1"}\n')
        mock_cfg.return_value = MagicMock(workspace=str(tmp_path))
        mock_init.return_value = ("run1", tmp_path)
        mock_load_cands.return_value = [{"title": "p1"}]
        c = MagicMock(cluster_id=0, label="AI", paper_ids=["p1"], top_terms=["ai"])
        mock_cluster.return_value = [c]

        with patch(
            "research_pipeline.cli.cmd_cluster.get_stage_dir",
            return_value=stage_dir,
        ):
            cluster_cmd(run_id="run1", stage="screen", threshold=0.15, output="")

        mock_cluster.assert_called_once()
        out_file = stage_dir / "clusters.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["num_clusters"] == 1

    @patch("research_pipeline.cli.cmd_cluster.init_run")
    @patch("research_pipeline.cli.cmd_cluster.load_config")
    def test_cluster_no_jsonl_exits(
        self,
        mock_cfg: MagicMock,
        mock_init: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_cluster import cluster_cmd

        stage_dir = tmp_path / "screen"
        stage_dir.mkdir()
        mock_cfg.return_value = MagicMock(workspace=str(tmp_path))
        mock_init.return_value = ("run1", tmp_path)

        with (
            patch(
                "research_pipeline.cli.cmd_cluster.get_stage_dir",
                return_value=stage_dir,
            ),
            pytest.raises(click.exceptions.Exit),
        ):
            cluster_cmd(run_id="run1", stage="screen", threshold=0.15, output="")
