"""Tests for app.py CLI wrapper commands — covers untested thin wrappers.

Each app.py command is a thin wrapper that imports a handler,
calls setup_logging, and delegates. We mock the handler to verify wiring.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# coherence
# ---------------------------------------------------------------------------
class TestCoherenceCommand:
    @patch("research_pipeline.cli.cmd_coherence.run_coherence_cmd")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_coherence_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import coherence

        coherence(
            run_ids=["r1", "r2"],
            verbose=False,
            config=None,
            workspace=None,
            output=None,
        )
        mock_run.assert_called_once()
        kw = mock_run.call_args
        assert kw[1]["run_ids"] == ["r1", "r2"] or kw[0][0] == ["r1", "r2"]


# ---------------------------------------------------------------------------
# consolidate
# ---------------------------------------------------------------------------
class TestConsolidateCommand:
    @patch("research_pipeline.cli.cmd_consolidate.run_consolidate_cmd")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_consolidate_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import consolidate

        consolidate(
            run_ids=[],
            verbose=False,
            config=None,
            workspace=None,
            output=None,
            dry_run=False,
            capacity=100,
            threshold=0.5,
            min_support=2,
            staleness_days=90,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# analyze-claims
# ---------------------------------------------------------------------------
class TestAnalyzeClaimsCommand:
    @patch("research_pipeline.cli.cmd_analyze_claims.run_analyze_claims")
    @patch("research_pipeline.infra.logging.setup_logging")
    @patch("research_pipeline.config.loader.load_config")
    def test_analyze_claims_delegates(
        self, mock_cfg: MagicMock, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import analyze_claims

        analyze_claims(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-run",
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# score-claims
# ---------------------------------------------------------------------------
class TestScoreClaimsCommand:
    @patch("research_pipeline.cli.cmd_score_claims.run_score_claims")
    @patch("research_pipeline.infra.logging.setup_logging")
    @patch("research_pipeline.config.loader.load_config")
    def test_score_claims_delegates(
        self, mock_cfg: MagicMock, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import score_claims

        score_claims(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-run",
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# kg-stats
# ---------------------------------------------------------------------------
class TestKgStatsCommand:
    @patch("research_pipeline.cli.cmd_kg.run_kg_stats")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_kg_stats_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import kg_stats

        kg_stats(db_path=None, verbose=False)
        mock_run.assert_called_once_with(db_path=None)


# ---------------------------------------------------------------------------
# kg-query
# ---------------------------------------------------------------------------
class TestKgQueryCommand:
    @patch("research_pipeline.cli.cmd_kg.run_kg_query")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_kg_query_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import kg_query

        kg_query(entity_id="2401.12345", db_path=None, verbose=False)
        mock_run.assert_called_once_with(
            entity_id="2401.12345", db_path=None
        )


# ---------------------------------------------------------------------------
# kg-ingest
# ---------------------------------------------------------------------------
class TestKgIngestCommand:
    @patch("research_pipeline.cli.cmd_kg.run_kg_ingest")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_kg_ingest_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import kg_ingest

        kg_ingest(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-run",
            db_path=None,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# memory-stats
# ---------------------------------------------------------------------------
class TestMemoryStatsCommand:
    @patch("research_pipeline.cli.cmd_memory.run_memory_stats")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_memory_stats_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import memory_stats

        memory_stats(verbose=False, episodic_db=None, kg_db=None)
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# memory-episodes
# ---------------------------------------------------------------------------
class TestMemoryEpisodesCommand:
    @patch("research_pipeline.cli.cmd_memory.run_memory_episodes")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_memory_episodes_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import memory_episodes

        memory_episodes(limit=10, verbose=False, episodic_db=None)
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# memory-search
# ---------------------------------------------------------------------------
class TestMemorySearchCommand:
    @patch("research_pipeline.cli.cmd_memory.run_memory_search")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_memory_search_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import memory_search

        memory_search(
            topic="transformers", limit=5, verbose=False, episodic_db=None
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
class TestEvaluateCommand:
    @patch("research_pipeline.cli.cmd_evaluate.evaluate_cmd")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_evaluate_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import evaluate

        evaluate(
            run_id="test-run",
            stage="",
            workspace="runs",
            verbose=False,
        )
        mock_run.assert_called_once_with(
            run_id="test-run", stage="", workspace="runs"
        )


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------
class TestFeedbackCommand:
    @patch("research_pipeline.cli.cmd_feedback.feedback_cmd")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_feedback_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import feedback

        feedback(
            verbose=False,
            workspace=None,
            run_id="test-run",
            accept=["p1"],
            reject=["p2"],
            reason="test",
            show=False,
            adjust=False,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# eval-log
# ---------------------------------------------------------------------------
class TestEvalLogCommand:
    @patch("research_pipeline.cli.cmd_eval_log.eval_log_cmd")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_eval_log_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import eval_log

        eval_log(
            verbose=False,
            workspace=None,
            run_id="test-run",
            channel="all",
            stage="",
            limit=50,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
class TestAggregateCommand:
    @patch("research_pipeline.cli.cmd_aggregate.aggregate_cmd")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_aggregate_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import aggregate_command

        aggregate_command(
            run_id="test-run",
            min_pointers=0,
            max_words=50,
            similarity_threshold=0.7,
            no_strip_rhetoric=False,
            output_format="text",
            config_path="config.toml",
            verbose=False,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# export-html
# ---------------------------------------------------------------------------
class TestExportHtmlCommand:
    @patch("research_pipeline.cli.cmd_export_html.export_html_cmd")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_export_html_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import export_html_command

        export_html_command(
            run_id="test-run",
            markdown_file="",
            output="",
            title="Research Report",
            config_path="config.toml",
            verbose=False,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# export-bibtex
# ---------------------------------------------------------------------------
class TestExportBibtexCommand:
    @patch("research_pipeline.cli.cmd_export_bibtex.export_bibtex_cmd")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_export_bibtex_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import export_bibtex_command

        export_bibtex_command(
            run_id="test-run",
            stage="screen",
            output="",
            verbose=False,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
class TestReportCommand:
    @patch("research_pipeline.cli.cmd_report.report_cmd")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_report_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import report_command

        report_command(
            run_id="test-run",
            template="survey",
            custom_template="",
            output="",
            verbose=False,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------
class TestClusterCommand:
    @patch("research_pipeline.cli.cmd_cluster.cluster_cmd")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_cluster_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import cluster_command

        cluster_command(
            run_id="test-run",
            stage="screen",
            threshold=0.15,
            output="",
            verbose=False,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# enrich
# ---------------------------------------------------------------------------
class TestEnrichCommand:
    @patch("research_pipeline.cli.cmd_enrich.enrich_command")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_enrich_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import enrich_command

        enrich_command(
            run_id="test-run",
            stage="candidates",
            config_path="config.toml",
            level="INFO",
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# cite-context
# ---------------------------------------------------------------------------
class TestCiteContextCommand:
    @patch("research_pipeline.cli.cmd_cite_context.cite_context_command")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_cite_context_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import cite_context_cmd

        cite_context_cmd(
            run_id="test-run",
            context_window=1,
            output="",
            config_path="config.toml",
            level="INFO",
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------
class TestWatchCommand:
    @patch(
        "research_pipeline.cli.cmd_watch.watch_command"
    )
    @patch(
        "research_pipeline.cli.cmd_watch.DEFAULT_QUERIES_FILE",
        Path("/tmp/queries.json"),
    )
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_watch_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import watch_cmd

        watch_cmd(
            queries_file="",
            lookback_days=7,
            max_results=20,
            output="",
            config_path="config.toml",
            level="INFO",
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# blinding-audit
# ---------------------------------------------------------------------------
class TestBlindingAuditCommand:
    @patch("research_pipeline.cli.cmd_blinding_audit.handle_blinding_audit")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_blinding_audit_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import blinding_audit_command

        blinding_audit_command(
            workspace="workspace",
            run_id="test-run",
            threshold=0.4,
            no_store=False,
            output_json=False,
            verbose=False,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# dual-metrics
# ---------------------------------------------------------------------------
class TestDualMetricsCommand:
    @patch("research_pipeline.cli.cmd_dual_metrics.handle_dual_metrics")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_dual_metrics_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import dual_metrics_command

        dual_metrics_command(
            workspace="workspace",
            query="transformers",
            run_ids="r1,r2",
            k=5,
            no_store=False,
            output_json=False,
            verbose=False,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# cbr-lookup
# ---------------------------------------------------------------------------
class TestCbrLookupCommand:
    @patch("research_pipeline.cli.cmd_cbr.handle_cbr_lookup")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_cbr_lookup_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import cbr_lookup_command

        cbr_lookup_command(
            workspace="workspace",
            topic="transformers",
            max_results=5,
            min_quality=0.0,
            output_json=False,
            verbose=False,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# cbr-retain
# ---------------------------------------------------------------------------
class TestCbrRetainCommand:
    @patch("research_pipeline.cli.cmd_cbr.handle_cbr_retain")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_cbr_retain_delegates(
        self, mock_log: MagicMock, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import cbr_retain_command

        cbr_retain_command(
            workspace="workspace",
            run_id="test-run",
            topic="transformers",
            outcome="good",
            strategy_notes="test",
            output_json=False,
            verbose=False,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# kg-quality
# ---------------------------------------------------------------------------
class TestKgQualityCommand:
    @patch("research_pipeline.cli.cmd_kg_quality.kg_quality_command")
    def test_kg_quality_delegates(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import kg_quality_command

        kg_quality_command(
            db_path="",
            staleness_days=365.0,
            sample_size=0,
            output_json=False,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# adaptive-stopping
# ---------------------------------------------------------------------------
class TestAdaptiveStoppingCommand:
    @patch(
        "research_pipeline.cli.cmd_adaptive_stopping.adaptive_stopping_command"
    )
    def test_adaptive_stopping_delegates(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        from research_pipeline.cli.app import adaptive_stopping_cli

        scores_file = tmp_path / "scores.json"
        scores_file.write_text("[[0.5, 0.6]]")

        adaptive_stopping_cli(
            scores_file=scores_file,
            query="test",
            query_type="auto",
            min_results=5,
            max_budget=500,
            relevance_threshold=0.5,
            output=None,
        )
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# confidence-layers
# ---------------------------------------------------------------------------
class TestConfidenceLayersCommand:
    @patch(
        "research_pipeline.cli.cmd_confidence_layers.run_confidence_layers"
    )
    def test_confidence_layers_delegates(
        self, mock_run: MagicMock
    ) -> None:
        from research_pipeline.cli.app import confidence_layers_cli

        confidence_layers_cli(
            config=None,
            workspace=None,
            run_id=None,
            l4_threshold=0.50,
            damping=0.80,
            calibrate=False,
        )
        mock_run.assert_called_once()
