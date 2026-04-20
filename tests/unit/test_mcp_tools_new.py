"""Tests for the 9 new MCP tool implementations (CLI parity)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from mcp_server.schemas import (
    AnalyzeClaimsInput,
    EvaluateInput,
    KGIngestInput,
    KGQueryInput,
    KGStatsInput,
    MemoryEpisodesInput,
    MemorySearchInput,
    MemoryStatsInput,
    ScoreClaimsInput,
)
from mcp_server.tools import (
    analyze_claims_tool,
    evaluate_tool,
    kg_ingest_tool,
    kg_query_tool,
    kg_stats_tool,
    memory_episodes_tool,
    memory_search_tool,
    memory_stats_tool,
    score_claims_tool,
)

# ---------------------------------------------------------------------------
# analyze_claims_tool
# ---------------------------------------------------------------------------


class TestAnalyzeClaimsTool:
    def test_no_summaries(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-1"
        (run_dir / "summarize").mkdir(parents=True)
        result = analyze_claims_tool(
            AnalyzeClaimsInput(run_id="run-1", workspace=str(tmp_path))
        )
        assert result.success is False
        assert "No paper summaries" in result.message

    def test_decompose_papers(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-1"
        summary_dir = run_dir / "summarize"
        summary_dir.mkdir(parents=True)
        convert_dir = run_dir / "convert"
        convert_dir.mkdir(parents=True)

        summary = {
            "arxiv_id": "2401.00001",
            "version": "v1",
            "title": "Test Paper",
            "objective": "Test objective",
            "methodology": "Survey",
            "findings": ["Finding 1"],
            "limitations": [],
            "evidence": [],
            "uncertainties": [],
        }
        summary_path = summary_dir / "paper_summaries.jsonl"
        summary_path.write_text(json.dumps(summary) + "\n")

        mock_decomp = MagicMock()
        mock_decomp.total_claims = 3
        mock_decomp.evidence_summary = {"supported": 2, "unsupported": 1}
        mock_decomp.model_dump.return_value = {
            "paper_id": "2401.00001",
            "total_claims": 3,
            "evidence_summary": {"supported": 2},
        }

        with (
            patch("mcp_server.tools.decompose_paper", create=True),
            patch.dict(
                "sys.modules",
                {
                    "research_pipeline.analysis.decomposer": MagicMock(
                        decompose_paper=MagicMock(return_value=mock_decomp)
                    ),
                },
            ),
        ):
            result = analyze_claims_tool(
                AnalyzeClaimsInput(run_id="run-1", workspace=str(tmp_path))
            )

        assert result.success is True
        assert "3 claims" in result.message
        assert result.artifacts["papers"] == 1


# ---------------------------------------------------------------------------
# score_claims_tool
# ---------------------------------------------------------------------------


class TestScoreClaimsTool:
    def test_no_decompositions(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-1"
        (run_dir / "summarize" / "claims").mkdir(parents=True)
        result = score_claims_tool(
            ScoreClaimsInput(run_id="run-1", workspace=str(tmp_path))
        )
        assert result.success is False
        assert "No claim decompositions" in result.message


# ---------------------------------------------------------------------------
# kg_stats_tool
# ---------------------------------------------------------------------------


class TestKGStatsTool:
    def test_stats_returns_counts(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        mock_kg = MagicMock()
        mock_kg.stats.return_value = {
            "total_entities": 10,
            "total_triples": 25,
        }
        mock_kg.close = MagicMock()

        with patch(
            "research_pipeline.storage.knowledge_graph.KnowledgeGraph",
            return_value=mock_kg,
        ):
            result = kg_stats_tool(KGStatsInput(db_path=str(db_path)))

        assert result.success is True
        assert "10 entities" in result.message
        assert result.artifacts["total_entities"] == 10
        assert result.artifacts["total_triples"] == 25
        mock_kg.close.assert_called_once()

    def test_stats_default_path(self) -> None:
        mock_kg = MagicMock()
        mock_kg.stats.return_value = {"total_entities": 0, "total_triples": 0}
        mock_kg.close = MagicMock()

        with patch(
            "research_pipeline.storage.knowledge_graph.KnowledgeGraph",
            return_value=mock_kg,
        ):
            result = kg_stats_tool(KGStatsInput())

        assert result.success is True


# ---------------------------------------------------------------------------
# kg_query_tool
# ---------------------------------------------------------------------------


class TestKGQueryTool:
    def test_entity_not_found(self) -> None:
        mock_kg = MagicMock()
        mock_kg.get_entity.return_value = None
        mock_kg.close = MagicMock()

        with patch(
            "research_pipeline.storage.knowledge_graph.KnowledgeGraph",
            return_value=mock_kg,
        ):
            result = kg_query_tool(KGQueryInput(entity_id="nonexistent"))

        assert result.success is False
        assert "not found" in result.message

    def test_entity_found_with_relations(self) -> None:
        mock_entity = MagicMock()
        mock_entity.entity_id = "paper-1"
        mock_entity.name = "Test Paper"
        mock_entity.entity_type.value = "paper"
        mock_entity.properties = {"year": "2024"}

        mock_triple = MagicMock()
        mock_triple.subject_id = "paper-1"
        mock_triple.object_id = "author-1"
        mock_triple.relation.value = "authored_by"
        mock_triple.confidence = 0.9

        mock_kg = MagicMock()
        mock_kg.get_entity.return_value = mock_entity
        mock_kg.get_neighbors.return_value = [mock_triple]
        mock_kg.close = MagicMock()

        with patch(
            "research_pipeline.storage.knowledge_graph.KnowledgeGraph",
            return_value=mock_kg,
        ):
            result = kg_query_tool(KGQueryInput(entity_id="paper-1"))

        assert result.success is True
        assert "Test Paper" in result.message
        assert len(result.artifacts["relations"]) == 1
        assert result.artifacts["relations"][0]["direction"] == "outgoing"


# ---------------------------------------------------------------------------
# kg_ingest_tool
# ---------------------------------------------------------------------------


class TestKGIngestTool:
    def test_ingest_with_no_data(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-1"
        (run_dir / "screen").mkdir(parents=True)
        (run_dir / "summarize" / "claims").mkdir(parents=True)

        mock_kg = MagicMock()
        mock_kg.stats.return_value = {"total_entities": 0, "total_triples": 0}
        mock_kg.close = MagicMock()

        with patch(
            "research_pipeline.storage.knowledge_graph.KnowledgeGraph",
            return_value=mock_kg,
        ):
            result = kg_ingest_tool(
                KGIngestInput(run_id="run-1", workspace=str(tmp_path))
            )

        assert result.success is True
        assert result.artifacts["total_ingested"] == 0


# ---------------------------------------------------------------------------
# memory_stats_tool
# ---------------------------------------------------------------------------


class TestMemoryStatsTool:
    def test_stats(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.summary.return_value = {
            "episodic": {"total_episodes": 5},
            "knowledge_graph": {"entities": 10},
        }
        mock_mgr.close = MagicMock()

        with patch(
            "research_pipeline.memory.manager.MemoryManager",
            return_value=mock_mgr,
        ):
            result = memory_stats_tool(MemoryStatsInput())

        assert result.success is True
        assert "statistics" in result.message.lower()


# ---------------------------------------------------------------------------
# memory_episodes_tool
# ---------------------------------------------------------------------------


class TestMemoryEpisodesTool:
    def test_recent_episodes(self) -> None:
        from research_pipeline.memory.episodic import Episode

        ep = Episode(
            run_id="run-1",
            topic="transformers",
            paper_count=5,
            shortlist_count=3,
            stages_completed=["plan", "search"],
            started_at="2024-01-01T00:00:00",
        )

        mock_mem = MagicMock()
        mock_mem.recent_episodes.return_value = [ep]
        mock_mem.close = MagicMock()

        with patch(
            "research_pipeline.memory.episodic.EpisodicMemory",
            return_value=mock_mem,
        ):
            result = memory_episodes_tool(MemoryEpisodesInput(limit=5))

        assert result.success is True
        assert len(result.artifacts["episodes"]) == 1
        assert result.artifacts["episodes"][0]["topic"] == "transformers"

    def test_empty_episodes(self) -> None:
        mock_mem = MagicMock()
        mock_mem.recent_episodes.return_value = []
        mock_mem.close = MagicMock()

        with patch(
            "research_pipeline.memory.episodic.EpisodicMemory",
            return_value=mock_mem,
        ):
            result = memory_episodes_tool(MemoryEpisodesInput())

        assert result.success is True
        assert result.artifacts["episodes"] == []


# ---------------------------------------------------------------------------
# memory_search_tool
# ---------------------------------------------------------------------------


class TestMemorySearchTool:
    def test_search_finds_results(self) -> None:
        from research_pipeline.memory.episodic import Episode

        ep = Episode(
            run_id="run-2",
            topic="RAG systems",
            paper_count=8,
            shortlist_count=4,
            stages_completed=["plan", "search", "screen"],
            started_at="2024-03-15T10:00:00",
        )

        mock_mem = MagicMock()
        mock_mem.search_by_topic.return_value = [ep]
        mock_mem.close = MagicMock()

        with patch(
            "research_pipeline.memory.episodic.EpisodicMemory",
            return_value=mock_mem,
        ):
            result = memory_search_tool(MemorySearchInput(topic="RAG", limit=5))

        assert result.success is True
        assert len(result.artifacts["episodes"]) == 1
        assert result.artifacts["query"] == "RAG"

    def test_search_no_results(self) -> None:
        mock_mem = MagicMock()
        mock_mem.search_by_topic.return_value = []
        mock_mem.close = MagicMock()

        with patch(
            "research_pipeline.memory.episodic.EpisodicMemory",
            return_value=mock_mem,
        ):
            result = memory_search_tool(MemorySearchInput(topic="nonexistent"))

        assert result.success is True
        assert len(result.artifacts["episodes"]) == 0


# ---------------------------------------------------------------------------
# evaluate_tool
# ---------------------------------------------------------------------------


class TestEvaluateTool:
    def test_run_not_found(self, tmp_path: Path) -> None:
        result = evaluate_tool(
            EvaluateInput(
                run_id="no-such-run",
                workspace=str(tmp_path),
            )
        )
        assert result.success is False
        assert "not found" in result.message.lower()

    def test_evaluate_all_stages(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        mock_report = MagicMock()
        mock_report.passed = True
        mock_report.stage = "plan"
        mock_report.error_count = 0
        mock_report.warning_count = 0
        mock_check = MagicMock()
        mock_check.name = "schema_valid"
        mock_check.passed = True
        mock_check.description = "OK"
        mock_check.details = ""
        mock_check.severity = "error"
        mock_report.checks = [mock_check]

        with patch(
            "research_pipeline.evaluation.schema_eval.evaluate_run",
            return_value=[mock_report],
        ):
            result = evaluate_tool(
                EvaluateInput(
                    run_id="test-run",
                    workspace=str(tmp_path),
                )
            )

        assert result.success is True
        assert result.artifacts["verdict"] == "PASS"
        assert result.artifacts["all_passed"] is True

    def test_evaluate_single_stage(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        mock_report = MagicMock()
        mock_report.passed = False
        mock_report.stage = "search"
        mock_report.error_count = 2
        mock_report.warning_count = 1
        mock_check = MagicMock()
        mock_check.name = "file_exists"
        mock_check.passed = False
        mock_check.description = "Missing file"
        mock_check.details = "candidates.jsonl not found"
        mock_check.severity = "error"
        mock_report.checks = [mock_check]

        with patch(
            "research_pipeline.evaluation.schema_eval.evaluate_stage",
            return_value=mock_report,
        ):
            result = evaluate_tool(
                EvaluateInput(
                    run_id="test-run",
                    workspace=str(tmp_path),
                    stage="search",
                )
            )

        assert result.success is True
        assert result.artifacts["verdict"] == "FAIL"
        assert result.artifacts["all_passed"] is False
        assert result.artifacts["stages"][0]["error_count"] == 2
