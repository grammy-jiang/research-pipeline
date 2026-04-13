"""Tests for MCP prompt templates."""

from __future__ import annotations

from mcp_server import prompts
from mcp_server.server import mcp


class TestPromptRegistration:
    def test_prompt_count(self) -> None:
        """All 5 prompts should be registered."""
        registered = mcp._prompt_manager._prompts
        assert len(registered) == 5

    def test_prompt_names(self) -> None:
        registered = set(mcp._prompt_manager._prompts.keys())
        expected = {
            "research_topic",
            "analyze_paper",
            "compare_papers",
            "refine_search",
            "quality_assessment",
        }
        assert expected == registered


class TestResearchTopicPrompt:
    def test_returns_messages(self) -> None:
        msgs = prompts.research_topic_prompt("transformer attention")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "transformer attention" in msgs[1]["content"]

    def test_mentions_pipeline_stages(self) -> None:
        msgs = prompts.research_topic_prompt("test topic")
        system_msg = msgs[0]["content"]
        assert "plan" in system_msg
        assert "search" in system_msg
        assert "summarize" in system_msg


class TestAnalyzePaperPrompt:
    def test_returns_messages(self) -> None:
        msgs = prompts.analyze_paper_prompt("run-001", "2401.12345")
        assert len(msgs) == 2
        assert "2401.12345" in msgs[1]["content"]
        assert "run-001" in msgs[1]["content"]

    def test_includes_resource_uri(self) -> None:
        msgs = prompts.analyze_paper_prompt("run-001", "2401.12345")
        assert "runs://run-001/markdown/2401.12345" in msgs[1]["content"]


class TestComparePapersPrompt:
    def test_returns_messages(self) -> None:
        msgs = prompts.compare_papers_prompt("run-001")
        assert len(msgs) == 2
        assert "run-001" in msgs[1]["content"]
        assert "synthesis" in msgs[1]["content"].lower()


class TestRefineSearchPrompt:
    def test_returns_messages(self) -> None:
        msgs = prompts.refine_search_prompt("run-001")
        assert len(msgs) == 2
        assert "candidates" in msgs[1]["content"].lower()


class TestQualityAssessmentPrompt:
    def test_returns_messages(self) -> None:
        msgs = prompts.quality_assessment_prompt("run-001")
        assert len(msgs) == 2
        assert "quality" in msgs[1]["content"].lower()
