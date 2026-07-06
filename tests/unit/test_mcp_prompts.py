"""Tests for MCP prompt templates."""

from __future__ import annotations

from research_pipeline.mcp_server import prompts
from research_pipeline.mcp_server.server import mcp

# Every builder in prompts.py, invoked with representative arguments. Used by
# the role-validity regression guard for #36.
ALL_PROMPT_MESSAGES = [
    prompts.research_topic_prompt("transformer attention"),
    prompts.analyze_paper_prompt("run-001", "2401.12345"),
    prompts.compare_papers_prompt("run-001"),
    prompts.refine_search_prompt("run-001"),
    prompts.quality_assessment_prompt("run-001"),
]


class TestPromptRegistration:
    def test_prompt_count(self) -> None:
        """All 6 prompts should be registered."""
        registered = mcp._prompt_manager._prompts
        assert len(registered) == 6

    def test_prompt_names(self) -> None:
        registered = set(mcp._prompt_manager._prompts.keys())
        expected = {
            "research_topic",
            "research_workflow",
            "analyze_paper",
            "compare_papers",
            "refine_search",
            "quality_assessment",
        }
        assert expected == registered


class TestPromptRoles:
    """Regression guard for #36: MCP prompt messages may only use the
    ``user``/``assistant`` roles. A ``system`` role makes ``prompts/get``
    raise a ValidationError, breaking the prompt at selection time."""

    def test_no_invalid_roles(self) -> None:
        for messages in ALL_PROMPT_MESSAGES:
            assert messages, "prompt returned no messages"
            for msg in messages:
                assert msg["role"] in {"user", "assistant"}, (
                    f"invalid MCP prompt role: {msg['role']!r}"
                )


class TestResearchTopicPrompt:
    def test_returns_messages(self) -> None:
        msgs = prompts.research_topic_prompt("transformer attention")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "transformer attention" in msgs[0]["content"]

    def test_mentions_pipeline_stages(self) -> None:
        msgs = prompts.research_topic_prompt("test topic")
        content = msgs[0]["content"]
        assert "plan" in content
        assert "search" in content
        assert "summarize" in content


class TestAnalyzePaperPrompt:
    def test_returns_messages(self) -> None:
        msgs = prompts.analyze_paper_prompt("run-001", "2401.12345")
        assert len(msgs) == 1
        assert "2401.12345" in msgs[0]["content"]
        assert "run-001" in msgs[0]["content"]

    def test_includes_resource_uri(self) -> None:
        msgs = prompts.analyze_paper_prompt("run-001", "2401.12345")
        assert "runs://run-001/markdown/2401.12345" in msgs[0]["content"]


class TestComparePapersPrompt:
    def test_returns_messages(self) -> None:
        msgs = prompts.compare_papers_prompt("run-001")
        assert len(msgs) == 1
        assert "run-001" in msgs[0]["content"]
        assert "synthesis" in msgs[0]["content"].lower()


class TestRefineSearchPrompt:
    def test_returns_messages(self) -> None:
        msgs = prompts.refine_search_prompt("run-001")
        assert len(msgs) == 1
        assert "candidates" in msgs[0]["content"].lower()


class TestQualityAssessmentPrompt:
    def test_returns_messages(self) -> None:
        msgs = prompts.quality_assessment_prompt("run-001")
        assert len(msgs) == 1
        assert "quality" in msgs[0]["content"].lower()
