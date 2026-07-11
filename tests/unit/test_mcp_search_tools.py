"""Unit tests for the search_tools MCP meta-tool (#120)."""

from __future__ import annotations

from research_pipeline.mcp_server.schemas import SearchToolsInput
from research_pipeline.mcp_server.tools.evaluation import search_tools


def test_name_hit_ranks_above_description_only_hit() -> None:
    """A term in the tool name outweighs the same term in a description."""
    catalog = {
        "tool_convert_pdf": "Does something unrelated.\nSecond line.",
        "tool_other": "Convert files to markdown here.",
    }
    result = search_tools(SearchToolsInput(query="convert"), catalog)
    assert result.success is True
    matches = result.artifacts["matches"]
    assert [m["tool"] for m in matches] == ["tool_convert_pdf", "tool_other"]
    # The summary is only the first line of the description.
    assert matches[0]["summary"] == "Does something unrelated."


def test_limit_truncates_results() -> None:
    """``limit`` caps the number of returned matches; ties break by name."""
    catalog = {
        "tool_data_a": "alpha",
        "tool_data_b": "beta",
        "tool_data_c": "gamma",
    }
    result = search_tools(SearchToolsInput(query="data", limit=2), catalog)
    matches = result.artifacts["matches"]
    assert len(matches) == 2
    assert [m["tool"] for m in matches] == ["tool_data_a", "tool_data_b"]


def test_empty_query_returns_failure() -> None:
    """A whitespace-only query yields success=False and no matches."""
    catalog = {"tool_x": "anything"}
    result = search_tools(SearchToolsInput(query="   "), catalog)
    assert result.success is False
    assert result.artifacts["matches"] == []


def test_no_match_returns_empty_list() -> None:
    """A non-empty query with no hits still succeeds but returns no matches."""
    catalog = {"tool_x": "converts pdfs", "tool_y": "screens papers"}
    result = search_tools(SearchToolsInput(query="zzznomatch"), catalog)
    assert result.success is True
    assert result.artifacts["matches"] == []


def test_multi_term_scoring_prefers_more_matched_terms() -> None:
    """Matching more of the query terms ranks a tool higher."""
    catalog = {
        "tool_alpha": "convert pdf files",
        "tool_beta": "convert only",
    }
    result = search_tools(SearchToolsInput(query="convert pdf"), catalog)
    matches = result.artifacts["matches"]
    # tool_alpha matches both terms; tool_beta matches one → alpha ranks first.
    assert [m["tool"] for m in matches] == ["tool_alpha", "tool_beta"]
    assert result.artifacts["query"] == "convert pdf"
