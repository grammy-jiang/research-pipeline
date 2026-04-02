"""Tests for MCP server registration."""

from mcp_server.server import mcp


class TestServerRegistration:
    def test_server_name(self) -> None:
        assert mcp.name == "research-pipeline"

    def test_all_tools_registered(self) -> None:
        expected_tools = {
            "tool_plan_topic",
            "tool_search",
            "tool_screen_candidates",
            "tool_download_pdfs",
            "tool_convert_pdfs",
            "tool_extract_content",
            "tool_summarize_papers",
            "tool_run_pipeline",
            "tool_get_run_manifest",
            "tool_convert_file",
        }
        registered = set(mcp._tool_manager._tools.keys())
        assert expected_tools.issubset(
            registered
        ), f"Missing tools: {expected_tools - registered}"

    def test_tool_count(self) -> None:
        # 9 pipeline tools + 1 convert_file = 10
        assert len(mcp._tool_manager._tools) == 10
