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
            "tool_list_backends",
            "tool_expand_citations",
            "tool_evaluate_quality",
            "tool_convert_rough",
            "tool_convert_fine",
            "tool_manage_index",
        }
        registered = set(mcp._tool_manager._tools.keys())
        assert expected_tools.issubset(
            registered
        ), f"Missing tools: {expected_tools - registered}"

    def test_tool_count(self) -> None:
        # 9 pipeline + convert_file + list_backends + 5 new
        # + workflow + 4 quality + feedback + eval_log + aggregate
        # + export_html + model_routing_info + gate_info = 27
        assert len(mcp._tool_manager._tools) == 33

    def test_all_tools_have_annotations(self) -> None:
        """Every registered tool must have ToolAnnotations set."""
        for name, tool in mcp._tool_manager._tools.items():
            assert tool.annotations is not None, f"Tool {name} missing annotations"

    def test_readonly_tools(self) -> None:
        """Read-only tools should have readOnlyHint=True."""
        readonly_tools = {"tool_get_run_manifest", "tool_list_backends"}
        for name in readonly_tools:
            tool = mcp._tool_manager._tools[name]
            assert tool.annotations.readOnlyHint is True, f"{name} should be readOnly"

    def test_openworld_tools(self) -> None:
        """Tools that call external APIs should have openWorldHint=True."""
        openworld_tools = {
            "tool_search",
            "tool_download_pdfs",
            "tool_run_pipeline",
            "tool_expand_citations",
            "tool_evaluate_quality",
        }
        for name in openworld_tools:
            tool = mcp._tool_manager._tools[name]
            assert tool.annotations.openWorldHint is True, f"{name} should be openWorld"
