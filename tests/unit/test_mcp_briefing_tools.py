"""Tests for namespaced brief_* MCP tool implementations (G04)."""

from __future__ import annotations

import inspect
from pathlib import Path

from research_pipeline.mcp_server import tools
from research_pipeline.mcp_server.schemas import (
    BriefCommonInput,
    BriefExportObsidianInput,
    BriefGenerateDailyInput,
    BriefGenerateDossierInput,
    BriefPollSourcesInput,
    BriefRankEventsInput,
    BriefRecordFeedbackInput,
    BriefRunInput,
    BriefValidateReportInput,
    BriefWeeklySynthesisInput,
    ToolResult,
)

BRIEF_TOOL_NAMES = [
    "brief_poll_sources_tool",
    "brief_rank_events_tool",
    "brief_generate_daily_tool",
    "brief_validate_report_tool",
    "brief_run_tool",
    "brief_export_obsidian_tool",
    "brief_record_feedback_tool",
    "brief_generate_dossier_tool",
    "brief_weekly_synthesis_tool",
]


def test_all_brief_tools_present_and_callable() -> None:
    for name in BRIEF_TOOL_NAMES:
        fn = getattr(tools, name, None)
        assert fn is not None, f"Missing brief tool: {name}"
        assert callable(fn), f"Brief tool not callable: {name}"


def test_brief_tools_use_brief_namespace() -> None:
    for name in BRIEF_TOOL_NAMES:
        assert name.startswith("brief_"), name
        assert name.endswith("_tool"), name


def test_brief_tools_first_param_is_brief_input(tmp_path) -> None:
    expected = {
        "brief_poll_sources_tool": BriefPollSourcesInput,
        "brief_rank_events_tool": BriefRankEventsInput,
        "brief_generate_daily_tool": BriefGenerateDailyInput,
        "brief_validate_report_tool": BriefValidateReportInput,
        "brief_run_tool": BriefRunInput,
        "brief_export_obsidian_tool": BriefExportObsidianInput,
        "brief_record_feedback_tool": BriefRecordFeedbackInput,
        "brief_generate_dossier_tool": BriefGenerateDossierInput,
        "brief_weekly_synthesis_tool": BriefWeeklySynthesisInput,
    }
    for name, schema in expected.items():
        fn = getattr(tools, name)
        sig = inspect.signature(fn)
        first_param = next(iter(sig.parameters.values()))
        ann = first_param.annotation
        # Annotations may be a forward-ref string under PEP 563.
        if isinstance(ann, str):
            assert ann == schema.__name__, (
                f"{name}: expected {schema.__name__}, got {ann}"
            )
        else:
            assert ann is schema, f"{name}: expected {schema}, got {ann}"


def test_brief_tool_returns_tool_result_on_failure(tmp_path) -> None:
    # Pointing at an empty tmp dir with no fixtures yields a controlled failure.
    params = BriefPollSourcesInput(
        workspace=str(tmp_path),
        date="2026-04-20",
        registry_path="",
        fixture_base_dir="",
    )
    result = tools.brief_poll_sources_tool(params)
    assert isinstance(result, ToolResult)
    # No registry => failure message; success may be False.
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)


def test_brief_tools_surface_errors_via_iserror(tmp_path) -> None:
    """On missing inputs each brief tool either returns a ToolResult (an
    expected business outcome) or raises a RuntimeError — which FastMCP maps to
    isError=true — with the caller's home directory scrubbed from the message.
    Genuine failures are no longer swallowed into a success-shaped ToolResult.
    See #38/#44."""
    cases = [
        (
            "brief_validate_report_tool",
            BriefValidateReportInput(workspace=str(tmp_path), date="2026-04-20"),
        ),
        (
            "brief_generate_daily_tool",
            BriefGenerateDailyInput(workspace=str(tmp_path), date="2026-04-20"),
        ),
        (
            "brief_export_obsidian_tool",
            BriefExportObsidianInput(
                workspace=str(tmp_path),
                date="2026-04-20",
                vault_path=str(tmp_path / "v"),
            ),
        ),
        (
            "brief_record_feedback_tool",
            BriefRecordFeedbackInput(
                workspace=str(tmp_path),
                date="2026-04-20",
                target_type="event",
                target_id="x",
                signal="upvote",
            ),
        ),
        (
            "brief_generate_dossier_tool",
            BriefGenerateDossierInput(
                workspace=str(tmp_path), date="2026-04-20", cluster_id="cl1"
            ),
        ),
        (
            "brief_weekly_synthesis_tool",
            BriefWeeklySynthesisInput(workspace=str(tmp_path), week="2026-W18"),
        ),
    ]
    home = str(Path.home())
    for name, params in cases:
        fn = getattr(tools, name)
        result = None
        error_msg = None
        try:
            result = fn(params)
        except RuntimeError as exc:
            error_msg = str(exc)
        if error_msg is not None:
            # Genuine failure surfaced as a raise (isError) with home scrubbed.
            assert home not in error_msg, name
        else:
            assert isinstance(result, ToolResult), name


def test_brief_common_input_is_base() -> None:
    assert issubclass(BriefPollSourcesInput, BriefCommonInput)
    assert issubclass(BriefRankEventsInput, BriefCommonInput)
    assert issubclass(BriefGenerateDailyInput, BriefCommonInput)
    assert issubclass(BriefRunInput, BriefPollSourcesInput)
