"""Tests for horizon_metric and rrp_diagnostic MCP tools."""

from __future__ import annotations

from mcp_server.schemas import HorizonMetricInput, RRPDiagnosticInput
from mcp_server.tools import horizon_metric_tool, rrp_diagnostic_tool


def test_horizon_metric_tool_success() -> None:
    result = horizon_metric_tool(
        HorizonMetricInput(
            normalized_score=0.8,
            difficulty=0.5,
            achieved_steps=50,
            target_steps=50,
        )
    )
    assert result.success is True
    assert "uhm" in result.artifacts
    assert 0.0 <= result.artifacts["uhm"] <= 1.0


def test_horizon_metric_tool_zero_components() -> None:
    result = horizon_metric_tool(
        HorizonMetricInput(
            normalized_score=0.0,
            achieved_steps=0,
            target_steps=100,
        )
    )
    assert result.success is True
    assert result.artifacts["uhm"] == 0.0


def test_rrp_diagnostic_tool_success() -> None:
    result = rrp_diagnostic_tool(
        RRPDiagnosticInput(
            report_text=(
                "# Executive Summary\narXiv:X [1]\n## Themes\n## Contradictions\n"
                "contradict this. ## Gaps\n open question. "
                "## Confidence\n high confidence. " + ("word " * 600)
            ),
            shortlist_ids=["arXiv:X"],
        )
    )
    assert result.success is True
    assert set(result.artifacts) >= {
        "recall",
        "reasoning",
        "presentation",
        "overall",
        "bottleneck",
    }


def test_rrp_diagnostic_tool_empty_shortlist() -> None:
    result = rrp_diagnostic_tool(
        RRPDiagnosticInput(report_text="Some text.", shortlist_ids=[])
    )
    assert result.success is True
    assert result.artifacts["bottleneck"] in {
        "recall",
        "reasoning",
        "presentation",
    }
