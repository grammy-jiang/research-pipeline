"""Capability-domain toolsets for the MCP server (#46).

All 64 tools are statically injected on every session, so a client pays the
whole schema set (~tens of K tokens) even when it only needs the research
pipeline. This module groups the tools into capability domains and lets an
operator select a subset via the ``RESEARCH_PIPELINE_MCP_TOOLSETS`` environment
variable (comma-separated domain names). The default — unset — keeps every
domain, so behaviour is unchanged unless a client opts in.

Example: ``RESEARCH_PIPELINE_MCP_TOOLSETS=pipeline,inspection`` exposes only the
core pipeline + inspection tools; the briefing, knowledge, quality, and
diagnostics domains are pruned before the client ever sees ``tools/list``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

ENV_VAR = "RESEARCH_PIPELINE_MCP_TOOLSETS"

# Explicit domain -> tool-name mapping. Every registered tool must appear in
# exactly one domain (enforced by tests/unit/test_mcp_toolsets.py so a new tool
# cannot be added without classifying it).
TOOLSETS: dict[str, frozenset[str]] = {
    "pipeline": frozenset(
        {
            "tool_plan_topic",
            "tool_search",
            "tool_screen_candidates",
            "tool_download_pdfs",
            "tool_convert_file",
            "tool_convert_fine",
            "tool_convert_pdfs",
            "tool_convert_rough",
            "tool_extract_content",
            "tool_summarize_papers",
            "tool_run_pipeline",
            "tool_research_workflow",
            "tool_expand_citations",
            "tool_enrich",
            "tool_watch",
        }
    ),
    "inspection": frozenset(
        {
            "tool_get_run_manifest",
            "tool_list_backends",
            "tool_manage_index",
            "tool_model_routing_info",
            "tool_gate_info",
            "tool_query_eval_log",
            "tool_search_tools",
        }
    ),
    "quality": frozenset(
        {
            "tool_report",
            "tool_cluster",
            "tool_cite_context",
            "tool_export_bibtex",
            "tool_export_html",
            "tool_aggregate_evidence",
            "tool_evaluate_quality",
            "tool_get_venue_tier",
            "tool_compute_semantic_scores",
            "tool_analyze_papers",
            "tool_compare_runs",
            "tool_validate_report",
            "tool_verify_stage",
            "tool_record_feedback",
            "tool_evaluate",
        }
    ),
    "diagnostics": frozenset(
        {
            "tool_adaptive_stopping",
            "tool_blinding_audit",
            "tool_coherence",
            "tool_confidence_layers",
            "tool_consolidation",
            "tool_dual_metrics",
            "tool_horizon_metric",
            "tool_rrp_diagnostic",
        }
    ),
    "knowledge": frozenset(
        {
            "tool_kg_ingest",
            "tool_kg_quality",
            "tool_kg_query",
            "tool_kg_stats",
            "tool_memory_episodes",
            "tool_memory_search",
            "tool_memory_stats",
            "tool_cbr_lookup",
            "tool_cbr_retain",
            "tool_analyze_claims",
            "tool_score_claims",
        }
    ),
    "briefing": frozenset(
        {
            "brief_export_obsidian",
            "brief_generate_daily",
            "brief_generate_dossier",
            "brief_poll_sources",
            "brief_rank_events",
            "brief_record_feedback",
            "brief_run",
            "brief_validate_report",
            "brief_weekly_synthesis",
        }
    ),
}


def _requested_domains(raw: str | None) -> set[str] | None:
    """Parse the env var into a set of valid domain names, or None for all."""
    if not raw or not raw.strip():
        return None
    requested = {part.strip() for part in raw.split(",") if part.strip()}
    valid = requested & set(TOOLSETS)
    unknown = requested - set(TOOLSETS)
    if unknown:
        logger.warning("%s: ignoring unknown toolset(s): %s", ENV_VAR, sorted(unknown))
    if not valid:
        logger.warning("%s: no valid toolsets, keeping all", ENV_VAR)
        return None
    return valid


def apply_toolsets(mcp: FastMCP, raw: str | None = None) -> set[str]:
    """Prune registered tools to the toolsets selected via the environment.

    Returns the set of active domain names. With no selection every domain is
    kept and nothing is pruned.
    """
    if raw is None:
        raw = os.environ.get(ENV_VAR)
    active = _requested_domains(raw)
    if active is None:
        return set(TOOLSETS)

    keep: set[str] = set()
    for domain in active:
        keep |= TOOLSETS[domain]
    tools = mcp._tool_manager._tools
    pruned = [name for name in tools if name not in keep]
    for name in pruned:
        del tools[name]
    logger.info(
        "%s active: %s (%d tools, pruned %d)",
        ENV_VAR,
        sorted(active),
        len(tools),
        len(pruned),
    )
    return active
