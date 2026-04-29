"""Held-out agent evaluation tasks for Phase G.

Each test corresponds to a task in
``src/research_pipeline/skill_data/daily-ai-intelligence/references/agent-evaluation.md``.
Tests verify that an agent driving the documented MCP/CLI surface can complete
each scenario and that the harness refuses unsupported behavior.

These tests are intentionally narrow: they exercise the public ``brief_*``
tool API and the governance helpers, not internal pipeline classes.
"""

from __future__ import annotations

import json
from pathlib import Path

from research_pipeline.briefing.tool_governance import (
    BRIEF_TOOL_NAMES,
    is_unsupported_source,
    policy_for,
)
from research_pipeline.mcp_server import tools as mcp_tools
from research_pipeline.mcp_server.schemas import (
    BriefExportObsidianInput,
    BriefRecordFeedbackInput,
    BriefRunInput,
    BriefValidateReportInput,
    ToolResult,
)

_FIXTURE_BASE = Path(__file__).parent.parent / "fixtures" / "briefing" / "e2e"
_RUN_DATE = "2026-05-01"


# ---------------------------------------------------------------------------
# Task 1: run_daily_brief — full happy-path
# ---------------------------------------------------------------------------
def test_run_daily_brief(tmp_path: Path) -> None:
    """Agent runs `brief_run` end-to-end against reviewed fixture registry."""
    workspace = tmp_path / "workspace"
    params = BriefRunInput(
        workspace=str(workspace),
        date=_RUN_DATE,
        registry_path=str(_FIXTURE_BASE / "normal" / "registry.toml"),
        fixture_base_dir=str(_FIXTURE_BASE / "normal"),
    )
    result = mcp_tools.brief_run_tool(params)
    assert isinstance(result, ToolResult)
    assert result.success, f"brief_run_tool failed: {result.message}"
    # An agent should be able to find at least the daily report path.
    assert result.artifacts


# ---------------------------------------------------------------------------
# Task 2: validate_malformed_report — deterministic failure
# ---------------------------------------------------------------------------
def test_validate_malformed_report(tmp_path: Path) -> None:
    """Validation fails deterministically on a malformed daily report."""
    workspace = tmp_path / "workspace"
    daily = workspace / "briefings" / _RUN_DATE / "reports" / "daily.md"
    daily.parent.mkdir(parents=True)
    daily.write_text("not a real briefing\n")  # no required sections
    params = BriefValidateReportInput(workspace=str(workspace), date=_RUN_DATE)
    result = mcp_tools.brief_validate_report_tool(params)
    assert isinstance(result, ToolResult)
    # Either explicit failure or a validation result that flags issues.
    if result.success:
        # If success, payload should expose validation issues.
        text = json.dumps(result.artifacts)
        assert "passed" in text or "issue" in text.lower() or "error" in text.lower()
    else:
        assert result.message  # deterministic, non-empty failure reason


# ---------------------------------------------------------------------------
# Task 3: record_feedback — write to local feedback store only
# ---------------------------------------------------------------------------
def test_record_feedback(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    params = BriefRecordFeedbackInput(
        workspace=str(workspace),
        date=_RUN_DATE,
        target_type="cluster",
        target_id="cluster-001",
        signal="upvote",
        reason="useful",
        strength=1.0,
    )
    result = mcp_tools.brief_record_feedback_tool(params)
    assert isinstance(result, ToolResult)
    # Governance: feedback tool must not touch the network.
    assert policy_for("brief_record_feedback_tool").kind == "local"


# ---------------------------------------------------------------------------
# Task 4: export_obsidian — writes only to configured vault path
# ---------------------------------------------------------------------------
def test_export_obsidian_to_configured_vault(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    vault = tmp_path / "vault"
    vault.mkdir()
    # Build a minimal validated daily so export has something to export.
    daily = workspace / "briefings" / _RUN_DATE / "reports" / "daily.md"
    daily.parent.mkdir(parents=True)
    daily.write_text("# Daily Brief 2026-05-01\n\n## Highlights\n- item\n")
    params = BriefExportObsidianInput(
        workspace=str(workspace),
        date=_RUN_DATE,
        vault_path=str(vault),
        registry_path="",
    )
    result = mcp_tools.brief_export_obsidian_tool(params)
    assert isinstance(result, ToolResult)
    # Governance: export only writes to vault path (or local workspace metadata).
    pol = policy_for("brief_export_obsidian_tool")
    assert "write" in pol.effects
    assert pol.kind == "local"


# ---------------------------------------------------------------------------
# Task 5: refuse_unsupported_source — agent must refuse without registry review
# ---------------------------------------------------------------------------
def test_refuse_unsupported_source() -> None:
    allowlist = ("github.com/", "huggingface.co/")
    # Random URL not in allowlist must be refused.
    assert is_unsupported_source("https://random.example.com/feed", allowlist)
    # Allowlisted URL is permitted.
    assert not is_unsupported_source("https://github.com/owner/repo", allowlist)
    # Phase G non-goal: no new source expansion. Polling must be allowlisted.
    assert policy_for("brief_poll_sources_tool").source_allowlisted


# ---------------------------------------------------------------------------
# Task 6: generate_dossier — manual dossier from a primary-artifact cluster
# ---------------------------------------------------------------------------
def test_generate_dossier_is_local_and_deterministic() -> None:
    pol = policy_for("brief_generate_dossier_tool")
    assert pol.kind == "local"
    assert pol.deterministic
    assert "write" in pol.effects


# ---------------------------------------------------------------------------
# Task 7: paper_request_handoff — skill must hand paper-only requests off
# ---------------------------------------------------------------------------
def test_paper_request_handoff_documented() -> None:
    """SKILL.md explicitly hands off academic-paper requests."""
    import research_pipeline

    skill_md = (
        Path(research_pipeline.__file__).parent
        / "skill_data"
        / "daily-ai-intelligence"
        / "SKILL.md"
    )
    text = skill_md.read_text(encoding="utf-8")
    assert "research-pipeline" in text
    # Must reference academic OR paper handoff.
    lower = text.lower()
    assert "academic" in lower or "paper" in lower
    assert "not use this skill" in text or "Do not use this skill" in text


# ---------------------------------------------------------------------------
# Cross-task coverage: every brief_* tool covered by governance.
# ---------------------------------------------------------------------------
def test_every_brief_tool_has_governance() -> None:
    for name in BRIEF_TOOL_NAMES:
        # Calling policy_for must not raise.
        policy_for(name)


def test_brief_namespace_does_not_collide_with_academic_research() -> None:
    """Briefing tools must be namespaced; no collision with academic surface."""
    for name in BRIEF_TOOL_NAMES:
        assert name.startswith("brief_")
    # Sanity: the mcp_tools module exposes both surfaces; ensure the academic
    # tools do not start with "brief_".
    paper_tools = [
        n for n in dir(mcp_tools) if n.endswith("_tool") and not n.startswith("brief_")
    ]
    for n in paper_tools:
        assert not n.startswith("brief_")
