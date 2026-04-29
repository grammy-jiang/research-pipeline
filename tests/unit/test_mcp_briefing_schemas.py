"""Tests for namespaced brief_* MCP input schemas (G03)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_pipeline.mcp_server import schemas

BRIEF_INPUT_CLASSES = [
    "BriefCommonInput",
    "BriefPollSourcesInput",
    "BriefRankEventsInput",
    "BriefGenerateDailyInput",
    "BriefValidateReportInput",
    "BriefRunInput",
    "BriefExportObsidianInput",
    "BriefRecordFeedbackInput",
    "BriefGenerateDossierInput",
    "BriefWeeklySynthesisInput",
]


def test_all_brief_input_schemas_exist() -> None:
    for name in BRIEF_INPUT_CLASSES:
        assert hasattr(schemas, name), f"Missing brief schema: {name}"


def test_brief_common_input_defaults() -> None:
    obj = schemas.BriefCommonInput()
    # Workspace defaults to a relative path; date may default empty (current).
    assert obj.workspace
    assert obj.date == ""


def test_brief_poll_sources_input_optional_paths() -> None:
    obj = schemas.BriefPollSourcesInput()
    assert obj.registry_path == ""
    assert obj.fixture_base_dir == ""
    full = schemas.BriefPollSourcesInput(
        workspace="./ws",
        date="2026-04-20",
        registry_path="reg.toml",
        fixture_base_dir="/tmp/fix",
    )
    assert full.date == "2026-04-20"
    assert full.registry_path == "reg.toml"


def test_brief_rank_events_defaults() -> None:
    obj = schemas.BriefRankEventsInput()
    assert obj.use_memory is True
    assert obj.use_feedback is True


def test_brief_generate_daily_inherits_common() -> None:
    obj = schemas.BriefGenerateDailyInput(workspace="./ws", date="2026-04-20")
    assert obj.workspace == "./ws"
    assert obj.date == "2026-04-20"


def test_brief_validate_report_inherits_common() -> None:
    obj = schemas.BriefValidateReportInput()
    assert isinstance(obj, schemas.BriefCommonInput)


def test_brief_run_extends_poll_sources() -> None:
    obj = schemas.BriefRunInput()
    assert isinstance(obj, schemas.BriefPollSourcesInput)


def test_brief_export_obsidian_requires_vault_path() -> None:
    with pytest.raises(ValidationError):
        schemas.BriefExportObsidianInput()  # type: ignore[call-arg]
    obj = schemas.BriefExportObsidianInput(vault_path="/v")
    assert obj.vault_path == "/v"


def test_brief_record_feedback_requires_fields() -> None:
    with pytest.raises(ValidationError):
        schemas.BriefRecordFeedbackInput()  # type: ignore[call-arg]
    obj = schemas.BriefRecordFeedbackInput(
        target_type="event", target_id="ev1", signal="upvote"
    )
    assert obj.target_type == "event"
    assert obj.strength == 1.0
    assert obj.reason == ""


def test_brief_generate_dossier_requires_cluster_id() -> None:
    with pytest.raises(ValidationError):
        schemas.BriefGenerateDossierInput()  # type: ignore[call-arg]
    obj = schemas.BriefGenerateDossierInput(cluster_id="cl1")
    assert obj.cluster_id == "cl1"


def test_brief_weekly_synthesis_requires_week() -> None:
    with pytest.raises(ValidationError):
        schemas.BriefWeeklySynthesisInput()  # type: ignore[call-arg]
    obj = schemas.BriefWeeklySynthesisInput(week="2026-W18")
    assert obj.week == "2026-W18"
    assert obj.workspace
    assert obj.output_path == ""


def test_brief_namespace_is_consistent() -> None:
    # All briefing input classes start with "Brief" (no other namespace).
    for name in BRIEF_INPUT_CLASSES:
        assert name.startswith("Brief"), name


def test_no_tool_uses_unbriefed_naming_for_briefing_input() -> None:
    # Ensure briefing input schemas are not aliased under a non-namespaced name.
    suspect = ["DailyPollInput", "PollSourcesInput", "RankEventsInput"]
    for name in suspect:
        assert not hasattr(schemas, name), f"Non-namespaced briefing schema: {name}"
