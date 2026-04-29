"""Tests for the briefing workflow state model (G01)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.briefing.models import BriefingWorkflowState
from research_pipeline.briefing.workflow_state import (
    advance_workflow_state,
    load_workflow_state,
    save_workflow_state,
    state_path,
)


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    return root


def test_state_path_returns_workflow_state_json(run_root: Path) -> None:
    assert state_path(run_root) == run_root / "workflow_state.json"


def test_load_initial_state_when_missing(run_root: Path) -> None:
    state = load_workflow_state(run_root, "2026-04-20")
    assert state.run_date == "2026-04-20"
    assert state.current_stage == "planned"
    assert state.completed_stages == ()
    assert state.artifact_paths == {}
    assert state.last_error == ""
    # No file is created on a pure load.
    assert not state_path(run_root).exists()


def test_save_and_reload_roundtrip(run_root: Path) -> None:
    original = BriefingWorkflowState(
        run_date="2026-04-20",
        current_stage="ranked",
        completed_stages=("planned", "polled", "ranked"),
        artifact_paths={"polled": "polled.jsonl"},
        last_error="",
    )
    save_workflow_state(run_root, original)
    assert state_path(run_root).exists()

    loaded = load_workflow_state(run_root, "2026-04-20")
    assert loaded == original


def test_advance_creates_state_and_records_stage(run_root: Path) -> None:
    state = advance_workflow_state(
        run_root,
        run_date="2026-04-20",
        stage="polled",
        artifacts={"polled": "polled.jsonl"},
    )
    assert state.current_stage == "polled"
    assert state.completed_stages == ("polled",)
    assert state.artifact_paths == {"polled": "polled.jsonl"}
    # Persisted to disk.
    on_disk = load_workflow_state(run_root, "2026-04-20")
    assert on_disk == state


def test_advance_merges_artifacts_and_dedupes_completed(run_root: Path) -> None:
    advance_workflow_state(
        run_root,
        run_date="2026-04-20",
        stage="polled",
        artifacts={"polled": "polled.jsonl"},
    )
    advance_workflow_state(
        run_root,
        run_date="2026-04-20",
        stage="ranked",
        artifacts={"ranked": "ranked.jsonl"},
    )
    state = advance_workflow_state(
        run_root,
        run_date="2026-04-20",
        stage="ranked",  # repeated stage
        artifacts={"ranked": "ranked.jsonl"},
    )
    # Stage repeated — completed_stages dedupes, preserves order.
    assert state.completed_stages == ("polled", "ranked")
    assert state.current_stage == "ranked"
    assert state.artifact_paths == {
        "polled": "polled.jsonl",
        "ranked": "ranked.jsonl",
    }


def test_advance_clears_last_error(run_root: Path) -> None:
    failing = BriefingWorkflowState(
        run_date="2026-04-20",
        current_stage="failed",
        completed_stages=("planned",),
        artifact_paths={},
        last_error="boom",
    )
    save_workflow_state(run_root, failing)

    state = advance_workflow_state(
        run_root,
        run_date="2026-04-20",
        stage="polled",
        artifacts=None,
    )
    assert state.last_error == ""
    assert state.current_stage == "polled"


def test_advance_with_no_artifacts_preserves_existing(run_root: Path) -> None:
    advance_workflow_state(
        run_root,
        run_date="2026-04-20",
        stage="polled",
        artifacts={"polled": "polled.jsonl"},
    )
    state = advance_workflow_state(
        run_root,
        run_date="2026-04-20",
        stage="ranked",
        artifacts=None,
    )
    assert state.artifact_paths == {"polled": "polled.jsonl"}


def test_workflow_state_model_is_frozen() -> None:
    state = BriefingWorkflowState(run_date="2026-04-20")
    with pytest.raises(Exception):  # noqa: B017  # pydantic ValidationError on frozen
        state.current_stage = "polled"  # type: ignore[misc]


def test_invalid_stage_rejected() -> None:
    with pytest.raises(Exception):  # noqa: B017
        BriefingWorkflowState(
            run_date="2026-04-20",
            current_stage="not-a-stage",  # type: ignore[arg-type]
        )
