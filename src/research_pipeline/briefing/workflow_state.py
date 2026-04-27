"""Replayable workflow state for daily intelligence briefing runs."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.briefing.io import read_json, write_json
from research_pipeline.briefing.models import BriefingWorkflowState


def state_path(run_root: Path) -> Path:
    """Return the workflow state path for a briefing run."""
    return run_root / "workflow_state.json"


def load_workflow_state(run_root: Path, run_date: str) -> BriefingWorkflowState:
    """Load workflow state or create the initial planned state."""
    path = state_path(run_root)
    if not path.exists():
        return BriefingWorkflowState(run_date=run_date)
    return BriefingWorkflowState.model_validate(read_json(path))


def save_workflow_state(run_root: Path, state: BriefingWorkflowState) -> None:
    """Persist workflow state."""
    write_json(state_path(run_root), state.model_dump(mode="json"))


def advance_workflow_state(
    run_root: Path,
    *,
    run_date: str,
    stage: str,
    artifacts: dict[str, str] | None = None,
) -> BriefingWorkflowState:
    """Advance state after a successful stage."""
    current = load_workflow_state(run_root, run_date)
    completed = tuple(dict.fromkeys((*current.completed_stages, stage)))
    merged_artifacts = {**current.artifact_paths, **(artifacts or {})}
    next_state = current.model_copy(
        update={
            "current_stage": stage,
            "completed_stages": completed,
            "artifact_paths": merged_artifacts,
            "last_error": "",
        }
    )
    save_workflow_state(run_root, next_state)
    return next_state
