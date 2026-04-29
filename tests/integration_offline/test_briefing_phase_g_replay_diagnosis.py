"""Replay / diagnosis integration test for Phase G (G09).

Verifies that:

1. The replay-diagnosis and final-acceptance-gate docs exist and contain the
   required sections.
2. ``BriefingWorkflowState`` + stage verifiers can drive a replay scenario:
   load state, verify each completed stage, detect drift when an artifact is
   removed.
"""

from __future__ import annotations

import json
from pathlib import Path

from research_pipeline.briefing.workflow_state import (
    advance_workflow_state,
    load_workflow_state,
)
from research_pipeline.briefing.workflow_verification import (
    verify_completed_stages,
    verify_stage,
)

_DOCS = Path(__file__).parent.parent.parent / "docs" / "daily-ai-intelligence"


def test_replay_diagnosis_doc_exists() -> None:
    path = _DOCS / "replay-diagnosis.md"
    assert path.exists(), "replay-diagnosis.md missing"
    text = path.read_text(encoding="utf-8")
    for section in (
        "State model",
        "Stage verifiers",
        "Replay procedure",
        "Diagnosis checklist",
    ):
        assert section in text, f"replay-diagnosis.md missing section: {section}"


def test_final_acceptance_gate_doc_exists() -> None:
    path = _DOCS / "final-acceptance-gate.md"
    assert path.exists(), "final-acceptance-gate.md missing"
    text = path.read_text(encoding="utf-8")
    for section in (
        "Required commands",
        "Required documents",
        "Completion condition",
    ):
        assert section in text, f"final-acceptance-gate.md missing section: {section}"
    # Must reference Phase G test files.
    assert "phase_g" in text or "Phase G" in text
    assert "governance" in text.lower()


def _build_run_root(tmp_path: Path) -> Path:
    """Create a run root with all stage artifacts the verifiers expect."""
    root = tmp_path / "run"
    root.mkdir()
    (root / "polled.jsonl").write_text('{"id": "ev1"}\n')
    (root / "ranked.jsonl").write_text('{"cluster_id": "c1"}\n')
    (root / "daily.md").write_text("# Daily Brief\n")
    (root / "validation.json").write_text(json.dumps({"passed": True}))
    (root / "archived.json").write_text(json.dumps({"archived_at": "2026-04-20"}))
    return root


def test_replay_loads_state_and_verifies_completed_stages(tmp_path: Path) -> None:
    root = _build_run_root(tmp_path)
    # Drive the workflow through every stage.
    advance_workflow_state(root, run_date="2026-04-20", stage="planned")
    advance_workflow_state(root, run_date="2026-04-20", stage="polled")
    advance_workflow_state(root, run_date="2026-04-20", stage="ranked")
    advance_workflow_state(root, run_date="2026-04-20", stage="generated")
    advance_workflow_state(root, run_date="2026-04-20", stage="validated")
    advance_workflow_state(root, run_date="2026-04-20", stage="archived")

    state = load_workflow_state(root, run_date="2026-04-20")
    assert state.current_stage == "archived"
    assert "polled" in state.completed_stages
    assert "validated" in state.completed_stages

    results = verify_completed_stages(state.completed_stages, root)
    assert all(r.ok for r in results), [r for r in results if not r.ok]


def test_replay_detects_artifact_drift(tmp_path: Path) -> None:
    """If a stage artifact disappears, the verifier flags it."""
    root = _build_run_root(tmp_path)
    advance_workflow_state(root, run_date="2026-04-20", stage="planned")
    advance_workflow_state(root, run_date="2026-04-20", stage="polled")
    advance_workflow_state(root, run_date="2026-04-20", stage="ranked")

    # Drift: ranked artifact deleted post-run.
    (root / "ranked.jsonl").unlink()

    result = verify_stage("ranked", root)
    assert result.ok is False
    assert result.issues
    # All issues are sorted strings.
    assert list(result.issues) == sorted(result.issues)


def test_failed_run_state_is_replayable(tmp_path: Path) -> None:
    """A failed run keeps last_error and current_stage for diagnosis."""
    root = tmp_path / "run"
    root.mkdir()
    advance_workflow_state(root, run_date="2026-04-20", stage="planned")
    state = load_workflow_state(root, run_date="2026-04-20")
    # Simulate failure mid-run by writing failed state directly via the model.
    failed = state.model_copy(
        update={"current_stage": "failed", "last_error": "poll timeout"}
    )
    from research_pipeline.briefing.workflow_state import save_workflow_state

    save_workflow_state(root, failed)
    reloaded = load_workflow_state(root, run_date="2026-04-20")
    assert reloaded.current_stage == "failed"
    assert reloaded.last_error == "poll timeout"
