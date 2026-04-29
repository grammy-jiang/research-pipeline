"""Stage verifier registry for replayable briefing workflows (G02).

Each pipeline stage exposes a structural verifier that runs offline against
its persisted artifacts in the run root. Verifiers return a deterministic
``StageVerification`` result with ``ok`` and a sorted list of issues.

Verifiers are pure functions over the run root. They never reach the
network, never invoke LLMs, and never re-execute the stage. They are the
"verify-before-trust" gate replayable from the workflow state file.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict


class StageVerification(BaseModel):
    """Outcome of running a stage verifier."""

    model_config = ConfigDict(frozen=True)

    stage: str
    ok: bool
    issues: tuple[str, ...] = ()


VerifierFn = Callable[[Path], StageVerification]


def _check_jsonl(path: Path, stage: str, *, allow_empty: bool = False) -> list[str]:
    issues: list[str] = []
    if not path.exists():
        return [f"{stage}: missing artifact {path.name}"]
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines and not allow_empty:
        issues.append(f"{stage}: artifact {path.name} is empty")
    for idx, ln in enumerate(lines, start=1):
        try:
            json.loads(ln)
        except json.JSONDecodeError as exc:
            issues.append(
                f"{stage}: invalid JSON on line {idx} of {path.name}: {exc.msg}"
            )
    return issues


def _check_json(path: Path, stage: str) -> list[str]:
    if not path.exists():
        return [f"{stage}: missing artifact {path.name}"]
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{stage}: invalid JSON in {path.name}: {exc.msg}"]
    return []


def _check_markdown(path: Path, stage: str) -> list[str]:
    if not path.exists():
        return [f"{stage}: missing artifact {path.name}"]
    if not path.read_text(encoding="utf-8").strip():
        return [f"{stage}: markdown artifact {path.name} is empty"]
    return []


def verify_planned(run_root: Path) -> StageVerification:
    """Verify the workflow state was initialized."""
    state_file = run_root / "workflow_state.json"
    issues = _check_json(state_file, "planned")
    return StageVerification(
        stage="planned", ok=not issues, issues=tuple(sorted(issues))
    )


def verify_polled(run_root: Path) -> StageVerification:
    """Verify the polled events JSONL is well-formed."""
    issues = _check_jsonl(run_root / "polled.jsonl", "polled")
    return StageVerification(
        stage="polled", ok=not issues, issues=tuple(sorted(issues))
    )


def verify_ranked(run_root: Path) -> StageVerification:
    """Verify the ranked clusters JSONL is well-formed."""
    issues = _check_jsonl(run_root / "ranked.jsonl", "ranked")
    return StageVerification(
        stage="ranked", ok=not issues, issues=tuple(sorted(issues))
    )


def verify_generated(run_root: Path) -> StageVerification:
    """Verify the generated daily markdown report exists."""
    issues = _check_markdown(run_root / "daily.md", "generated")
    return StageVerification(
        stage="generated", ok=not issues, issues=tuple(sorted(issues))
    )


def verify_validated(run_root: Path) -> StageVerification:
    """Verify the validation result JSON exists and parses."""
    issues = _check_json(run_root / "validation.json", "validated")
    return StageVerification(
        stage="validated", ok=not issues, issues=tuple(sorted(issues))
    )


def verify_archived(run_root: Path) -> StageVerification:
    """Verify archive marker is present."""
    issues: list[str] = []
    marker = run_root / "archived.json"
    if not marker.exists():
        issues.append("archived: missing marker archived.json")
    else:
        issues.extend(_check_json(marker, "archived"))
    return StageVerification(
        stage="archived", ok=not issues, issues=tuple(sorted(issues))
    )


_REGISTRY: Final[dict[str, VerifierFn]] = {
    "planned": verify_planned,
    "polled": verify_polled,
    "ranked": verify_ranked,
    "generated": verify_generated,
    "validated": verify_validated,
    "archived": verify_archived,
}


def stage_verifiers() -> Mapping[str, VerifierFn]:
    """Return the registry as a read-only mapping."""
    return dict(_REGISTRY)


def get_stage_verifier(stage: str) -> VerifierFn:
    """Return the verifier for a stage, or raise ``KeyError``."""
    if stage not in _REGISTRY:
        raise KeyError(f"unknown briefing stage: {stage}")
    return _REGISTRY[stage]


def verify_stage(stage: str, run_root: Path) -> StageVerification:
    """Verify a single stage by name."""
    return get_stage_verifier(stage)(run_root)


def verify_completed_stages(
    completed_stages: tuple[str, ...], run_root: Path
) -> tuple[StageVerification, ...]:
    """Verify every completed stage in order; raises on unknown stages."""
    return tuple(verify_stage(stage, run_root) for stage in completed_stages)
