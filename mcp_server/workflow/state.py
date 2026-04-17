"""Workflow state model with AER-pattern execution records.

Implements the state foundation for the research workflow engine.
Key harness principles:
- Schema-level safety: invalid state transitions are type errors (OpenDev)
- Non-identifiability: execution records capture decision provenance (AER)
- Recovery dominance: state is JSON-serializable and crash-resumable

Every stage transition produces a new immutable WorkflowState snapshot.
The execution_log is append-only — provenance cannot be reconstructed
from state alone, so we record it at runtime.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StageStatus(StrEnum):
    """Status of an individual pipeline stage.

    The VERIFIED status enforces verify-before-commit (SAVER pattern):
    a stage must be structurally verified, not just completed, before
    the next stage can begin.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStage(StrEnum):
    """Pipeline stages in execution order."""

    PLAN = "plan"
    SEARCH = "search"
    SCREEN = "screen"
    QUALITY = "quality"
    EXPAND = "expand"
    DOWNLOAD = "download"
    CONVERT = "convert"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    ITERATE = "iterate"
    REPORT = "report"


# Schema-level safety: valid transitions are structurally defined.
# Any transition not in this map is a type/governance error.
VALID_TRANSITIONS: dict[str, set[str]] = {
    WorkflowStage.PLAN: {WorkflowStage.SEARCH},
    WorkflowStage.SEARCH: {WorkflowStage.SCREEN},
    WorkflowStage.SCREEN: {
        WorkflowStage.QUALITY,
        WorkflowStage.EXPAND,
        WorkflowStage.DOWNLOAD,
    },
    WorkflowStage.QUALITY: {WorkflowStage.EXPAND, WorkflowStage.DOWNLOAD},
    WorkflowStage.EXPAND: {WorkflowStage.SCREEN, WorkflowStage.DOWNLOAD},
    WorkflowStage.DOWNLOAD: {WorkflowStage.CONVERT},
    WorkflowStage.CONVERT: {WorkflowStage.EXTRACT},
    WorkflowStage.EXTRACT: {WorkflowStage.ANALYZE},
    WorkflowStage.ANALYZE: {WorkflowStage.SYNTHESIZE},
    WorkflowStage.SYNTHESIZE: {WorkflowStage.ITERATE, WorkflowStage.REPORT},
    WorkflowStage.ITERATE: {WorkflowStage.PLAN},
    WorkflowStage.REPORT: set(),
}


class ExecutionRecord(BaseModel):
    """AER-pattern execution record capturing decision provenance.

    Each record documents: what was intended, what was observed, what was
    decided and why, and what was produced. This is append-only — records
    are never modified after creation (non-identifiability principle).
    """

    stage: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    intent: str
    observation: str
    inference: str
    artifacts_produced: list[str] = Field(default_factory=list)
    verification_result: str | None = None
    elapsed_seconds: float = 0.0
    iteration: int = 0
    metadata: dict = Field(default_factory=dict)


class ContextBudget(BaseModel):
    """Token budget tracking for context engineering (Tokalator pattern).

    Five-category decomposition tracks where tokens are spent.
    O(T²) cost growth without management — budget governance is essential.
    """

    max_tokens_per_sampling: int = Field(
        default=100_000,
        description="Max tokens per individual sampling call.",
    )
    max_tokens_per_paper: int = Field(
        default=50_000,
        description="Max tokens for paper content in a single call.",
    )
    total_budget: int = Field(
        default=2_000_000,
        description="Total token budget across all sampling calls.",
    )
    used: int = Field(default=0, description="Total tokens consumed so far.")
    system_prompt_tokens: int = 0
    paper_content_tokens: int = 0
    analysis_prompt_tokens: int = 0
    conversation_tokens: int = 0
    output_reserve_tokens: int = 4_000

    @property
    def remaining(self) -> int:
        """Tokens remaining in the total budget."""
        return max(0, self.total_budget - self.used)

    @property
    def budget_utilization(self) -> float:
        """Fraction of budget consumed (0.0–1.0)."""
        if self.total_budget == 0:
            return 1.0
        return min(1.0, self.used / self.total_budget)

    def can_afford(self, tokens: int) -> bool:
        """Check whether a sampling call of given size fits in budget."""
        return self.used + tokens <= self.total_budget


class WorkflowState(BaseModel):
    """Immutable-per-stage workflow state with full provenance.

    A new WorkflowState is created after each stage transition.
    The state is JSON-serializable for crash-recovery persistence.
    """

    run_id: str
    topic: str
    workspace: str = "./workspace"
    config_path: str = ""

    current_stage: str = WorkflowStage.PLAN
    stages: dict[str, str] = Field(default_factory=dict)
    execution_log: list[ExecutionRecord] = Field(default_factory=list)

    iteration: int = 0
    max_iterations: int = 3
    system_building: bool = False

    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    content_fingerprints: dict[str, str] = Field(default_factory=dict)

    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def model_post_init(self, __context: object) -> None:
        """Initialize stage statuses if empty."""
        if not self.stages:
            for stage in WorkflowStage:
                self.stages[stage.value] = StageStatus.PENDING.value

    def get_stage_status(self, stage: str) -> StageStatus:
        """Get the status of a specific stage."""
        return StageStatus(self.stages.get(stage, StageStatus.PENDING.value))

    def with_stage_status(self, stage: str, status: StageStatus) -> WorkflowState:
        """Return a new state with the given stage status updated."""
        new_stages = dict(self.stages)
        new_stages[stage] = status.value
        return self.model_copy(
            update={
                "stages": new_stages,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )

    def with_execution_record(self, record: ExecutionRecord) -> WorkflowState:
        """Return a new state with the record appended (append-only log)."""
        new_log = list(self.execution_log) + [record]
        return self.model_copy(
            update={
                "execution_log": new_log,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )

    def with_fingerprint(self, key: str, content: str) -> WorkflowState:
        """Return a new state with a content fingerprint recorded."""
        fp = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
        new_fps = dict(self.content_fingerprints)
        new_fps[key] = fp
        return self.model_copy(
            update={
                "content_fingerprints": new_fps,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )

    def with_budget_update(
        self, tokens_used: int, **category_deltas: int
    ) -> WorkflowState:
        """Return a new state with budget updated."""
        budget_data = self.context_budget.model_dump()
        budget_data["used"] = budget_data["used"] + tokens_used
        for cat, delta in category_deltas.items():
            if cat in budget_data:
                budget_data[cat] = budget_data[cat] + delta
        new_budget = ContextBudget(**budget_data)
        return self.model_copy(
            update={
                "context_budget": new_budget,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )


class GovernanceError(Exception):
    """Raised when a state transition violates governance rules."""


def validate_transition(state: WorkflowState, to_stage: str) -> None:
    """Validate that a stage transition is allowed.

    Enforces two governance rules:
    1. The current stage must be VERIFIED (verify-before-commit)
    2. The target stage must be in VALID_TRANSITIONS for current stage

    Raises GovernanceError if either rule is violated.
    """
    current = state.current_stage

    # The PLAN stage on iteration 0 doesn't need to be verified (initial entry)
    is_initial_entry = (
        current == WorkflowStage.PLAN
        and state.get_stage_status(current) == StageStatus.PENDING
        and to_stage == WorkflowStage.PLAN
    )

    if not is_initial_entry:
        current_status = state.get_stage_status(current)
        if current_status != StageStatus.VERIFIED:
            raise GovernanceError(
                f"Stage '{current}' has status '{current_status.value}', "
                f"must be '{StageStatus.VERIFIED.value}' before transitioning "
                f"to '{to_stage}'."
            )

        valid = VALID_TRANSITIONS.get(current, set())
        if to_stage not in valid:
            raise GovernanceError(
                f"Invalid transition: '{current}' → '{to_stage}'. "
                f"Valid targets: {sorted(valid) if valid else 'none (terminal stage)'}."
            )


def transition_state(state: WorkflowState, to_stage: str) -> WorkflowState:
    """Perform a validated state transition.

    Returns a new WorkflowState with the current_stage advanced.
    Raises GovernanceError if the transition is invalid.
    """
    validate_transition(state, to_stage)
    return state.model_copy(
        update={
            "current_stage": to_stage,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# Persistence (Recovery Layer)
# ---------------------------------------------------------------------------


def _workflow_state_path(workspace: str, run_id: str) -> Path:
    """Path to the persisted workflow state file."""
    return Path(workspace) / run_id / "workflow" / "state.json"


def save_state(state: WorkflowState) -> Path:
    """Persist workflow state to disk for crash-recovery.

    Creates the workflow directory if it doesn't exist.
    Returns the path to the saved file.
    """
    path = _workflow_state_path(state.workspace, state.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2))
    logger.debug("Workflow state saved: %s", path)
    return path


def load_state(workspace: str, run_id: str) -> WorkflowState | None:
    """Load a previously saved workflow state for crash-recovery.

    Returns None if no saved state exists.
    """
    path = _workflow_state_path(workspace, run_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return WorkflowState(**data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to load workflow state from %s: %s", path, exc)
        return None
