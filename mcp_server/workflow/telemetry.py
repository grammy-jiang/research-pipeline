"""Three-surface telemetry for workflow observability.

Implements the AgentTrace three-surface pattern:
- Cognitive surface: sampling prompts/responses, analysis decisions
- Operational surface: stage execution times, artifact counts, errors
- Contextual surface: token budgets, iteration state, user decisions

All telemetry is emitted via MCP logging (ctx.info/warning/error) AND
written to a JSONL file for post-hoc analysis (AER dual-path pattern).

Principle: "Instrument before you govern" — telemetry is the foundation
that all other harness layers depend on.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)


class TelemetrySurface(str, Enum):
    """Three telemetry surfaces from the AgentTrace taxonomy."""

    COGNITIVE = "cognitive"
    OPERATIONAL = "operational"
    CONTEXTUAL = "contextual"


class TelemetryEvent(dict):
    """A structured telemetry event.

    Inherits from dict for JSON serialization while providing
    a typed constructor.
    """

    def __init__(
        self,
        surface: str,
        event_type: str,
        stage: str,
        message: str,
        *,
        iteration: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            surface=surface,
            event_type=event_type,
            stage=stage,
            message=message,
            timestamp=datetime.now(UTC).isoformat(),
            iteration=iteration,
            metadata=metadata or {},
        )


class WorkflowTelemetry:
    """Three-surface telemetry emitter for the workflow engine.

    Emits events via MCP logging (if Context available) and writes
    to a JSONL file for post-hoc analysis.
    """

    def __init__(
        self,
        workspace: str,
        run_id: str,
        ctx: Context | None = None,
    ) -> None:
        self._workspace = workspace
        self._run_id = run_id
        self._ctx = ctx
        self._events: list[dict] = []
        self._log_path = Path(workspace) / run_id / "workflow" / "telemetry.jsonl"

    @property
    def events(self) -> list[dict]:
        """All emitted events (in-memory copy)."""
        return list(self._events)

    def set_context(self, ctx: Context | None) -> None:
        """Update the MCP context (may change between tool calls)."""
        self._ctx = ctx

    # -- Cognitive surface --------------------------------------------------

    def log_sampling_request(
        self,
        stage: str,
        prompt_summary: str,
        token_estimate: int,
        *,
        iteration: int = 0,
    ) -> None:
        """Log an outgoing sampling request."""
        self._emit(
            TelemetryEvent(
                surface=TelemetrySurface.COGNITIVE,
                event_type="sampling_request",
                stage=stage,
                message=f"Sampling request: {prompt_summary}",
                iteration=iteration,
                metadata={"token_estimate": token_estimate},
            )
        )

    def log_sampling_response(
        self,
        stage: str,
        response_summary: str,
        tokens_used: int,
        *,
        iteration: int = 0,
    ) -> None:
        """Log a received sampling response."""
        self._emit(
            TelemetryEvent(
                surface=TelemetrySurface.COGNITIVE,
                event_type="sampling_response",
                stage=stage,
                message=f"Sampling response: {response_summary}",
                iteration=iteration,
                metadata={"tokens_used": tokens_used},
            )
        )

    def log_analysis_decision(
        self,
        stage: str,
        decision: str,
        rationale: str,
        *,
        iteration: int = 0,
    ) -> None:
        """Log an analysis decision with rationale (provenance)."""
        self._emit(
            TelemetryEvent(
                surface=TelemetrySurface.COGNITIVE,
                event_type="analysis_decision",
                stage=stage,
                message=f"Decision: {decision}",
                iteration=iteration,
                metadata={"rationale": rationale},
            )
        )

    # -- Operational surface ------------------------------------------------

    def log_stage_start(
        self,
        stage: str,
        intent: str,
        *,
        iteration: int = 0,
    ) -> None:
        """Log the start of a pipeline stage."""
        self._emit(
            TelemetryEvent(
                surface=TelemetrySurface.OPERATIONAL,
                event_type="stage_start",
                stage=stage,
                message=f"Stage started: {intent}",
                iteration=iteration,
            )
        )

    def log_stage_complete(
        self,
        stage: str,
        elapsed_seconds: float,
        artifact_count: int,
        *,
        iteration: int = 0,
    ) -> None:
        """Log the completion of a pipeline stage."""
        self._emit(
            TelemetryEvent(
                surface=TelemetrySurface.OPERATIONAL,
                event_type="stage_complete",
                stage=stage,
                message=f"Stage completed in {elapsed_seconds:.1f}s",
                iteration=iteration,
                metadata={
                    "elapsed_seconds": elapsed_seconds,
                    "artifact_count": artifact_count,
                },
            )
        )

    def log_stage_failed(
        self,
        stage: str,
        error: str,
        *,
        iteration: int = 0,
    ) -> None:
        """Log a stage failure."""
        self._emit(
            TelemetryEvent(
                surface=TelemetrySurface.OPERATIONAL,
                event_type="stage_failed",
                stage=stage,
                message=f"Stage failed: {error}",
                iteration=iteration,
                metadata={"error": error},
            ),
            level="error",
        )

    def log_verification_result(
        self,
        stage: str,
        passed: bool,
        details: str,
        *,
        iteration: int = 0,
    ) -> None:
        """Log a stage verification result."""
        status = "PASSED" if passed else "FAILED"
        self._emit(
            TelemetryEvent(
                surface=TelemetrySurface.OPERATIONAL,
                event_type="verification",
                stage=stage,
                message=f"Verification {status}: {details}",
                iteration=iteration,
                metadata={"passed": passed, "details": details},
            ),
            level="info" if passed else "warning",
        )

    # -- Contextual surface -------------------------------------------------

    def log_budget_update(
        self,
        stage: str,
        tokens_used: int,
        budget_remaining: int,
        utilization: float,
        *,
        iteration: int = 0,
    ) -> None:
        """Log a token budget update."""
        level = "info"
        if utilization > 0.9:
            level = "error"
        elif utilization > 0.75:
            level = "warning"

        self._emit(
            TelemetryEvent(
                surface=TelemetrySurface.CONTEXTUAL,
                event_type="budget_update",
                stage=stage,
                message=(
                    f"Budget: {tokens_used:,} tokens used, "
                    f"{budget_remaining:,} remaining ({utilization:.0%})"
                ),
                iteration=iteration,
                metadata={
                    "tokens_used": tokens_used,
                    "budget_remaining": budget_remaining,
                    "utilization": utilization,
                },
            ),
            level=level,
        )

    def log_iteration_state(
        self,
        iteration: int,
        max_iterations: int,
        papers_found: int,
        gaps_remaining: int,
    ) -> None:
        """Log iteration progress for drift monitoring."""
        self._emit(
            TelemetryEvent(
                surface=TelemetrySurface.CONTEXTUAL,
                event_type="iteration_state",
                stage="iterate",
                message=(
                    f"Iteration {iteration}/{max_iterations}: "
                    f"{papers_found} papers, {gaps_remaining} gaps"
                ),
                iteration=iteration,
                metadata={
                    "max_iterations": max_iterations,
                    "papers_found": papers_found,
                    "gaps_remaining": gaps_remaining,
                },
            )
        )

    def log_user_decision(
        self,
        stage: str,
        decision: str,
        *,
        iteration: int = 0,
    ) -> None:
        """Log a user decision from elicitation (provenance)."""
        self._emit(
            TelemetryEvent(
                surface=TelemetrySurface.CONTEXTUAL,
                event_type="user_decision",
                stage=stage,
                message=f"User decision: {decision}",
                iteration=iteration,
            )
        )

    def log_doom_loop_check(
        self,
        stage: str,
        is_loop: bool,
        similarity: float,
        *,
        iteration: int = 0,
    ) -> None:
        """Log a doom-loop detection check."""
        self._emit(
            TelemetryEvent(
                surface=TelemetrySurface.CONTEXTUAL,
                event_type="doom_loop_check",
                stage=stage,
                message=(
                    f"Doom-loop {'DETECTED' if is_loop else 'clear'}: "
                    f"similarity={similarity:.2f}"
                ),
                iteration=iteration,
                metadata={"is_loop": is_loop, "similarity": similarity},
            ),
            level="warning" if is_loop else "info",
        )

    # -- Internal -----------------------------------------------------------

    def _emit(self, event: TelemetryEvent, level: str = "info") -> None:
        """Emit an event via MCP logging + JSONL file + Python logger."""
        self._events.append(dict(event))

        # Python logging
        log_fn = getattr(logger, level, logger.info)
        log_fn("[%s/%s] %s", event["surface"], event["event_type"], event["message"])

        # MCP logging (if context available)
        if self._ctx is not None:
            with contextlib.suppress(Exception):
                ctx_fn = getattr(self._ctx, level, self._ctx.info)
                ctx_fn(event["message"])

        # JSONL persistence (AER dual-path)
        with contextlib.suppress(Exception):
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a") as f:
                f.write(json.dumps(dict(event)) + "\n")

    def flush(self) -> Path:
        """Return the path to the telemetry log file."""
        return self._log_path
