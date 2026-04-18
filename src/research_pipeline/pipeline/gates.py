"""Human-in-the-loop approval gates between pipeline stages.

Provides configurable checkpoints where a human can review intermediate
results before the pipeline proceeds. Supports CLI interactive mode,
auto-approve mode, and a callback protocol for MCP/UI integration.

References:
    Deep research Pattern 3: Human gates necessary — 4 papers recommend
    human checkpoints over fully autonomous operation.
    Flowr (2604.05987): Human approval gates at critical junctions.
"""

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import IO, Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Recommended gate positions (stage pairs where review is most valuable)
DEFAULT_GATE_STAGES = ["screen", "download", "summarize"]


class GateDecision(StrEnum):
    """Possible outcomes from a human gate check."""

    APPROVE = "approve"
    REJECT = "reject"
    SKIP = "skip"


@dataclass(frozen=True)
class GateContext:
    """Information passed to a gate for human review.

    Attributes:
        completed_stage: The stage that just finished.
        next_stage: The stage about to start (or None if last).
        run_id: Current pipeline run identifier.
        run_root: Path to the run directory.
        stage_summary: Brief summary of what the completed stage produced.
        artifact_counts: Key metrics (e.g., {"papers_screened": 42}).
    """

    completed_stage: str
    next_stage: str | None
    run_id: str
    run_root: Path
    stage_summary: str = ""
    artifact_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class GateResult:
    """Result from a gate evaluation.

    Attributes:
        decision: The human's decision (approve/reject/skip).
        reason: Optional reason text provided by the reviewer.
        reviewer: Identifier for who made the decision.
    """

    decision: GateDecision
    reason: str = ""
    reviewer: str = "unknown"


class GateRejectedError(Exception):
    """Raised when a human gate rejects pipeline continuation."""

    def __init__(self, stage: str, reason: str = "") -> None:
        self.stage = stage
        self.reason = reason
        msg = f"Pipeline rejected at gate after '{stage}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


@runtime_checkable
class GateCallback(Protocol):
    """Protocol for gate implementations.

    Implementors decide how to present the gate to the user
    and collect their decision.
    """

    def check(self, context: GateContext) -> GateResult:
        """Present the gate and return the decision.

        Args:
            context: Information about the completed stage.

        Returns:
            GateResult with the decision.
        """
        ...  # pragma: no cover


class AutoApproveGate:
    """Gate that automatically approves all stages.

    Used when running in non-interactive mode or with --auto-approve.
    """

    def check(self, context: GateContext) -> GateResult:
        """Auto-approve the gate.

        Args:
            context: Gate context (logged but not reviewed).

        Returns:
            Approved gate result.
        """
        logger.info(
            "Auto-approving gate after '%s' (next: %s)",
            context.completed_stage,
            context.next_stage or "end",
        )
        return GateResult(
            decision=GateDecision.APPROVE,
            reason="auto-approved",
            reviewer="auto",
        )


class CliGate:
    """Interactive CLI gate using stdin/stdout.

    Presents a summary of the completed stage and prompts the user
    to approve, reject, or skip.
    """

    def __init__(self, input_stream: IO[str] | None = None) -> None:
        """Initialize CLI gate.

        Args:
            input_stream: Input stream for testing (defaults to sys.stdin).
        """
        self._input = input_stream or sys.stdin

    def _format_summary(self, context: GateContext) -> str:
        """Format a human-readable summary for the gate prompt."""
        lines = [
            "",
            "=" * 60,
            f"  GATE: Review after '{context.completed_stage}'",
            "=" * 60,
            f"  Run ID: {context.run_id}",
        ]
        if context.stage_summary:
            lines.append(f"  Summary: {context.stage_summary}")
        if context.artifact_counts:
            counts_str = ", ".join(
                f"{k}={v}" for k, v in context.artifact_counts.items()
            )
            lines.append(f"  Metrics: {counts_str}")
        if context.next_stage:
            lines.append(f"  Next stage: {context.next_stage}")
        else:
            lines.append("  This is the final stage.")
        lines.extend(
            [
                "",
                "  [a]pprove  — continue to next stage",
                "  [r]eject   — stop pipeline",
                "  [s]kip     — skip remaining gates",
                "",
            ]
        )
        return "\n".join(lines)

    def check(self, context: GateContext) -> GateResult:
        """Prompt the user and return their decision.

        Args:
            context: Information about the completed stage.

        Returns:
            GateResult with the user's decision.
        """
        summary = self._format_summary(context)
        print(summary, flush=True)

        while True:
            try:
                raw = self._input.readline()
            except (EOFError, KeyboardInterrupt):
                return GateResult(
                    decision=GateDecision.REJECT,
                    reason="interrupted",
                    reviewer="cli_user",
                )

            # readline() returns "" at EOF (no trailing newline)
            if raw == "":
                return GateResult(
                    decision=GateDecision.REJECT,
                    reason="end of input",
                    reviewer="cli_user",
                )

            choice = raw.strip().lower()

            if choice in ("a", "approve", "y", "yes", ""):
                return GateResult(
                    decision=GateDecision.APPROVE,
                    reason="",
                    reviewer="cli_user",
                )
            if choice in ("r", "reject", "n", "no"):
                return GateResult(
                    decision=GateDecision.REJECT,
                    reason="user rejected",
                    reviewer="cli_user",
                )
            if choice in ("s", "skip"):
                return GateResult(
                    decision=GateDecision.SKIP,
                    reason="user skipped remaining gates",
                    reviewer="cli_user",
                )
            print(
                "  Invalid choice. Enter [a]pprove, [r]eject, or [s]kip: ",
                end="",
                flush=True,
            )


class CallbackGate:
    """Gate that delegates to a user-provided callback function.

    Useful for MCP server or UI integration where the decision
    comes from an external system.
    """

    def __init__(
        self,
        callback: Callable[[Any], GateResult],
    ) -> None:
        """Initialize callback gate.

        Args:
            callback: Function that takes GateContext and returns GateResult.
        """
        self._callback = callback

    def check(self, context: GateContext) -> GateResult:
        """Delegate to the callback.

        Args:
            context: Information about the completed stage.

        Returns:
            GateResult from the callback.
        """
        return self._callback(context)


def build_stage_summary(
    stage: str,
    run_root: Path,
) -> tuple[str, dict[str, int]]:
    """Build a human-readable summary and counts for a completed stage.

    Args:
        stage: Name of the completed stage.
        run_root: Path to the run directory.

    Returns:
        Tuple of (summary_text, artifact_counts).
    """
    summary = ""
    counts: dict[str, int] = {}

    if stage == "plan":
        plan_path = run_root / "plan" / "query_plan.json"
        if plan_path.exists():
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                n_queries = len(plan.get("query_variants", []))
                summary = f"Generated {n_queries} query variants"
                counts["query_variants"] = n_queries
            except (json.JSONDecodeError, KeyError):
                summary = "Plan generated"

    elif stage == "search":
        search_dir = run_root / "search"
        if search_dir.exists():
            jsonl_files = list(search_dir.glob("*.jsonl"))
            total = sum(
                sum(1 for _ in f.open(encoding="utf-8"))
                for f in jsonl_files
                if f.exists()
            )
            summary = f"Found {total} candidate papers"
            counts["candidates"] = total

    elif stage == "screen":
        shortlist = run_root / "screen" / "shortlist.jsonl"
        if shortlist.exists():
            n = sum(1 for _ in shortlist.open(encoding="utf-8"))
            summary = f"Shortlisted {n} papers for download"
            counts["shortlisted"] = n

    elif stage == "download":
        pdf_dir = run_root / "download" / "pdf"
        if pdf_dir.exists():
            n = len(list(pdf_dir.glob("*.pdf")))
            summary = f"Downloaded {n} PDFs"
            counts["pdfs"] = n

    elif stage == "convert":
        md_dir = run_root / "convert" / "markdown"
        if md_dir.exists():
            n = len(list(md_dir.glob("*.md")))
            summary = f"Converted {n} papers to Markdown"
            counts["markdown_files"] = n

    elif stage == "extract":
        extract_dir = run_root / "extract"
        if extract_dir.exists():
            n = len(list(extract_dir.glob("*.json")))
            summary = f"Extracted chunks from {n} papers"
            counts["extracted_papers"] = n

    elif stage == "summarize":
        synth = run_root / "summarize" / "synthesis.json"
        if synth.exists():
            summary = "Synthesis report generated"
            counts["synthesis"] = 1

    if not summary:
        summary = f"Stage '{stage}' completed"

    return summary, counts


def check_gate(
    gate: GateCallback,
    completed_stage: str,
    next_stage: str | None,
    run_id: str,
    run_root: Path,
    gate_stages: list[str] | None = None,
) -> GateDecision:
    """Check a gate after a stage, if the stage is in the gate list.

    Args:
        gate: Gate implementation to use.
        completed_stage: Stage that just finished.
        next_stage: Stage about to start (or None).
        run_id: Run identifier.
        run_root: Path to run directory.
        gate_stages: List of stages after which to check (None = defaults).

    Returns:
        The gate decision.

    Raises:
        GateRejectedError: If the gate rejects continuation.
    """
    stages = gate_stages if gate_stages is not None else DEFAULT_GATE_STAGES

    if completed_stage not in stages:
        return GateDecision.APPROVE

    summary_text, artifact_counts = build_stage_summary(completed_stage, run_root)

    context = GateContext(
        completed_stage=completed_stage,
        next_stage=next_stage,
        run_id=run_id,
        run_root=run_root,
        stage_summary=summary_text,
        artifact_counts=artifact_counts,
    )

    result = gate.check(context)

    logger.info(
        "Gate after '%s': %s (reviewer=%s, reason=%s)",
        completed_stage,
        result.decision.value,
        result.reviewer,
        result.reason or "none",
    )

    if result.decision == GateDecision.REJECT:
        raise GateRejectedError(completed_stage, result.reason)

    return result.decision
