"""Interactive CLI human-in-the-loop gate (#124).

``CliGate`` reads stdin / writes stdout, so it is a presentation concern and
lives in ``cli/`` rather than the Core pipeline. The orchestrator receives it via
dependency injection (``run_pipeline(interactive_gate=...)``) and depends only on
the surface-agnostic ``GateCallback`` protocol in ``pipeline/gates.py``.
"""

from __future__ import annotations

import sys
from typing import IO

from research_pipeline.pipeline.gates import (
    GateContext,
    GateDecision,
    GateResult,
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
