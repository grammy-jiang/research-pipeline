"""Failure taxonomy for structured pipeline error classification.

Categorizes pipeline failures into a predefined taxonomy so that
error patterns can be aggregated, queried, and acted upon.  Each
failure is tagged with a :class:`FailureCategory` and optional
sub-category, stage, and severity.

References:
    Deep-research report Theme 8 (Failure-Mode Cataloguing) and
    Engineering Gap 4 (Error Taxonomy).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FailureCategory(StrEnum):
    """Top-level failure categories."""

    RETRIEVAL_MISS = "retrieval_miss"
    CONVERSION_ERROR = "conversion_error"
    SYNTHESIS_GAP = "synthesis_gap"
    SCREENING_MISS = "screening_miss"
    DOWNLOAD_FAILURE = "download_failure"
    EXTRACTION_ERROR = "extraction_error"
    LLM_ERROR = "llm_error"
    RATE_LIMIT = "rate_limit"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    CONFIG_ERROR = "config_error"
    UNKNOWN = "unknown"


class LongHorizonFailureMode(StrEnum):
    """UltraHorizon (Paper 29) long-horizon agent failure modes.

    These are orthogonal to :class:`FailureCategory` (which is about single
    events) — long-horizon modes describe patterns across many iterations
    of a TER loop or multi-stage synthesis trajectory.
    """

    PREMATURE_CONVERGENCE = "premature_convergence"
    REPETITIVE_LOOPING = "repetitive_looping"
    PLAN_DRIFT = "plan_drift"
    CONTEXT_LOCK = "context_lock"
    TOOL_MISUSE = "tool_misuse"
    CITATION_FABRICATION = "citation_fabrication"
    GAP_BLINDNESS = "gap_blindness"
    MEMORY_DRIFT = "memory_drift"


LONG_HORIZON_DESCRIPTIONS: dict[LongHorizonFailureMode, str] = {
    LongHorizonFailureMode.PREMATURE_CONVERGENCE: (
        "Synthesis terminates after too few iterations, leaving known gaps."
    ),
    LongHorizonFailureMode.REPETITIVE_LOOPING: (
        "TER loop revises to queries that semantically repeat earlier ones."
    ),
    LongHorizonFailureMode.PLAN_DRIFT: (
        "Revised plan loses the original research question's core terms."
    ),
    LongHorizonFailureMode.CONTEXT_LOCK: (
        "LLM output entropy collapses; responses become template-like."
    ),
    LongHorizonFailureMode.TOOL_MISUSE: (
        "MCP tools called with arguments outside their declared schema."
    ),
    LongHorizonFailureMode.CITATION_FABRICATION: (
        "Claims cite papers that were never retrieved."
    ),
    LongHorizonFailureMode.GAP_BLINDNESS: (
        "Synthesis omits a required section even though evidence exists."
    ),
    LongHorizonFailureMode.MEMORY_DRIFT: (
        "Cross-run semantic memory state diverges from its last checkpoint."
    ),
}


class FailureSeverity(StrEnum):
    """Severity levels for failures."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailureRecord:
    """A single classified failure event.

    Attributes:
        category: Top-level failure category.
        subcategory: Optional finer classification.
        stage: Pipeline stage where the failure occurred.
        severity: How severe the failure is.
        message: Human-readable description.
        details: Arbitrary additional data.
        timestamp: ISO-8601 timestamp of the failure.
        paper_id: Associated paper identifier, if applicable.
    """

    category: FailureCategory
    subcategory: str = ""
    stage: str = ""
    severity: FailureSeverity = FailureSeverity.MEDIUM
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat(),
    )
    paper_id: str = ""


class FailureTaxonomyLogger:
    """Structured failure logger that writes JSONL records.

    Failures are appended to ``<run_root>/logs/failures.jsonl``.
    Provides helper methods for common failure types and a
    ``summary()`` method that aggregates counts by category.

    Args:
        run_root: Root directory of the pipeline run.
        enabled: When ``False`` all calls are no-ops.
    """

    def __init__(self, run_root: Path, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._log_path = run_root / "logs" / "failures.jsonl"
        self._records: list[FailureRecord] = []

        if enabled:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def records(self) -> list[FailureRecord]:
        """Return all recorded failures."""
        return list(self._records)

    def log(self, record: FailureRecord) -> None:
        """Log a failure record.

        Args:
            record: The failure record to log.
        """
        if not self._enabled:
            return

        self._records.append(record)
        self._append_jsonl(record)

        logger.warning(
            "Pipeline failure [%s/%s] in %s: %s",
            record.category.value,
            record.subcategory or "-",
            record.stage or "unknown",
            record.message,
        )

    def log_failure(
        self,
        category: FailureCategory,
        message: str,
        *,
        subcategory: str = "",
        stage: str = "",
        severity: FailureSeverity = FailureSeverity.MEDIUM,
        paper_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> FailureRecord:
        """Convenience method to log a failure with keyword arguments.

        Args:
            category: Top-level failure category.
            message: Human-readable description.
            subcategory: Optional finer classification.
            stage: Pipeline stage where the failure occurred.
            severity: How severe the failure is.
            paper_id: Associated paper identifier.
            details: Arbitrary additional data.

        Returns:
            The created :class:`FailureRecord`.
        """
        record = FailureRecord(
            category=category,
            subcategory=subcategory,
            stage=stage,
            severity=severity,
            message=message,
            details=details or {},
            paper_id=paper_id,
        )
        self.log(record)
        return record

    # ── Stage-specific helpers ────────────────────────────────────

    def retrieval_miss(
        self,
        message: str,
        *,
        stage: str = "search",
        paper_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> FailureRecord:
        """Log a retrieval miss (expected paper not found)."""
        return self.log_failure(
            FailureCategory.RETRIEVAL_MISS,
            message,
            stage=stage,
            severity=FailureSeverity.MEDIUM,
            paper_id=paper_id,
            details=details,
        )

    def conversion_error(
        self,
        message: str,
        *,
        paper_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> FailureRecord:
        """Log a PDF conversion failure."""
        return self.log_failure(
            FailureCategory.CONVERSION_ERROR,
            message,
            stage="convert",
            severity=FailureSeverity.HIGH,
            paper_id=paper_id,
            details=details,
        )

    def download_failure(
        self,
        message: str,
        *,
        paper_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> FailureRecord:
        """Log a download failure."""
        return self.log_failure(
            FailureCategory.DOWNLOAD_FAILURE,
            message,
            stage="download",
            severity=FailureSeverity.HIGH,
            paper_id=paper_id,
            details=details,
        )

    def screening_miss(
        self,
        message: str,
        *,
        paper_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> FailureRecord:
        """Log a screening miss (relevant paper excluded)."""
        return self.log_failure(
            FailureCategory.SCREENING_MISS,
            message,
            stage="screen",
            severity=FailureSeverity.MEDIUM,
            paper_id=paper_id,
            details=details,
        )

    def synthesis_gap(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> FailureRecord:
        """Log a synthesis gap (topic not covered in report)."""
        return self.log_failure(
            FailureCategory.SYNTHESIS_GAP,
            message,
            stage="summarize",
            severity=FailureSeverity.MEDIUM,
            details=details,
        )

    # ── Aggregation ───────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Aggregate failure counts by category and severity.

        Returns:
            Dict with ``total``, ``by_category``, ``by_severity``,
            and ``by_stage`` breakdowns.
        """
        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_stage: dict[str, int] = {}

        for rec in self._records:
            cat = rec.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            sev = rec.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

            stg = rec.stage or "unknown"
            by_stage[stg] = by_stage.get(stg, 0) + 1

        return {
            "total": len(self._records),
            "by_category": by_category,
            "by_severity": by_severity,
            "by_stage": by_stage,
        }

    # ── Persistence ───────────────────────────────────────────────

    def _append_jsonl(self, record: FailureRecord) -> None:
        """Append a record to the JSONL log file."""
        data = asdict(record)
        data["category"] = record.category.value
        data["severity"] = record.severity.value
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, default=str) + "\n")

    # ── Long-horizon mode detection ───────────────────────────────

    def detect_long_horizon_modes(
        self,
        *,
        plan_revisions: list[object] | None = None,
        entropy_readings: list[object] | None = None,
        drift_score: float | None = None,
        cited_ids: set[str] | None = None,
        retrieved_ids: set[str] | None = None,
        missing_sections: list[str] | None = None,
        tool_calls_outside_schema: int = 0,
        max_plan_iterations: int = 8,
    ) -> list[LongHorizonFailureMode]:
        """Detect which UltraHorizon failure modes are present in a run.

        All arguments are optional; only the signals that are supplied are
        evaluated. Each detected mode is also logged as a HIGH-severity
        :class:`FailureRecord` so downstream pipeline code can react.
        """
        modes: list[LongHorizonFailureMode] = []

        if plan_revisions is not None:
            revs = list(plan_revisions)
            if len(revs) >= max_plan_iterations:
                modes.append(LongHorizonFailureMode.REPETITIVE_LOOPING)
            composites: list[float] = []
            for r in revs:
                score = getattr(r, "score", None)
                if score is not None:
                    composites.append(float(getattr(score, "composite", 0.0)))
            if composites and all(c < 0.2 for c in composites[-3:]):
                modes.append(LongHorizonFailureMode.PREMATURE_CONVERGENCE)
            preservations = [
                float(getattr(getattr(r, "score", None), "preservation", 1.0))
                for r in revs
                if getattr(r, "score", None) is not None
            ]
            if preservations and min(preservations) < 0.3:
                modes.append(LongHorizonFailureMode.PLAN_DRIFT)

        if entropy_readings is not None:
            alarms = sum(1 for r in entropy_readings if getattr(r, "alarm", False))
            if alarms >= 2:
                modes.append(LongHorizonFailureMode.CONTEXT_LOCK)

        if tool_calls_outside_schema > 0:
            modes.append(LongHorizonFailureMode.TOOL_MISUSE)

        if cited_ids is not None and retrieved_ids is not None:
            fabricated = cited_ids - retrieved_ids
            if fabricated:
                modes.append(LongHorizonFailureMode.CITATION_FABRICATION)

        if missing_sections:
            modes.append(LongHorizonFailureMode.GAP_BLINDNESS)

        if drift_score is not None and drift_score > 0.5:
            modes.append(LongHorizonFailureMode.MEMORY_DRIFT)

        for mode in modes:
            self.log_failure(
                FailureCategory.VALIDATION_ERROR,
                LONG_HORIZON_DESCRIPTIONS[mode],
                subcategory=mode.value,
                severity=FailureSeverity.HIGH,
            )
        return modes

    def load_from_disk(self) -> list[FailureRecord]:
        """Reload records from the JSONL log file.

        Returns:
            List of :class:`FailureRecord` instances loaded from disk.
        """
        if not self._log_path.exists():
            return []

        loaded: list[FailureRecord] = []
        for line in self._log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            rec = FailureRecord(
                category=FailureCategory(data["category"]),
                subcategory=data.get("subcategory", ""),
                stage=data.get("stage", ""),
                severity=FailureSeverity(data.get("severity", "medium")),
                message=data.get("message", ""),
                details=data.get("details", {}),
                timestamp=data.get("timestamp", ""),
                paper_id=data.get("paper_id", ""),
            )
            loaded.append(rec)
        return loaded
