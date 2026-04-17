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
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FailureCategory(str, Enum):
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


class FailureSeverity(str, Enum):
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
