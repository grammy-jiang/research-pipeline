"""Security gates for pipeline ingestion boundaries.

A security gate runs at each stage boundary where external content enters:
- search: abstracts/titles from APIs
- download: PDF files
- convert: extracted markdown text
- extract: chunked content

Each gate: classify → sanitize if needed → record taint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from research_pipeline.infra.sanitize import sanitize_text
from research_pipeline.security.classifier import (
    ClassificationResult,
    classify_content,
)
from research_pipeline.security.taint import TaintLabel, TaintTracker, TrustLevel

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of passing content through a security gate."""

    content: str
    classification: ClassificationResult
    taint: TaintLabel
    passed: bool
    quarantined: bool = False
    sanitized: bool = False


class SecurityGate:
    """Boundary security gate: classify, sanitize, track.

    Usage::

        gate = SecurityGate(tracker)
        result = gate.check("abstract:2401.12345", text, "abstract", "search", "arxiv")
        if result.passed:
            use(result.content)
    """

    def __init__(
        self,
        tracker: TaintTracker | None = None,
        quarantine_high: bool = True,
    ) -> None:
        self._tracker = tracker if tracker is not None else TaintTracker()
        self._quarantine_high = quarantine_high
        self._stats: dict[str, int] = {
            "checked": 0,
            "passed": 0,
            "sanitized": 0,
            "quarantined": 0,
        }

    @property
    def tracker(self) -> TaintTracker:
        """Access the underlying taint tracker."""
        return self._tracker

    def check(
        self,
        key: str,
        content: str,
        content_type: str,
        stage: str,
        source: str,
    ) -> GateResult:
        """Run content through the security gate.

        Args:
            key: Content identifier (e.g., ``abstract:2401.12345``).
            content: Raw content to check.
            content_type: Type (abstract, title, markdown, pdf_text).
            stage: Pipeline stage.
            source: Content source (arxiv, scholar, pdf, etc.).

        Returns:
            GateResult with processed content and metadata.
        """
        self._stats["checked"] += 1

        # Step 1: Classify
        classification = classify_content(content, content_type)

        # Step 2: Determine trust level
        trust = TrustLevel.SEMI_TRUSTED
        if source in ("pdf", "markdown", "user"):
            trust = TrustLevel.UNTRUSTED

        # Step 3: Sanitize if needed
        output_content = content
        was_sanitized = False
        if classification.should_sanitize:
            output_content = sanitize_text(content)
            was_sanitized = True
            self._stats["sanitized"] += 1

        # Step 4: Quarantine if needed
        quarantined = False
        passed = True
        if classification.should_quarantine and self._quarantine_high:
            quarantined = True
            passed = False
            self._stats["quarantined"] += 1
            logger.warning(
                "Content quarantined: %s (flags: %s)",
                key,
                classification.flags,
            )
        else:
            self._stats["passed"] += 1

        # Step 5: Record taint
        taint = TaintLabel(
            source=source,
            stage=stage,
            trust_level=trust,
            sanitized=was_sanitized,
            classified=True,
            risk_flags=classification.flags,
        )
        self._tracker.mark(key, taint)

        return GateResult(
            content=output_content,
            classification=classification,
            taint=taint,
            passed=passed,
            quarantined=quarantined,
            sanitized=was_sanitized,
        )

    def stats(self) -> dict[str, int]:
        """Gate processing statistics."""
        return dict(self._stats)
