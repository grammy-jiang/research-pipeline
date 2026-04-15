"""Content classifiers for pipeline ingestion boundaries.

Classifies content risk level at each stage boundary:
- CLEAN: no suspicious patterns detected
- LOW: minor patterns (common false positives)
- MEDIUM: suspicious patterns requiring sanitization
- HIGH: likely injection/attack, should be quarantined
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Content risk classification level."""

    CLEAN = "clean"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ClassificationResult:
    """Result of content classification."""

    risk_level: RiskLevel
    flags: list[str]
    content_type: str
    recommended_action: str

    @property
    def should_sanitize(self) -> bool:
        """True if content should be sanitized before use."""
        return self.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)

    @property
    def should_quarantine(self) -> bool:
        """True if content should be blocked entirely."""
        return self.risk_level == RiskLevel.HIGH


# Detection patterns with risk weights
_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel]] = [
    (
        "prompt_injection_xml",
        re.compile(
            r"</?(?:tool_call|system|assistant|function_call|invoke)[^>]*>", re.I
        ),
        RiskLevel.HIGH,
    ),
    (
        "prompt_injection_marker",
        re.compile(r"(?:SYSTEM|INSTRUCTION|OVERRIDE)\s*:", re.I),
        RiskLevel.HIGH,
    ),
    (
        "template_injection",
        re.compile(r"\{\{.*?\}\}|\{%.*?%\}", re.DOTALL),
        RiskLevel.MEDIUM,
    ),
    (
        "data_uri",
        re.compile(r"data:\w+/\w+;base64,", re.I),
        RiskLevel.MEDIUM,
    ),
    (
        "executable_code",
        re.compile(r"```(?:python|javascript|bash|sh|exec)\b", re.I),
        RiskLevel.LOW,
    ),
    (
        "excessive_backticks",
        re.compile(r"`{4,}"),
        RiskLevel.LOW,
    ),
    (
        "json_function_call",
        re.compile(r'"function_call"\s*:\s*\{', re.I),
        RiskLevel.MEDIUM,
    ),
    (
        "base64_blob",
        re.compile(r"[A-Za-z0-9+/]{100,}={0,2}"),
        RiskLevel.LOW,
    ),
    (
        "null_bytes",
        re.compile(r"\x00"),
        RiskLevel.HIGH,
    ),
    (
        "unicode_override",
        re.compile(r"[\u202a-\u202e\u2066-\u2069]"),
        RiskLevel.MEDIUM,
    ),
]


def classify_content(text: str, content_type: str = "text") -> ClassificationResult:
    """Classify content risk level.

    Args:
        text: Content to classify.
        content_type: Type of content for context.

    Returns:
        ClassificationResult with risk assessment.
    """
    if not text:
        return ClassificationResult(
            risk_level=RiskLevel.CLEAN,
            flags=[],
            content_type=content_type,
            recommended_action="pass",
        )

    flags: list[str] = []
    max_risk = RiskLevel.CLEAN

    for name, pattern, risk in _PATTERNS:
        if pattern.search(text):
            flags.append(name)
            if _risk_order(risk) > _risk_order(max_risk):
                max_risk = risk

    # Length-based risk
    if len(text) > 100_000:
        flags.append("excessive_length")
        if _risk_order(RiskLevel.LOW) > _risk_order(max_risk):
            max_risk = RiskLevel.LOW

    action = {
        RiskLevel.CLEAN: "pass",
        RiskLevel.LOW: "pass",
        RiskLevel.MEDIUM: "sanitize",
        RiskLevel.HIGH: "quarantine",
    }[max_risk]

    if flags:
        logger.debug("Content classified as %s: %s", max_risk.value, flags)

    return ClassificationResult(
        risk_level=max_risk,
        flags=flags,
        content_type=content_type,
        recommended_action=action,
    )


def _risk_order(level: RiskLevel) -> int:
    """Numeric ordering for risk levels."""
    return {"clean": 0, "low": 1, "medium": 2, "high": 3}[level.value]
