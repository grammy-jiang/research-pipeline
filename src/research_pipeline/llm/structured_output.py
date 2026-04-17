"""Structured output enforcement for LLM responses.

Validates that LLM outputs conform to required schemas with evidence
citations, confidence levels, and source references.  Provides both
strict (raise on violation) and lenient (repair with defaults) modes.

References:
    Deep-research report Theme 9 (Structured Output Protocols) and
    Theme 13 (Evidence-Based Reporting Requirements).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EnforcementMode(str, Enum):
    """How to handle schema violations."""

    STRICT = "strict"
    LENIENT = "lenient"


@dataclass(frozen=True)
class FieldRequirement:
    """Specification for a required output field.

    Attributes:
        name: Field name in the output dict.
        expected_type: Python type (or tuple of types) the value must be.
        default: Default value to inject in lenient mode when missing.
        min_items: For list fields, minimum number of items.
    """

    name: str
    expected_type: type | tuple[type, ...] = object
    default: Any = None
    min_items: int = 0


@dataclass
class Violation:
    """A single schema violation."""

    field: str
    message: str


@dataclass
class EnforcementResult:
    """Result of structured-output enforcement.

    Attributes:
        valid: Whether the output passed all checks.
        violations: List of violations found.
        repaired: The (possibly repaired) output dict.
    """

    valid: bool
    violations: list[Violation] = field(default_factory=list)
    repaired: dict[str, Any] = field(default_factory=dict)


# ── Common requirement presets ────────────────────────────────────────

EVIDENCE_REQUIREMENTS: list[FieldRequirement] = [
    FieldRequirement("evidence_refs", list, default=[], min_items=1),
]

CONFIDENCE_REQUIREMENTS: list[FieldRequirement] = [
    FieldRequirement("confidence", (int, float), default=0.0),
]

CITATION_REQUIREMENTS: list[FieldRequirement] = [
    FieldRequirement("source_ids", list, default=[]),
]

STANDARD_REQUIREMENTS: list[FieldRequirement] = [
    *EVIDENCE_REQUIREMENTS,
    *CONFIDENCE_REQUIREMENTS,
    *CITATION_REQUIREMENTS,
]


def enforce(
    output: dict[str, Any],
    requirements: list[FieldRequirement] | None = None,
    mode: EnforcementMode = EnforcementMode.LENIENT,
) -> EnforcementResult:
    """Validate and optionally repair an LLM output dict.

    Args:
        output: The raw LLM output dictionary.
        requirements: Field requirements to enforce.  Defaults to
            ``STANDARD_REQUIREMENTS``.
        mode: ``STRICT`` raises on first violation; ``LENIENT``
            repairs missing/invalid fields with defaults.

    Returns:
        An ``EnforcementResult`` with validity flag, violations, and
        the (possibly repaired) output.

    Raises:
        StructuredOutputError: In strict mode, when any violation is found.
    """
    if requirements is None:
        requirements = STANDARD_REQUIREMENTS

    violations: list[Violation] = []
    repaired = dict(output)

    for req in requirements:
        _check_field(req, repaired, violations, mode)

    valid = len(violations) == 0

    if violations:
        logger.info(
            "Structured-output enforcement: %d violation(s) in %s mode",
            len(violations),
            mode.value,
        )
        for v in violations:
            logger.debug("  %s: %s", v.field, v.message)

    if not valid and mode == EnforcementMode.STRICT:
        raise StructuredOutputError(violations)

    return EnforcementResult(valid=valid, violations=violations, repaired=repaired)


def _check_field(
    req: FieldRequirement,
    output: dict[str, Any],
    violations: list[Violation],
    mode: EnforcementMode,
) -> None:
    """Check a single field requirement and optionally repair."""
    if req.name not in output:
        violations.append(Violation(req.name, f"Missing required field '{req.name}'"))
        if mode == EnforcementMode.LENIENT:
            output[req.name] = req.default
        return

    value = output[req.name]

    if not isinstance(value, req.expected_type):
        violations.append(
            Violation(
                req.name,
                f"Expected type {req.expected_type}, got {type(value).__name__}",
            )
        )
        if mode == EnforcementMode.LENIENT:
            output[req.name] = req.default
        return

    if isinstance(value, list) and len(value) < req.min_items:
        violations.append(
            Violation(
                req.name,
                f"Expected at least {req.min_items} item(s), got {len(value)}",
            )
        )


class StructuredOutputError(Exception):
    """Raised in strict mode when output violates requirements."""

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        messages = [f"{v.field}: {v.message}" for v in violations]
        super().__init__(f"Structured output violations: {'; '.join(messages)}")
