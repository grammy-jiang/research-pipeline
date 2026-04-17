"""Tests for LLM structured output enforcement (v0.13.22).

Covers:
- enforce() with default STANDARD_REQUIREMENTS
- enforce() in STRICT vs LENIENT mode
- Custom FieldRequirement definitions
- LLMOutputEnvelope.enforce_structure() integration
- LLMOutputEnvelope.to_flat_dict() helper
- Violation details, repair behaviour, min_items checks
"""

from __future__ import annotations

import pytest

from research_pipeline.llm.envelopes import LLMOutputEnvelope
from research_pipeline.llm.structured_output import (
    CITATION_REQUIREMENTS,
    CONFIDENCE_REQUIREMENTS,
    EVIDENCE_REQUIREMENTS,
    STANDARD_REQUIREMENTS,
    EnforcementMode,
    EnforcementResult,
    FieldRequirement,
    StructuredOutputError,
    Violation,
    enforce,
)

# ── enforce() basics ────────────────────────────────────────────────


class TestEnforceBasics:
    """Basic enforce() behaviour."""

    def test_valid_output_passes(self) -> None:
        output = {
            "evidence_refs": ["paper-1"],
            "confidence": 0.85,
            "source_ids": ["arxiv:2401.01234"],
        }
        result = enforce(output)
        assert result.valid is True
        assert result.violations == []

    def test_completely_empty_output_lenient(self) -> None:
        result = enforce({})
        assert result.valid is False
        assert len(result.violations) == 3
        # Repaired output should have defaults
        assert result.repaired["evidence_refs"] == []
        assert result.repaired["confidence"] == 0.0
        assert result.repaired["source_ids"] == []

    def test_completely_empty_output_strict(self) -> None:
        with pytest.raises(StructuredOutputError) as exc_info:
            enforce({}, mode=EnforcementMode.STRICT)
        assert len(exc_info.value.violations) == 3

    def test_partial_output_lenient(self) -> None:
        output = {"confidence": 0.9}
        result = enforce(output)
        assert result.valid is False
        assert len(result.violations) == 2
        # confidence kept, others repaired
        assert result.repaired["confidence"] == 0.9
        assert result.repaired["evidence_refs"] == []
        assert result.repaired["source_ids"] == []


# ── Type checking ───────────────────────────────────────────────────


class TestTypeChecking:
    """Type mismatch detection."""

    def test_wrong_type_detected(self) -> None:
        output = {
            "evidence_refs": "not-a-list",
            "confidence": "high",
            "source_ids": 42,
        }
        result = enforce(output)
        assert result.valid is False
        assert len(result.violations) == 3

    def test_wrong_type_repaired_lenient(self) -> None:
        output = {"evidence_refs": "not-a-list", "confidence": 0.5, "source_ids": []}
        result = enforce(output, mode=EnforcementMode.LENIENT)
        assert result.repaired["evidence_refs"] == []  # repaired to default

    def test_int_confidence_accepted(self) -> None:
        """int should be accepted for confidence (int | float)."""
        output = {"evidence_refs": ["x"], "confidence": 1, "source_ids": []}
        result = enforce(output)
        assert result.valid is True


# ── min_items ───────────────────────────────────────────────────────


class TestMinItems:
    """min_items enforcement for list fields."""

    def test_evidence_refs_empty_violates_min_items(self) -> None:
        output = {"evidence_refs": [], "confidence": 0.5, "source_ids": []}
        result = enforce(output)
        assert result.valid is False
        assert any(
            "at least 1" in v.message
            for v in result.violations
            if v.field == "evidence_refs"
        )

    def test_evidence_refs_one_item_passes(self) -> None:
        output = {"evidence_refs": ["ref1"], "confidence": 0.5, "source_ids": []}
        result = enforce(output)
        assert result.valid is True


# ── Custom requirements ─────────────────────────────────────────────


class TestCustomRequirements:
    """Custom FieldRequirement definitions."""

    def test_custom_field_missing(self) -> None:
        reqs = [FieldRequirement("my_field", str, default="unknown")]
        result = enforce({}, requirements=reqs)
        assert result.valid is False
        assert result.repaired["my_field"] == "unknown"

    def test_custom_field_present(self) -> None:
        reqs = [FieldRequirement("my_field", str, default="unknown")]
        result = enforce({"my_field": "hello"}, requirements=reqs)
        assert result.valid is True

    def test_custom_min_items(self) -> None:
        reqs = [FieldRequirement("tags", list, default=[], min_items=2)]
        result = enforce({"tags": ["one"]}, requirements=reqs)
        assert result.valid is False
        assert any("at least 2" in v.message for v in result.violations)

    def test_empty_requirements_always_valid(self) -> None:
        result = enforce({"anything": "goes"}, requirements=[])
        assert result.valid is True


# ── Preset lists ────────────────────────────────────────────────────


class TestPresets:
    """Preset requirement lists."""

    def test_evidence_only(self) -> None:
        result = enforce(
            {"evidence_refs": ["p1"]},
            requirements=EVIDENCE_REQUIREMENTS,
        )
        assert result.valid is True

    def test_confidence_only(self) -> None:
        result = enforce(
            {"confidence": 0.7},
            requirements=CONFIDENCE_REQUIREMENTS,
        )
        assert result.valid is True

    def test_citation_only(self) -> None:
        result = enforce(
            {"source_ids": ["doi:10.1234"]},
            requirements=CITATION_REQUIREMENTS,
        )
        assert result.valid is True

    def test_standard_is_all_three(self) -> None:
        assert len(STANDARD_REQUIREMENTS) == 3


# ── LLMOutputEnvelope integration ──────────────────────────────────


class TestEnvelopeIntegration:
    """LLMOutputEnvelope.enforce_structure() method."""

    def test_envelope_valid_decision(self) -> None:
        env = LLMOutputEnvelope(
            schema_id="test",
            decision={
                "evidence_refs": ["paper-1"],
                "confidence": 0.9,
                "source_ids": ["arxiv:2401.01234"],
            },
        )
        result = env.enforce_structure()
        assert result.valid is True

    def test_envelope_repair_decision(self) -> None:
        env = LLMOutputEnvelope(
            schema_id="test",
            decision={"answer": "yes"},
        )
        result = env.enforce_structure()
        assert result.valid is False
        # Decision dict should now have defaults injected
        assert env.decision["evidence_refs"] == []
        assert env.decision["confidence"] == 0.0
        assert env.decision["source_ids"] == []
        # Original field preserved
        assert env.decision["answer"] == "yes"

    def test_envelope_strict_raises(self) -> None:
        env = LLMOutputEnvelope(schema_id="test", decision={})
        with pytest.raises(StructuredOutputError):
            env.enforce_structure(mode=EnforcementMode.STRICT)

    def test_envelope_custom_requirements(self) -> None:
        reqs = [FieldRequirement("rationale", str, default="none")]
        env = LLMOutputEnvelope(schema_id="test", decision={})
        result = env.enforce_structure(requirements=reqs)
        assert result.valid is False
        assert env.decision["rationale"] == "none"


# ── to_flat_dict ────────────────────────────────────────────────────


class TestToFlatDict:
    """LLMOutputEnvelope.to_flat_dict() helper."""

    def test_flat_dict_merges_envelope_and_decision(self) -> None:
        env = LLMOutputEnvelope(
            schema_id="test",
            decision={"answer": "yes", "score": 0.9},
            evidence_refs=["p1"],
            notes=["note1"],
        )
        flat = env.to_flat_dict()
        assert flat["schema_id"] == "test"
        assert flat["evidence_refs"] == ["p1"]
        assert flat["notes"] == ["note1"]
        assert flat["answer"] == "yes"
        assert flat["score"] == 0.9
        assert flat["abstain"] is False

    def test_flat_dict_empty(self) -> None:
        env = LLMOutputEnvelope(schema_id="test")
        flat = env.to_flat_dict()
        assert flat["schema_id"] == "test"
        assert flat["evidence_refs"] == []


# ── Violation / StructuredOutputError ───────────────────────────────


class TestViolationAndError:
    """Violation dataclass and StructuredOutputError."""

    def test_violation_fields(self) -> None:
        v = Violation(field="foo", message="bar")
        assert v.field == "foo"
        assert v.message == "bar"

    def test_error_message_format(self) -> None:
        violations = [
            Violation("a", "missing"),
            Violation("b", "wrong type"),
        ]
        err = StructuredOutputError(violations)
        assert "a: missing" in str(err)
        assert "b: wrong type" in str(err)
        assert err.violations == violations


# ── EnforcementResult ───────────────────────────────────────────────


class TestEnforcementResult:
    """EnforcementResult dataclass."""

    def test_default_construction(self) -> None:
        r = EnforcementResult(valid=True)
        assert r.violations == []
        assert r.repaired == {}
