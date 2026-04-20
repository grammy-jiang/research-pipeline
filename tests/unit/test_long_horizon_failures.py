"""Unit tests for UltraHorizon long-horizon failure-mode detection in
:mod:`research_pipeline.infra.failure_taxonomy`."""

from __future__ import annotations

from dataclasses import dataclass

from research_pipeline.infra.failure_taxonomy import (
    LONG_HORIZON_DESCRIPTIONS,
    FailureCategory,
    FailureTaxonomyLogger,
    LongHorizonFailureMode,
)


@dataclass
class _Score:
    composite: float
    preservation: float = 1.0


@dataclass
class _Rev:
    score: _Score | None


@dataclass
class _Entropy:
    alarm: bool


def test_enum_has_all_eight_modes():
    assert len(LongHorizonFailureMode) == 8
    assert set(LONG_HORIZON_DESCRIPTIONS.keys()) == set(LongHorizonFailureMode)


def test_failure_category_count_preserved():
    """Regression: extending the module must not add to FailureCategory."""
    assert len(FailureCategory) == 12


def test_premature_convergence_detected_from_low_composites(tmp_path):
    t = FailureTaxonomyLogger(tmp_path)
    revs = [_Rev(_Score(0.05)), _Rev(_Score(0.1)), _Rev(_Score(0.05))]
    modes = t.detect_long_horizon_modes(plan_revisions=revs)
    assert LongHorizonFailureMode.PREMATURE_CONVERGENCE in modes


def test_repetitive_looping_triggered_by_iteration_count(tmp_path):
    t = FailureTaxonomyLogger(tmp_path)
    revs = [_Rev(_Score(0.9)) for _ in range(10)]
    modes = t.detect_long_horizon_modes(plan_revisions=revs, max_plan_iterations=8)
    assert LongHorizonFailureMode.REPETITIVE_LOOPING in modes


def test_plan_drift_triggers_on_low_preservation(tmp_path):
    t = FailureTaxonomyLogger(tmp_path)
    revs = [_Rev(_Score(0.9, preservation=0.1))]
    modes = t.detect_long_horizon_modes(plan_revisions=revs)
    assert LongHorizonFailureMode.PLAN_DRIFT in modes


def test_context_lock_from_repeated_entropy_alarms(tmp_path):
    t = FailureTaxonomyLogger(tmp_path)
    readings = [_Entropy(True), _Entropy(False), _Entropy(True)]
    modes = t.detect_long_horizon_modes(entropy_readings=readings)
    assert LongHorizonFailureMode.CONTEXT_LOCK in modes


def test_tool_misuse_detected_from_schema_violations(tmp_path):
    t = FailureTaxonomyLogger(tmp_path)
    modes = t.detect_long_horizon_modes(tool_calls_outside_schema=1)
    assert LongHorizonFailureMode.TOOL_MISUSE in modes


def test_citation_fabrication_flagged(tmp_path):
    t = FailureTaxonomyLogger(tmp_path)
    modes = t.detect_long_horizon_modes(
        cited_ids={"paper_a", "paper_b"},
        retrieved_ids={"paper_a"},
    )
    assert LongHorizonFailureMode.CITATION_FABRICATION in modes


def test_gap_blindness_from_missing_sections(tmp_path):
    t = FailureTaxonomyLogger(tmp_path)
    modes = t.detect_long_horizon_modes(missing_sections=["Limitations"])
    assert LongHorizonFailureMode.GAP_BLINDNESS in modes


def test_memory_drift_from_drift_score(tmp_path):
    t = FailureTaxonomyLogger(tmp_path)
    modes = t.detect_long_horizon_modes(drift_score=0.8)
    assert LongHorizonFailureMode.MEMORY_DRIFT in modes


def test_clean_run_returns_empty(tmp_path):
    t = FailureTaxonomyLogger(tmp_path)
    revs = [_Rev(_Score(0.9, preservation=0.9)) for _ in range(2)]
    modes = t.detect_long_horizon_modes(
        plan_revisions=revs,
        entropy_readings=[_Entropy(False)],
        drift_score=0.1,
        cited_ids={"a"},
        retrieved_ids={"a"},
        missing_sections=[],
        tool_calls_outside_schema=0,
    )
    assert modes == []


def test_detection_logs_failure_records(tmp_path):
    t = FailureTaxonomyLogger(tmp_path)
    t.detect_long_horizon_modes(tool_calls_outside_schema=1)
    assert len(t.records) == 1
    assert t.records[0].subcategory == LongHorizonFailureMode.TOOL_MISUSE.value
