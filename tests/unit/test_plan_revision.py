"""Unit tests for :mod:`research_pipeline.pipeline.plan_revision`."""

from __future__ import annotations

from research_pipeline.pipeline.plan_revision import (
    PlanRevisionTracker,
    score_revision,
)


def test_identical_query_has_full_preservation_zero_coverage():
    s = score_revision("transformer time series", "transformer time series")
    assert s.preservation == 1.0
    assert s.coverage_delta == 0.0
    assert s.drift == 0.0


def test_completely_new_query_has_full_drift():
    s = score_revision("transformer time series", "bayesian filter kalman")
    assert s.preservation == 0.0
    assert s.coverage_delta >= 0.5
    assert s.drift == 1.0


def test_empty_revision_is_pathological():
    s = score_revision("something useful", "")
    assert s.composite == 0.0


def test_tracker_plateau_detection():
    t = PlanRevisionTracker(plateau_tolerance=0.02, plateau_window=2, min_iterations=2)
    # Three identical (no-op) revisions → identical composites → plateau fires.
    t.record("attention heads forecasting", "attention heads forecasting")
    t.record("attention heads forecasting", "attention heads forecasting")
    t.record("attention heads forecasting", "attention heads forecasting")
    assert t.should_stop() is True


def test_tracker_min_iterations_blocks_early_stop():
    t = PlanRevisionTracker(min_iterations=5, plateau_window=1)
    t.record("a b", "a b")
    t.record("a b", "a b")
    assert t.should_stop() is False


def test_best_returns_highest_composite():
    t = PlanRevisionTracker()
    t.record("apple", "apple pie recipe")  # decent preservation + coverage
    t.record("apple pie", "banana bread")  # low preservation
    best = t.best()
    assert best is not None
    # The preserved revision should outrank the drifting one on composite.
    assert best.iteration == 1


def test_composite_series_matches_revisions():
    t = PlanRevisionTracker()
    for original, revised in [
        ("x y z", "x y z w"),
        ("x y z w", "x y z w q"),
    ]:
        t.record(original, revised)
    series = t.composite_series()
    assert len(series) == 2
    assert all(0.0 <= v <= 1.0 for v in series)
