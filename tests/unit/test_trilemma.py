"""Unit tests for :mod:`research_pipeline.security.trilemma`."""

from __future__ import annotations

from research_pipeline.security.trilemma import TrilemmaMonitor


def test_single_stage_no_alarm_under_budget():
    m = TrilemmaMonitor(depth_budget=10.0)
    m.record("s1", "alpha beta gamma", "alpha beta gamma delta")
    assert m.alarm is False
    assert m.cumulative_lipschitz > 0.0


def test_cumulative_product_across_stages():
    m = TrilemmaMonitor(depth_budget=1000.0)
    # Each stage doubles the token count → Lipschitz proxy ≥ 2.
    m.record("s1", "a b", "a b a b")  # 2→4
    m.record("s2", "a b a b", "a b a b a b a b")  # 4→8
    m.record("s3", "a b a b a b a b", "a b " * 8)  # 8→16
    assert m.cumulative_lipschitz >= 8.0
    assert len(m.stages) == 3


def test_budget_exceeded_raises_alarm():
    m = TrilemmaMonitor(depth_budget=3.0)
    # Large growth at every stage: 1→4 tokens → K ≈ 4 per stage
    m.record("s1", "a", "a b c d")
    assert m.alarm is True


def test_reset_clears_stages():
    m = TrilemmaMonitor(depth_budget=1.5)
    m.record("s1", "a", "a b c d")
    m.reset()
    assert m.stages == []
    assert m.alarm is False


def test_sanitization_delta_feeds_lipschitz():
    m = TrilemmaMonitor(depth_budget=100.0)
    r = m.record("s1", "alpha", "alpha", sanitization_delta=0.9)
    # lipschitz_proxy should honour 1 + |sanitization_delta| ≥ 1.9
    assert r.lipschitz_proxy >= 1.9


def test_summary_serialisable():
    m = TrilemmaMonitor(depth_budget=5.0)
    m.record("s1", "a", "a b")
    s = m.summary()
    assert set(s.keys()) == {
        "stage_count",
        "cumulative_lipschitz",
        "budget",
        "alarm",
    }
    assert s["stage_count"] == 1
    assert s["budget"] == 5.0
