"""Unit tests for the Unified Horizon Metric (A3-5 gap closure)."""

from __future__ import annotations

import pytest

from research_pipeline.evaluation.horizon import (
    HorizonInputs,
    UnifiedHorizonResult,
    compute_unified_horizon_metric,
)


def _base(**overrides: object) -> HorizonInputs:
    defaults: dict[str, object] = {
        "normalized_score": 0.8,
        "difficulty": 0.5,
        "achieved_steps": 50,
        "target_steps": 50,
        "entropy_trend": 0.0,
        "reliability": 1.0,
    }
    defaults.update(overrides)
    return HorizonInputs(**defaults)  # type: ignore[arg-type]


def test_returns_result_in_unit_interval() -> None:
    result = compute_unified_horizon_metric(_base())
    assert isinstance(result, UnifiedHorizonResult)
    assert 0.0 <= result.uhm <= 1.0


def test_perfect_saturation() -> None:
    # score=1, difficulty=1, steps met, stable entropy, reliable → UHM very high.
    result = compute_unified_horizon_metric(_base(normalized_score=1.0, difficulty=1.0))
    assert result.uhm > 0.95


def test_zero_score_zeroes_metric() -> None:
    result = compute_unified_horizon_metric(_base(normalized_score=0.0))
    assert result.uhm == 0.0


def test_zero_achieved_steps_zeroes_horizon_efficiency() -> None:
    result = compute_unified_horizon_metric(_base(achieved_steps=0))
    assert result.horizon_efficiency == 0.0
    assert result.uhm == 0.0


def test_over_target_saturates_efficiency_at_one() -> None:
    result = compute_unified_horizon_metric(_base(achieved_steps=200, target_steps=50))
    assert result.horizon_efficiency == 1.0


def test_severe_entropy_decline_caps_stability() -> None:
    result = compute_unified_horizon_metric(_base(entropy_trend=-1.0))
    assert result.stability_factor == 0.5


def test_stability_interpolates() -> None:
    mid = compute_unified_horizon_metric(_base(entropy_trend=-0.25))
    assert 0.7 < mid.stability_factor < 0.8


def test_reliability_acts_as_multiplicative_gate() -> None:
    full = compute_unified_horizon_metric(_base(reliability=1.0))
    half = compute_unified_horizon_metric(_base(reliability=0.5))
    assert half.uhm == pytest.approx(full.uhm * 0.5)


def test_clamps_out_of_range_inputs() -> None:
    result = compute_unified_horizon_metric(
        _base(normalized_score=2.0, difficulty=-1.0, reliability=5.0)
    )
    assert 0.0 <= result.uhm <= 1.0
    assert 0.0 <= result.difficulty_weighted_score <= 1.0


def test_difficulty_boost_increases_score() -> None:
    easy = compute_unified_horizon_metric(_base(normalized_score=0.5, difficulty=0.0))
    hard = compute_unified_horizon_metric(_base(normalized_score=0.5, difficulty=1.0))
    assert hard.difficulty_weighted_score > easy.difficulty_weighted_score


def test_to_dict_is_json_compatible() -> None:
    import json

    result = compute_unified_horizon_metric(_base())
    payload = result.to_dict()
    # Must round-trip through JSON
    json.dumps(payload)
    assert set(payload) >= {
        "uhm",
        "difficulty_weighted_score",
        "horizon_efficiency",
        "stability_factor",
        "reliability",
        "components",
        "inputs",
    }


def test_geometric_mean_penalizes_weakest_component() -> None:
    # One weak component should cap overall more than one weak under arithmetic mean.
    result = compute_unified_horizon_metric(_base(achieved_steps=1, target_steps=10000))
    # horizon_eff ~ sqrt(1/10000) = 0.01 → geo mean collapses
    assert result.uhm < 0.3
