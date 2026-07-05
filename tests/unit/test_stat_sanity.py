"""Tests for statistic range sanity checks (#33)."""

from __future__ import annotations

from research_pipeline.quality.stat_sanity import find_stat_anomalies


def test_flags_impossible_correlation() -> None:
    anomalies = find_stat_anomalies({"spearman": -1.8})
    assert anomalies
    assert "correlation" in anomalies[0]
    assert "-1.8" in anomalies[0]


def test_valid_correlation_ok() -> None:
    assert find_stat_anomalies({"pearson_r": 0.9}) == []
    assert find_stat_anomalies({"kendall_tau": -1.0}) == []  # boundary inclusive


def test_flags_probability_over_one() -> None:
    assert find_stat_anomalies({"probability": 1.5})
    assert find_stat_anomalies({"p_value": -0.1})


def test_flags_bleu_over_100() -> None:
    assert find_stat_anomalies({"bleu": 120})
    assert find_stat_anomalies({"bleu_score": 55}) == []


def test_percentage_bounds() -> None:
    assert find_stat_anomalies({"percent_correct": 150})
    assert find_stat_anomalies({"percentage": 50}) == []


def test_nested_and_lists() -> None:
    data = {"results": [{"metrics": {"spearman_rho": -2.0}}]}
    anomalies = find_stat_anomalies(data)
    assert any("-2.0" in a for a in anomalies)


def test_booleans_are_not_statistics() -> None:
    assert find_stat_anomalies({"probability_high": True}) == []


def test_no_false_positive_on_unknown_keys() -> None:
    assert find_stat_anomalies({"temperature": 999, "count": -5, "year": 2026}) == []
