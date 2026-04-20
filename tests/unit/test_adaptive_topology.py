"""Unit tests for the adaptive-topology extension in :mod:`pipeline.topology`."""

from __future__ import annotations

from research_pipeline.pipeline.topology import (
    PipelineProfile,
    classify_query_complexity,
    select_topology_by_difficulty,
)


def test_easy_topic_maps_to_quick():
    profile, trace = select_topology_by_difficulty("cats")
    assert profile is PipelineProfile.QUICK
    assert trace["profile"] == "quick"
    assert trace["difficulty"] in {"trivial", "easy"}


def test_expert_topic_maps_to_deep():
    topic = (
        "prove sufficient and necessary conditions under which the causal "
        "correlation between hierarchical Bayesian inference and multi-step "
        "reinforcement-learning policy gradients preserves statistical "
        "significance given heavy-tailed reward distributions"
    )
    profile, trace = select_topology_by_difficulty(topic)
    assert profile is PipelineProfile.DEEP
    assert trace["difficulty"] in {"hard", "expert"}


def test_trace_contains_expected_keys():
    profile, trace = select_topology_by_difficulty(
        "hypothesis testing", trace_features=True
    )
    assert isinstance(profile, PipelineProfile)
    for key in ("profile", "difficulty", "routing_target", "raw_score", "features"):
        assert key in trace
    assert isinstance(trace["features"], dict)


def test_existing_classify_query_complexity_still_works():
    """Regression: ensure the original API is untouched."""
    assert classify_query_complexity("literature review on transformers") in (
        PipelineProfile.QUICK,
        PipelineProfile.STANDARD,
        PipelineProfile.DEEP,
    )
