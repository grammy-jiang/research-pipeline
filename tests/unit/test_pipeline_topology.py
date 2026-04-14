"""Tests for pipeline topology: profile selection and stage routing."""

from __future__ import annotations

import pytest

from research_pipeline.pipeline.topology import (
    PROFILE_STAGES,
    SKIPPABLE_STAGES,
    PipelineProfile,
    classify_query_complexity,
    get_stages,
    profile_summary,
    should_run_stage,
)


class TestPipelineProfileEnum:
    """PipelineProfile enum values."""

    def test_quick_value(self) -> None:
        assert PipelineProfile.QUICK.value == "quick"

    def test_standard_value(self) -> None:
        assert PipelineProfile.STANDARD.value == "standard"

    def test_deep_value(self) -> None:
        assert PipelineProfile.DEEP.value == "deep"

    def test_all_profiles_are_strings(self) -> None:
        for p in PipelineProfile:
            assert isinstance(p.value, str)

    def test_construct_from_string(self) -> None:
        assert PipelineProfile("quick") is PipelineProfile.QUICK
        assert PipelineProfile("standard") is PipelineProfile.STANDARD
        assert PipelineProfile("deep") is PipelineProfile.DEEP

    def test_invalid_profile_raises(self) -> None:
        with pytest.raises(ValueError):
            PipelineProfile("invalid")


class TestGetStages:
    """get_stages() returns correct stages per profile."""

    def test_quick_stages(self) -> None:
        stages = get_stages(PipelineProfile.QUICK)
        assert stages == ["plan", "search", "screen", "summarize"]

    def test_standard_stages(self) -> None:
        stages = get_stages(PipelineProfile.STANDARD)
        assert stages == [
            "plan",
            "search",
            "screen",
            "download",
            "convert",
            "extract",
            "summarize",
        ]

    def test_deep_stages(self) -> None:
        stages = get_stages(PipelineProfile.DEEP)
        assert "plan" in stages
        assert "summarize" in stages
        assert "expand" in stages
        assert "quality" in stages
        assert "analyze_claims" in stages
        assert "score_claims" in stages

    def test_returns_new_list(self) -> None:
        """get_stages returns a copy, not the original."""
        a = get_stages(PipelineProfile.STANDARD)
        b = get_stages(PipelineProfile.STANDARD)
        assert a == b
        assert a is not b


class TestShouldRunStage:
    """should_run_stage() correctly gates stages."""

    def test_quick_runs_plan(self) -> None:
        assert should_run_stage(PipelineProfile.QUICK, "plan") is True

    def test_quick_runs_search(self) -> None:
        assert should_run_stage(PipelineProfile.QUICK, "search") is True

    def test_quick_runs_screen(self) -> None:
        assert should_run_stage(PipelineProfile.QUICK, "screen") is True

    def test_quick_runs_summarize(self) -> None:
        assert should_run_stage(PipelineProfile.QUICK, "summarize") is True

    def test_quick_skips_download(self) -> None:
        assert should_run_stage(PipelineProfile.QUICK, "download") is False

    def test_quick_skips_convert(self) -> None:
        assert should_run_stage(PipelineProfile.QUICK, "convert") is False

    def test_quick_skips_extract(self) -> None:
        assert should_run_stage(PipelineProfile.QUICK, "extract") is False

    def test_standard_runs_all_seven(self) -> None:
        for stage in [
            "plan",
            "search",
            "screen",
            "download",
            "convert",
            "extract",
            "summarize",
        ]:
            assert should_run_stage(PipelineProfile.STANDARD, stage) is True

    def test_standard_skips_expand(self) -> None:
        assert should_run_stage(PipelineProfile.STANDARD, "expand") is False

    def test_deep_runs_expand(self) -> None:
        assert should_run_stage(PipelineProfile.DEEP, "expand") is True

    def test_deep_runs_quality(self) -> None:
        assert should_run_stage(PipelineProfile.DEEP, "quality") is True

    def test_deep_runs_analyze_claims(self) -> None:
        assert should_run_stage(PipelineProfile.DEEP, "analyze_claims") is True

    def test_deep_runs_score_claims(self) -> None:
        assert should_run_stage(PipelineProfile.DEEP, "score_claims") is True

    def test_unknown_stage_returns_false(self) -> None:
        assert should_run_stage(PipelineProfile.STANDARD, "nonexistent") is False


class TestClassifyQueryComplexity:
    """classify_query_complexity() heuristic classification."""

    def test_short_query_is_quick(self) -> None:
        assert classify_query_complexity("transformers") == PipelineProfile.QUICK

    def test_two_word_query_is_quick(self) -> None:
        assert classify_query_complexity("attention mechanism") == PipelineProfile.QUICK

    def test_three_word_query_is_quick(self) -> None:
        assert (
            classify_query_complexity("graph neural networks") == PipelineProfile.QUICK
        )

    def test_what_is_query_is_quick(self) -> None:
        assert classify_query_complexity("what is RLHF") == PipelineProfile.QUICK

    def test_define_query_is_quick(self) -> None:
        assert (
            classify_query_complexity("define reinforcement learning")
            == PipelineProfile.QUICK
        )

    def test_overview_query_is_quick(self) -> None:
        assert classify_query_complexity("overview of GANs") == PipelineProfile.QUICK

    def test_normal_academic_topic_is_standard(self) -> None:
        result = classify_query_complexity(
            "transformer architectures for time series forecasting"
        )
        assert result == PipelineProfile.STANDARD

    def test_medium_query_is_standard(self) -> None:
        result = classify_query_complexity("local memory systems for AI agents")
        assert result == PipelineProfile.STANDARD

    def test_comprehensive_survey_is_deep(self) -> None:
        result = classify_query_complexity(
            "comprehensive survey of memory systems in large language models"
        )
        assert result == PipelineProfile.DEEP

    def test_comparison_is_deep(self) -> None:
        result = classify_query_complexity(
            "comparison of retrieval augmented generation approaches"
        )
        assert result == PipelineProfile.DEEP

    def test_versus_is_deep(self) -> None:
        result = classify_query_complexity(
            "dense retrieval versus sparse retrieval for question answering"
        )
        assert result == PipelineProfile.DEEP

    def test_state_of_the_art_is_deep(self) -> None:
        result = classify_query_complexity(
            "state-of-the-art methods in neural architecture search"
        )
        assert result == PipelineProfile.DEEP

    def test_taxonomy_is_deep(self) -> None:
        result = classify_query_complexity(
            "taxonomy of prompt engineering techniques for large language models"
        )
        assert result == PipelineProfile.DEEP

    def test_long_query_is_deep(self) -> None:
        result = classify_query_complexity(
            "how do different attention mechanisms compare in terms of efficiency "
            "and accuracy for long sequence modeling tasks"
        )
        assert result == PipelineProfile.DEEP


class TestProfileSummary:
    """profile_summary() returns descriptions."""

    def test_quick_summary_not_empty(self) -> None:
        assert len(profile_summary(PipelineProfile.QUICK)) > 0

    def test_standard_summary_not_empty(self) -> None:
        assert len(profile_summary(PipelineProfile.STANDARD)) > 0

    def test_deep_summary_not_empty(self) -> None:
        assert len(profile_summary(PipelineProfile.DEEP)) > 0

    def test_quick_mentions_abstract(self) -> None:
        assert "abstract" in profile_summary(PipelineProfile.QUICK).lower()

    def test_standard_mentions_full(self) -> None:
        assert "full" in profile_summary(PipelineProfile.STANDARD).lower()

    def test_deep_mentions_expansion(self) -> None:
        summary = profile_summary(PipelineProfile.DEEP).lower()
        assert "expansion" in summary or "citation" in summary


class TestProfileStagesConsistency:
    """PROFILE_STAGES data consistency checks."""

    def test_all_standard_stages_in_deep(self) -> None:
        """Every stage in STANDARD must also be in DEEP."""
        standard = set(PROFILE_STAGES[PipelineProfile.STANDARD])
        deep = set(PROFILE_STAGES[PipelineProfile.DEEP])
        assert standard.issubset(deep)

    def test_all_quick_stages_in_standard(self) -> None:
        """Every stage in QUICK must also be in STANDARD."""
        quick = set(PROFILE_STAGES[PipelineProfile.QUICK])
        standard = set(PROFILE_STAGES[PipelineProfile.STANDARD])
        assert quick.issubset(standard)

    def test_plan_in_all_profiles(self) -> None:
        for profile in PipelineProfile:
            assert "plan" in PROFILE_STAGES[profile]

    def test_summarize_in_all_profiles(self) -> None:
        for profile in PipelineProfile:
            assert "summarize" in PROFILE_STAGES[profile]

    def test_skippable_stages_not_in_quick(self) -> None:
        quick_stages = set(PROFILE_STAGES[PipelineProfile.QUICK])
        for stage in SKIPPABLE_STAGES:
            assert stage not in quick_stages

    def test_deep_has_extra_stages(self) -> None:
        standard = set(PROFILE_STAGES[PipelineProfile.STANDARD])
        deep = set(PROFILE_STAGES[PipelineProfile.DEEP])
        extras = deep - standard
        assert len(extras) > 0
        assert "expand" in extras
        assert "quality" in extras
