"""Unit tests for Phase F source-expansion governance helpers."""

from __future__ import annotations

import pytest

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
)
from research_pipeline.briefing.registry import SourceRegistry
from research_pipeline.briefing.source_evaluation import (
    ReportComparisonResult,
    assert_disabled_by_default,
    compare_reports,
    evaluate_registry,
    evaluate_source_for_enablement,
    summarize_source_mix,
)


def _phase_a_source(**overrides: object) -> BriefingSourceConfig:
    base = {
        "source_id": "anthropic-releases",
        "source_name": "Anthropic Releases",
        "source_class": SourceClass.PRIMARY_ARTIFACT,
        "access_method": AccessMethod.GITHUB_RELEASES,
        "fixture_path": "github_releases.json",
        "retention_policy": "metadata_only",
        "rate_limit_policy": "polite",
        "cadence": "daily",
        "enabled": True,
    }
    base.update(overrides)
    return BriefingSourceConfig(**base)  # type: ignore[arg-type]


def _phase_f_social(**overrides: object) -> BriefingSourceConfig:
    base = {
        "source_id": "reddit-localllama",
        "source_name": "Reddit r/LocalLLaMA",
        "source_class": SourceClass.SOCIAL_SIGNAL,
        "access_method": AccessMethod.REDDIT_API,
        "fixture_path": "reddit/localllama.json",
        "retention_policy": "metadata_only",
        "rate_limit_policy": "polite",
        "cadence": "daily",
        "enabled": False,
        "auth_required": True,
    }
    base.update(overrides)
    return BriefingSourceConfig(**base)  # type: ignore[arg-type]


class TestEvaluateSourceForEnablement:
    def test_phase_a_source_passes(self) -> None:
        result = evaluate_source_for_enablement(_phase_a_source())
        assert result.passed
        assert result.reasons == ()

    def test_missing_fixture_path_flagged(self) -> None:
        result = evaluate_source_for_enablement(
            _phase_a_source(
                access_method=AccessMethod.RSS_ATOM,
                fixture_path=None,
                feed_url="https://example.com/feed.xml",
            )
        )
        assert not result.sanctioned
        assert any("fixture" in r for r in result.reasons)

    def test_missing_rate_limit_flagged(self) -> None:
        result = evaluate_source_for_enablement(_phase_a_source(rate_limit_policy=""))
        assert not result.sanctioned
        assert any("rate_limit_policy" in r for r in result.reasons)

    def test_phase_f_social_disabled_by_default_passes(self) -> None:
        result = evaluate_source_for_enablement(_phase_f_social())
        assert result.passed

    def test_phase_f_social_enabled_without_review_fails(self) -> None:
        result = evaluate_source_for_enablement(
            _phase_f_social(enabled=True, last_reviewed_at=None)
        )
        assert not result.enabled_safely
        assert any("disabled by default" in r for r in result.reasons)

    def test_phase_f_social_with_review_can_be_enabled(self) -> None:
        result = evaluate_source_for_enablement(
            _phase_f_social(enabled=True, last_reviewed_at="2026-05-01")
        )
        assert result.passed

    def test_x_api_requires_explicit_review(self) -> None:
        source = BriefingSourceConfig(
            source_id="x-anthropic",
            source_name="X / Anthropic",
            source_class=SourceClass.SOCIAL_SIGNAL,
            access_method=AccessMethod.X_API,
            fixture_path="x/timeline.json",
            api_url="https://api.x.com/2/users/by/username/anthropic",
            auth_required=True,
            enabled=True,
            last_reviewed_at=None,
        )
        result = evaluate_source_for_enablement(source)
        assert not result.enabled_safely
        assert any("X/Twitter" in r for r in result.reasons)


class TestAssertDisabledByDefault:
    def test_passes_for_disabled_phase_f_source(self) -> None:
        assert_disabled_by_default(_phase_f_social(enabled=False))

    def test_passes_for_phase_a_enabled_source(self) -> None:
        assert_disabled_by_default(_phase_a_source(enabled=True))

    def test_passes_when_review_present(self) -> None:
        assert_disabled_by_default(
            _phase_f_social(enabled=True, last_reviewed_at="2026-05-01")
        )

    def test_fails_when_phase_f_enabled_without_review(self) -> None:
        with pytest.raises(ValueError, match="disabled by default"):
            assert_disabled_by_default(
                _phase_f_social(enabled=True, last_reviewed_at=None)
            )


class TestCompareReports:
    def test_identical_reports_no_change(self) -> None:
        md = (
            "## Top Items\n\n### 1. Foo\n[a](https://example.com/a)\n"
            "### 2. Bar\n[b](https://example.com/b)\n"
        )
        result = compare_reports(md, md)
        assert isinstance(result, ReportComparisonResult)
        assert result.item_count_delta == 0
        assert result.link_count_delta == 0
        assert not result.coverage_increase
        assert not result.noise_increase

    def test_coverage_increase_detected(self) -> None:
        baseline = (
            "## Top Items\n\n### 1. Foo\n[a](https://example.com/a)\n"
            "### 2. Bar\n[b](https://example.com/b)\n"
        )
        candidate = baseline + "### 3. Baz\n[c](https://example.com/c)\n"
        result = compare_reports(baseline, candidate)
        assert result.coverage_increase
        assert "https://example.com/c" in result.new_links
        assert not result.noise_increase

    def test_noise_increase_flagged_when_growth_excessive(self) -> None:
        baseline = "## Top Items\n\n### 1. Foo\n[a](https://example.com/a)\n"
        # 5 new top items + 5 new links → ratio 5x
        extras = "".join(
            f"### {i}. Item{i}\n[l{i}](https://example.com/l{i})\n" for i in range(2, 7)
        )
        candidate = baseline + extras
        result = compare_reports(baseline, candidate)
        assert result.noise_increase
        assert not result.coverage_increase

    def test_removed_links_tracked(self) -> None:
        baseline = (
            "## Top Items\n\n### 1. Foo\n[a](https://example.com/a)\n"
            "### 2. Bar\n[b](https://example.com/b)\n"
        )
        candidate = "## Top Items\n\n### 1. Foo\n[a](https://example.com/a)\n"
        result = compare_reports(baseline, candidate)
        assert "https://example.com/b" in result.removed_links


class TestRegistryHelpers:
    def test_evaluate_registry_returns_one_per_source(self) -> None:
        registry = SourceRegistry(
            sources=(_phase_a_source(), _phase_f_social()),
            watchlist_terms=(),
        )
        results = evaluate_registry(registry)
        assert len(results) == 2
        assert {r.source_id for r in results} == {
            "anthropic-releases",
            "reddit-localllama",
        }

    def test_summarize_source_mix(self) -> None:
        registry = SourceRegistry(
            sources=(
                _phase_a_source(),
                _phase_f_social(enabled=True, last_reviewed_at="2026-05-01"),
            ),
            watchlist_terms=(),
        )
        mix = summarize_source_mix(registry)
        assert mix == {"primary_artifact": 1, "social_signal": 1}
