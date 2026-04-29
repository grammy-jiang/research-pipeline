from __future__ import annotations

import pytest
from pydantic import ValidationError

import research_pipeline.briefing as briefing
from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    IntelligenceEvent,
    ManualBriefingItem,
    SourceClass,
)


def test_briefing_package_exports_core_models() -> None:
    assert "IntelligenceEvent" in briefing.__all__
    assert "BriefingSourceConfig" in briefing.__all__
    assert "BriefingCluster" in briefing.__all__
    assert hasattr(briefing, "IntelligenceEvent")
    assert hasattr(briefing, "BriefingSourceConfig")


def test_intelligence_event_rejects_empty_title() -> None:
    with pytest.raises(ValidationError, match="title must not be empty"):
        IntelligenceEvent(
            event_id="evt_1",
            source_name="GitHub",
            source_id="github.releases",
            source_type=SourceClass.PRIMARY_ARTIFACT,
            item_type="release",
            canonical_url="https://example.com/release/1",
            title="   ",
            retrieved_at="2026-04-29T00:00:00Z",
            collection_method=AccessMethod.GITHUB_RELEASES,
            content_hash="hash_1",
            dedup_key="dedup_1",
        )


def test_github_releases_source_requires_repo_or_url_or_fixture() -> None:
    with pytest.raises(
        ValidationError,
        match="github_releases sources require repo, api_url, or fixture_path",
    ):
        BriefingSourceConfig(
            source_id="github.releases",
            source_name="GitHub Releases",
            source_class=SourceClass.PRIMARY_ARTIFACT,
            access_method=AccessMethod.GITHUB_RELEASES,
        )


def test_rss_atom_source_requires_feed_or_fixture() -> None:
    with pytest.raises(
        ValidationError,
        match="rss_atom sources require feed_url or fixture_path",
    ):
        BriefingSourceConfig(
            source_id="rss.feed",
            source_name="RSS Feed",
            source_class=SourceClass.TECHNICAL_DISCUSSION,
            access_method=AccessMethod.RSS_ATOM,
        )


def test_manual_source_requires_manual_items() -> None:
    with pytest.raises(
        ValidationError,
        match="manual sources require at least one manual item",
    ):
        BriefingSourceConfig(
            source_id="manual.curated",
            source_name="Curated",
            source_class=SourceClass.TECHNICAL_DISCUSSION,
            access_method=AccessMethod.MANUAL,
        )


def test_manual_source_accepts_non_empty_manual_items() -> None:
    config = BriefingSourceConfig(
        source_id="manual.curated",
        source_name="Curated",
        source_class=SourceClass.TECHNICAL_DISCUSSION,
        access_method=AccessMethod.MANUAL,
        manual_items=(
            ManualBriefingItem(
                title="Useful update",
                url="https://example.com/update",
                source_native_id="m-1",
            ),
        ),
    )

    assert len(config.manual_items) == 1
