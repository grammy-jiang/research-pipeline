"""Phase F unit tests for the Hacker News source adapter."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
    SourcePolicy,
)
from research_pipeline.briefing.sources.hacker_news import HackerNewsSource

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "briefing" / "hn"


def _source(fixture_name: str) -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="hn-ai",
        source_name="Hacker News (AI)",
        source_class=SourceClass.TECHNICAL_DISCUSSION,
        access_method=AccessMethod.HACKER_NEWS,
        fixture_path=fixture_name,
        query="ai agent",
        api_url="https://hn.algolia.com/api/v1/search",
        enabled=False,
        max_events_per_run=10,
    )


class TestHackerNewsSourceFromFixture:
    def test_single_item(self) -> None:
        events = HackerNewsSource(
            _source("item.json"), fixture_base_dir=_FIXTURES
        ).poll()
        assert len(events) == 1
        e = events[0]
        assert e.collection_method == AccessMethod.HACKER_NEWS
        assert e.source_policy == SourcePolicy.DISCUSSION_ONLY
        assert e.item_type == "hacker_news_item"
        assert e.confidence == "low"
        assert e.evidence_type == "speculation_or_watch_item"
        assert e.source_native_id == "987"

    def test_thread_with_kids(self) -> None:
        events = HackerNewsSource(
            _source("thread.json"), fixture_base_dir=_FIXTURES
        ).poll()
        assert len(events) == 1
        assert events[0].title.startswith("Long-running")
        # discussion-only governance
        assert events[0].source_policy == SourcePolicy.DISCUSSION_ONLY

    def test_max_events_respected(self) -> None:
        cfg = _source("item.json").model_copy(update={"max_events_per_run": 0})
        events = HackerNewsSource(cfg, fixture_base_dir=_FIXTURES).poll()
        assert events == []

    def test_dedup_key_stable(self) -> None:
        a = HackerNewsSource(_source("item.json"), fixture_base_dir=_FIXTURES).poll()
        b = HackerNewsSource(_source("item.json"), fixture_base_dir=_FIXTURES).poll()
        assert a[0].dedup_key == b[0].dedup_key

    def test_disabled_by_default_governance(self) -> None:
        cfg = _source("item.json")
        assert cfg.enabled is False
