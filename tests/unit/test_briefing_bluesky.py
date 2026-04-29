"""Unit tests for the Bluesky Phase F adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
    SourcePolicy,
)
from research_pipeline.briefing.sources.bluesky import BlueskySource

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "briefing" / "bluesky"


def _source() -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="bluesky-ai",
        source_name="Bluesky AI feed",
        source_class=SourceClass.SOCIAL_SIGNAL,
        access_method=AccessMethod.BLUESKY_API,
        fixture_path="feed.json",
        api_url="https://public.api.bsky.app/xrpc/app.bsky.feed.getFeed",
        enabled=False,
        max_events_per_run=10,
    )


class TestBlueskySource:
    def test_feed_parsed(self) -> None:
        events = BlueskySource(_source(), fixture_base_dir=_FIXTURES).poll()
        assert len(events) == 2
        e = events[0]
        assert e.collection_method == AccessMethod.BLUESKY_API
        assert e.source_policy == SourcePolicy.DISCUSSION_ONLY
        assert e.author_or_org == "example.bsky.social"
        assert e.source_native_id == "bafycid111"

    def test_unsupported_access_method_rejected(self) -> None:
        bad = _source().model_copy(
            update={
                "access_method": AccessMethod.RSS_ATOM,
                "feed_url": "https://example.com/feed.xml",
                "fixture_path": None,
            }
        )
        with pytest.raises(ValueError, match="BLUESKY_API"):
            BlueskySource(bad)

    def test_disabled_by_default(self) -> None:
        assert _source().enabled is False
