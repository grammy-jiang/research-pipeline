"""Unit tests for the Reddit Phase F adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
    SourcePolicy,
)
from research_pipeline.briefing.sources.reddit import RedditSource

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "briefing" / "reddit"


def _source() -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="reddit-ml",
        source_name="r/MachineLearning",
        source_class=SourceClass.SOCIAL_SIGNAL,
        access_method=AccessMethod.REDDIT_API,
        fixture_path="listing.json",
        api_url="https://www.reddit.com/r/MachineLearning.json",
        enabled=False,
        max_events_per_run=10,
    )


class TestRedditSource:
    def test_listing_parsed(self) -> None:
        events = RedditSource(_source(), fixture_base_dir=_FIXTURES).poll()
        assert len(events) == 2
        e = events[0]
        assert e.collection_method == AccessMethod.REDDIT_API
        assert e.source_policy == SourcePolicy.DISCUSSION_ONLY
        assert e.confidence == "low"
        assert e.source_native_id == "abc123"
        assert e.raw_metadata["subreddit"] == "MachineLearning"

    def test_unsupported_access_method_rejected(self) -> None:
        bad = _source().model_copy(
            update={
                "access_method": AccessMethod.RSS_ATOM,
                "feed_url": "https://example.com/feed.xml",
                "fixture_path": None,
            }
        )
        with pytest.raises(ValueError, match="REDDIT_API"):
            RedditSource(bad)

    def test_disabled_by_default(self) -> None:
        assert _source().enabled is False
