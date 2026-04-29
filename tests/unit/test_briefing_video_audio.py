"""Unit tests for the video/audio Phase F adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
)
from research_pipeline.briefing.sources.video_audio import VideoAudioSource

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "briefing" / "video_audio"


def _source() -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="yt-example",
        source_name="Example YT Channel",
        source_class=SourceClass.VIDEO_AUDIO,
        access_method=AccessMethod.VIDEO_AUDIO,
        fixture_path="youtube_feed.xml",
        feed_url="https://www.youtube.com/feeds/videos.xml?channel_id=example",
        enabled=False,
        max_events_per_run=10,
    )


class TestVideoAudioSource:
    def test_youtube_atom_parsed(self) -> None:
        events = VideoAudioSource(_source(), fixture_base_dir=_FIXTURES).poll()
        assert len(events) == 2
        first = events[0]
        assert first.collection_method == AccessMethod.VIDEO_AUDIO
        assert first.source_type == SourceClass.VIDEO_AUDIO
        assert first.item_type == "video"
        assert first.source_native_id == "abc123XYZ"
        assert first.author_or_org == "Example AI Channel"
        assert first.confidence == "medium"

    def test_unsupported_access_method_rejected(self) -> None:
        bad = _source().model_copy(
            update={
                "access_method": AccessMethod.RSS_ATOM,
                "fixture_path": None,
            }
        )
        with pytest.raises(ValueError, match="VIDEO_AUDIO"):
            VideoAudioSource(bad)

    def test_disabled_by_default(self) -> None:
        assert _source().enabled is False
