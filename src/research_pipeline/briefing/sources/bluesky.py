"""Bluesky source adapter (Phase F).

Reads AT Protocol ``app.bsky.feed.getAuthorFeed``-style JSON envelopes (or
public feed snapshots in the same shape). Disabled-by-default discussion-only
source. Auth tokens (when used live) are loaded by the calling layer and
passed through ``session``; the adapter itself is fixture-friendly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    IntelligenceEvent,
    SourcePolicy,
)
from research_pipeline.briefing.normalize import (
    canonicalize_url,
    content_hash_for,
    dedup_key_for,
    event_id_for,
    topic_id_for_title,
    utc_now_iso,
)
from research_pipeline.briefing.sources.base import (
    DEFAULT_HTTP_TIMEOUT,
    build_session,
    read_fixture_text,
)


class BlueskySource:
    """Poll a Bluesky AT Protocol feed JSON document."""

    def __init__(
        self,
        source: BriefingSourceConfig,
        *,
        fixture_base_dir: Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if source.access_method != AccessMethod.BLUESKY_API:
            raise ValueError(
                f"BlueskySource requires BLUESKY_API (got {source.access_method})"
            )
        self.source = source
        self.fixture_base_dir = fixture_base_dir
        self.session = session or build_session()

    def poll(self) -> list[IntelligenceEvent]:
        """Fetch the feed and return discussion-only events."""
        payload = self._load_payload()
        feed = payload.get("feed") if isinstance(payload, dict) else None
        if not isinstance(feed, list):
            return []
        events: list[IntelligenceEvent] = []
        for item in feed[: self.source.max_events_per_run]:
            if not isinstance(item, dict):
                continue
            post = item.get("post")
            if isinstance(post, dict):
                events.append(self._event_from_post(post))
        return events

    def _load_payload(self) -> Any:
        text = read_fixture_text(self.source, self.fixture_base_dir)
        if text is not None:
            return json.loads(text)
        url = str(self.source.api_url or "")
        if not url:
            raise ValueError("bluesky source requires api_url when no fixture")
        response = self.session.get(url, timeout=DEFAULT_HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def _event_from_post(self, post: dict[str, Any]) -> IntelligenceEvent:
        uri = str(post.get("uri") or "")
        cid = str(post.get("cid") or uri)
        record_field = post.get("record")
        record: dict[str, Any] = record_field if isinstance(record_field, dict) else {}
        text = str(record.get("text") or "")
        title = text.splitlines()[0][:160] if text else "Bluesky post"
        canonical_url = canonicalize_url(
            str(post.get("url") or f"https://bsky.app/profile/post/{cid}")
        )
        author_field = post.get("author")
        author: dict[str, Any] = author_field if isinstance(author_field, dict) else {}
        author_handle = str(author.get("handle") or author.get("did") or "")
        published_at = str(record.get("createdAt") or "") or None
        return IntelligenceEvent(
            event_id=event_id_for(self.source, canonical_url, cid),
            source_name=self.source.source_name,
            source_id=self.source.source_id,
            source_type=self.source.source_class,
            source_policy=SourcePolicy.DISCUSSION_ONLY,
            item_type="bluesky_post",
            canonical_url=canonical_url,
            title=title,
            retrieved_at=utc_now_iso(),
            collection_method=AccessMethod.BLUESKY_API,
            content_hash=content_hash_for(title, canonical_url, published_at, text),
            dedup_key=dedup_key_for(
                source=self.source,
                title=title,
                canonical_url=canonical_url,
                source_native_id=cid,
            ),
            author_or_org=author_handle or None,
            published_at=published_at,
            source_native_id=cid,
            summary_hint=text[:280],
            excerpt=text[:500],
            topics=(topic_id_for_title(title),),
            confidence="low",
            evidence_type="speculation_or_watch_item",
            raw_metadata={"uri": uri, "did": author.get("did")},
        )
