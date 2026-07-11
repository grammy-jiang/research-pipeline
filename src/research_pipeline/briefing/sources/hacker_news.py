"""Hacker News source adapter for Phase F source expansion."""

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


class HackerNewsSource:
    """Poll curated HN Algolia results as discussion-only corroboration."""

    def __init__(
        self,
        source: BriefingSourceConfig,
        *,
        fixture_base_dir: Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.source = source
        self.fixture_base_dir = fixture_base_dir
        self.session = session or build_session()

    def poll(self) -> list[IntelligenceEvent]:
        """Poll and normalize HN result hits."""
        payload = self._load_payload()
        hits = payload.get("hits", []) if isinstance(payload, dict) else payload
        if not isinstance(hits, list):
            return []
        return [
            self._event_from_hit(hit)
            for hit in hits[: self.source.max_events_per_run]
            if isinstance(hit, dict)
        ]

    def _load_payload(self) -> Any:
        fixture = read_fixture_text(self.source, self.fixture_base_dir)
        if fixture is not None:
            return json.loads(fixture)
        query = self.source.query or "AI agent"
        url = str(self.source.api_url or "https://hn.algolia.com/api/v1/search")
        response = self.session.get(
            url, params={"query": query}, timeout=DEFAULT_HTTP_TIMEOUT
        )
        response.raise_for_status()
        return response.json()

    def _event_from_hit(self, hit: dict[str, Any]) -> IntelligenceEvent:
        title = str(hit.get("title") or hit.get("story_title") or "HN discussion")
        object_id = str(hit.get("objectID") or hit.get("story_id") or title)
        linked_url = str(hit.get("url") or hit.get("story_url") or "")
        discussion_url = f"https://news.ycombinator.com/item?id={object_id}"
        canonical_url = canonicalize_url(linked_url or discussion_url)
        summary = str(hit.get("comment_text") or hit.get("story_text") or "")
        dedup_key = dedup_key_for(
            source=self.source,
            title=title,
            canonical_url=canonical_url,
            source_native_id=object_id,
        )
        return IntelligenceEvent(
            event_id=event_id_for(self.source, canonical_url, object_id),
            source_name=self.source.source_name,
            source_id=self.source.source_id,
            source_type=self.source.source_class,
            source_policy=SourcePolicy.DISCUSSION_ONLY,
            item_type="hacker_news_item",
            canonical_url=canonical_url,
            title=title,
            retrieved_at=utc_now_iso(),
            collection_method=AccessMethod.HACKER_NEWS,
            content_hash=content_hash_for(
                title, canonical_url, hit.get("created_at"), summary[:240]
            ),
            dedup_key=dedup_key,
            published_at=str(hit.get("created_at")) if hit.get("created_at") else None,
            source_native_id=object_id,
            summary_hint=summary[:500],
            excerpt=summary[:500],
            topics=(topic_id_for_title(title),),
            artifact_links=tuple(filter(None, (canonical_url, discussion_url))),
            confidence="low",
            evidence_type="speculation_or_watch_item",
        )
