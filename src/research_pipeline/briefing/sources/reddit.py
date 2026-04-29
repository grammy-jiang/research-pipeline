"""Reddit source adapter (Phase F).

Consumes the public Reddit JSON listing format (``/r/<sub>.json`` or saved
fixtures of the same shape). Disabled-by-default discussion-only source —
ranking weight is governed by the registry (``SourcePolicy.DISCUSSION_ONLY``).
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
from research_pipeline.briefing.sources.base import build_session, read_fixture_text


class RedditSource:
    """Poll a Reddit listing JSON document."""

    def __init__(
        self,
        source: BriefingSourceConfig,
        *,
        fixture_base_dir: Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if source.access_method != AccessMethod.REDDIT_API:
            raise ValueError(
                f"RedditSource requires REDDIT_API (got {source.access_method})"
            )
        self.source = source
        self.fixture_base_dir = fixture_base_dir
        self.session = session or build_session()

    def poll(self) -> list[IntelligenceEvent]:
        """Fetch the listing and return discussion-only events."""
        payload = self._load_payload()
        children = self._extract_children(payload)
        return [
            self._event_from_post(child["data"])
            for child in children[: self.source.max_events_per_run]
            if isinstance(child, dict) and isinstance(child.get("data"), dict)
        ]

    def _load_payload(self) -> Any:
        text = read_fixture_text(self.source, self.fixture_base_dir)
        if text is not None:
            return json.loads(text)
        url = str(self.source.api_url or "")
        if not url:
            raise ValueError("reddit source requires api_url when no fixture")
        response = self.session.get(url, timeout=20)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_children(payload: Any) -> list[Any]:
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if isinstance(data, dict):
            children = data.get("children")
            if isinstance(children, list):
                return children
        return []

    def _event_from_post(self, post: dict[str, Any]) -> IntelligenceEvent:
        title = str(post.get("title") or "Reddit post")
        post_id = str(post.get("id") or post.get("name") or title)
        permalink = post.get("permalink") or f"/r/{post.get('subreddit', '')}"
        url = post.get("url") or f"https://www.reddit.com{permalink}"
        canonical_url = canonicalize_url(str(url))
        summary = str(post.get("selftext") or "")[:500]
        created = post.get("created_utc")
        published_at = (
            f"{int(float(created))}" if isinstance(created, (int, float)) else None
        )
        return IntelligenceEvent(
            event_id=event_id_for(self.source, canonical_url, post_id),
            source_name=self.source.source_name,
            source_id=self.source.source_id,
            source_type=self.source.source_class,
            source_policy=SourcePolicy.DISCUSSION_ONLY,
            item_type="reddit_post",
            canonical_url=canonical_url,
            title=title,
            retrieved_at=utc_now_iso(),
            collection_method=AccessMethod.REDDIT_API,
            content_hash=content_hash_for(title, canonical_url, published_at, summary),
            dedup_key=dedup_key_for(
                source=self.source,
                title=title,
                canonical_url=canonical_url,
                source_native_id=post_id,
            ),
            published_at=published_at,
            source_native_id=post_id,
            summary_hint=summary[:280],
            excerpt=summary,
            topics=(topic_id_for_title(title),),
            confidence="low",
            evidence_type="speculation_or_watch_item",
            raw_metadata={
                "subreddit": post.get("subreddit"),
                "score": post.get("score"),
                "num_comments": post.get("num_comments"),
            },
        )
