"""Video / audio source adapter (Phase F).

Reads YouTube channel Atom feeds and podcast RSS feeds and emits events
classed as ``SourceClass.VIDEO_AUDIO`` (per registry policy). Audio/video
content is treated as a watch-item by default — confidence is medium and
events do not auto-promote to primary findings without curator approval.
"""

from __future__ import annotations

from pathlib import Path

import requests
from lxml import etree  # type: ignore[import-untyped]

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    IntelligenceEvent,
    SourceClass,
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

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_YT_NS = "{http://www.youtube.com/xml/schemas/2015}"


class VideoAudioSource:
    """Poll YouTube Atom or podcast RSS feeds."""

    def __init__(
        self,
        source: BriefingSourceConfig,
        *,
        fixture_base_dir: Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if source.access_method != AccessMethod.VIDEO_AUDIO:
            raise ValueError(
                f"VideoAudioSource requires VIDEO_AUDIO (got {source.access_method})"
            )
        self.source = source
        self.fixture_base_dir = fixture_base_dir
        self.session = session or build_session()

    def poll(self) -> list[IntelligenceEvent]:
        """Fetch and parse the feed."""
        text = self._load_text()
        if not text:
            return []
        parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
        root = etree.fromstring(text.encode("utf-8"), parser=parser)
        if root.tag.endswith("rss"):
            return self._parse_rss(root)
        return self._parse_atom(root)

    def _load_text(self) -> str | None:
        text = read_fixture_text(self.source, self.fixture_base_dir)
        if text is not None:
            return text
        url = str(self.source.feed_url or self.source.api_url or "")
        if not url:
            raise ValueError("video_audio source requires feed_url when no fixture")
        response = self.session.get(url, timeout=DEFAULT_HTTP_TIMEOUT)
        response.raise_for_status()
        return response.text

    def _parse_atom(self, root: etree._Element) -> list[IntelligenceEvent]:
        events: list[IntelligenceEvent] = []
        for entry in root.findall(f"{_ATOM_NS}entry")[: self.source.max_events_per_run]:
            video_id_el = entry.find(f"{_YT_NS}videoId")
            entry_id_el = entry.find(f"{_ATOM_NS}id")
            title_el = entry.find(f"{_ATOM_NS}title")
            link_el = entry.find(f"{_ATOM_NS}link")
            published_el = entry.find(f"{_ATOM_NS}published")
            author_el = entry.find(f"{_ATOM_NS}author/{_ATOM_NS}name")
            native_id = (
                (video_id_el.text or "").strip()
                if video_id_el is not None and video_id_el.text
                else (entry_id_el.text or "").strip()
                if entry_id_el is not None and entry_id_el.text
                else ""
            )
            title = (title_el.text or "").strip() if title_el is not None else ""
            url = link_el.attrib.get("href", "") if link_el is not None else ""
            published_at = (
                (published_el.text or "").strip()
                if published_el is not None and published_el.text
                else None
            )
            author = (
                (author_el.text or "").strip()
                if author_el is not None and author_el.text
                else None
            )
            if not title or not native_id:
                continue
            events.append(
                self._build_event(
                    title=title,
                    canonical_url=canonicalize_url(url),
                    native_id=native_id,
                    published_at=published_at,
                    author=author,
                    item_type="video",
                )
            )
        return events

    def _parse_rss(self, root: etree._Element) -> list[IntelligenceEvent]:
        events: list[IntelligenceEvent] = []
        items = root.findall("./channel/item")[: self.source.max_events_per_run]
        for item in items:
            title_el = item.find("title")
            link_el = item.find("link")
            guid_el = item.find("guid")
            pub_el = item.find("pubDate")
            title = (title_el.text or "").strip() if title_el is not None else ""
            url = (link_el.text or "").strip() if link_el is not None else ""
            native_id = (
                (guid_el.text or "").strip()
                if guid_el is not None and guid_el.text
                else url
            )
            published_at = (
                (pub_el.text or "").strip()
                if pub_el is not None and pub_el.text
                else None
            )
            if not title or not native_id:
                continue
            events.append(
                self._build_event(
                    title=title,
                    canonical_url=canonicalize_url(url),
                    native_id=native_id,
                    published_at=published_at,
                    author=None,
                    item_type="podcast_episode",
                )
            )
        return events

    def _build_event(
        self,
        *,
        title: str,
        canonical_url: str,
        native_id: str,
        published_at: str | None,
        author: str | None,
        item_type: str,
    ) -> IntelligenceEvent:
        return IntelligenceEvent(
            event_id=event_id_for(self.source, canonical_url, native_id),
            source_name=self.source.source_name,
            source_id=self.source.source_id,
            source_type=SourceClass.VIDEO_AUDIO,
            source_policy=SourcePolicy.PUBLIC_OFFICIAL,
            item_type=item_type,
            canonical_url=canonical_url,
            title=title,
            retrieved_at=utc_now_iso(),
            collection_method=AccessMethod.VIDEO_AUDIO,
            content_hash=content_hash_for(title, canonical_url, published_at, ""),
            dedup_key=dedup_key_for(
                source=self.source,
                title=title,
                canonical_url=canonical_url,
                source_native_id=native_id,
            ),
            author_or_org=author,
            published_at=published_at,
            source_native_id=native_id,
            topics=(topic_id_for_title(title),),
            confidence="medium",
            evidence_type="speculation_or_watch_item",
        )
