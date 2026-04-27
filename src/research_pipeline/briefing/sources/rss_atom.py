"""RSS/Atom source adapter for daily briefings."""

from __future__ import annotations

from pathlib import Path

import requests
from lxml import etree  # type: ignore[import-untyped]

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    IntelligenceEvent,
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


class RssAtomSource:
    """Poll conservative RSS/Atom fields from one whitelisted feed."""

    def __init__(
        self,
        source: BriefingSourceConfig,
        *,
        state: dict[str, str] | None = None,
        fixture_base_dir: Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.source = source
        self.state = state if state is not None else {}
        self.fixture_base_dir = fixture_base_dir
        self.session = session or build_session()

    def poll(self) -> list[IntelligenceEvent]:
        """Poll feed entries and normalize them into events."""
        text = self._load_text()
        if not text:
            return []
        parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
        root = etree.fromstring(text.encode("utf-8"), parser=parser)
        entries = self._entry_nodes(root)
        events = [
            self._event_from_entry(entry)
            for entry in entries[: self.source.max_events_per_run]
        ]
        return [event for event in events if event is not None]

    def _load_text(self) -> str:
        fixture = read_fixture_text(self.source, self.fixture_base_dir)
        if fixture is not None:
            return fixture
        if self.source.feed_url is None:
            raise ValueError("feed_url is required")
        headers: dict[str, str] = {}
        if etag := self.state.get("etag"):
            headers["If-None-Match"] = etag
        if last_modified := self.state.get("last_modified"):
            headers["If-Modified-Since"] = last_modified
        response = self.session.get(
            str(self.source.feed_url), headers=headers, timeout=20
        )
        self.state["status_code"] = str(response.status_code)
        if response.status_code == 304:
            return ""
        response.raise_for_status()
        if etag := response.headers.get("ETag"):
            self.state["etag"] = etag
        if last_modified := response.headers.get("Last-Modified"):
            self.state["last_modified"] = last_modified
        return response.text

    def _entry_nodes(self, root: etree._Element) -> list[etree._Element]:
        local = etree.QName(root).localname.lower()
        if local == "rss":
            return list(root.xpath("./channel/item"))
        if local == "feed":
            return list(root.xpath("./*[local-name()='entry']"))
        if local in {"rdf", "rdf:rdf"}:
            return list(root.xpath(".//*[local-name()='item']"))
        return []

    def _text(self, node: etree._Element, *names: str) -> str:
        for name in names:
            found = node.xpath(f"./*[local-name()='{name}']")
            if found:
                value = "".join(found[0].itertext()).strip()
                if value:
                    return value
        return ""

    def _link(self, node: etree._Element) -> str:
        rss_link = self._text(node, "link")
        if rss_link:
            return rss_link
        links = node.xpath("./*[local-name()='link']")
        for link in links:
            href = link.get("href")
            if href:
                return str(href)
        return str(self.source.official_url or self.source.feed_url or "")

    def _event_from_entry(self, entry: etree._Element) -> IntelligenceEvent | None:
        title = self._text(entry, "title")
        if not title:
            return None
        url = self._link(entry)
        if not url:
            return None
        canonical_url = canonicalize_url(url)
        guid = self._text(entry, "guid", "id")
        published_at = self._text(entry, "pubDate", "published", "updated") or None
        summary = self._text(entry, "summary", "description", "content")
        source_native_id = guid or canonical_url
        dedup_key = dedup_key_for(
            source=self.source,
            title=title,
            canonical_url=canonical_url,
            source_native_id=source_native_id,
        )
        return IntelligenceEvent(
            event_id=event_id_for(self.source, canonical_url, source_native_id),
            source_name=self.source.source_name,
            source_id=self.source.source_id,
            source_type=self.source.source_class,
            item_type="rss_atom_item",
            canonical_url=canonical_url,
            title=title,
            retrieved_at=utc_now_iso(),
            collection_method=AccessMethod.RSS_ATOM,
            content_hash=content_hash_for(
                title, canonical_url, published_at, summary[:240]
            ),
            dedup_key=dedup_key,
            published_at=published_at,
            source_native_id=source_native_id,
            summary_hint=summary[:500],
            excerpt=summary[:500],
            topics=(topic_id_for_title(title),),
            artifact_links=(canonical_url,),
            confidence="high",
        )
