"""HTML scraping source adapter for daily briefings.

Polls a public listing page (no auth) and extracts article links matching a
configured path prefix. Used for vendor blogs/news pages that do not publish
an RSS or Atom feed (e.g. Anthropic's research/news/engineering pages).

Conservative by design: no JavaScript execution, no recursive crawling, no
content body extraction beyond the visible anchor text. The listing page
itself is the only URL fetched.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from lxml import html as lxml_html  # type: ignore[import-untyped]

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
from research_pipeline.briefing.sources.base import (
    DEFAULT_HTTP_TIMEOUT,
    build_session,
    read_fixture_text,
)

_DATE_INLINE_RE = re.compile(
    r"(?P<date>"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r")",
    re.IGNORECASE,
)


class HtmlScrapeSource:
    """Poll an HTML listing page and emit one event per matching article link."""

    def __init__(
        self,
        source: BriefingSourceConfig,
        *,
        state: dict[str, str] | None = None,
        fixture_base_dir: Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if source.access_method != AccessMethod.HTML_SCRAPE:
            raise ValueError(
                f"HtmlScrapeSource requires HTML_SCRAPE (got {source.access_method})"
            )
        if source.link_path_prefix is None:
            raise ValueError("html_scrape sources require link_path_prefix")
        self.source = source
        self.state = state if state is not None else {}
        self.fixture_base_dir = fixture_base_dir
        self.session = session or build_session()

    def poll(self) -> list[IntelligenceEvent]:
        """Fetch the listing page and emit events for matching article links."""
        text = self._load_text()
        if not text:
            return []
        tree = lxml_html.fromstring(text)
        base_url = str(self.source.feed_url or self.source.official_url or "")
        prefix = self.source.link_path_prefix or "/"
        anchors = tree.xpath("//a[@href]")

        seen: set[str] = set()
        events: list[IntelligenceEvent] = []
        for anchor in anchors:
            href = (anchor.get("href") or "").strip()
            if not self._matches_prefix(href, prefix, base_url):
                continue
            absolute = urljoin(base_url, href) if base_url else href
            canonical = canonicalize_url(absolute)
            if canonical in seen or self._is_listing_root(canonical, prefix):
                continue
            if not self._is_leaf_path(canonical, prefix):
                continue
            seen.add(canonical)
            anchor_text = self._extract_spaced_text(anchor)
            event = self._build_event(canonical, anchor_text)
            if event is not None:
                events.append(event)
            if len(events) >= self.source.max_events_per_run:
                break
        return events

    def _matches_prefix(self, href: str, prefix: str, base_url: str) -> bool:
        if not href:
            return False
        if href.startswith(prefix):
            return True
        if base_url:
            base_host = urlparse(base_url).netloc
            parsed = urlparse(href)
            if parsed.netloc and parsed.netloc != base_host:
                return False
            return parsed.path.startswith(prefix)
        return False

    def _is_listing_root(self, canonical_url: str, prefix: str) -> bool:
        path = urlparse(canonical_url).path.rstrip("/")
        return path.endswith(prefix.rstrip("/"))

    def _is_leaf_path(self, canonical_url: str, prefix: str) -> bool:
        """Return True only if the URL path has exactly one segment after the prefix.

        Rejects nested category/team listing pages such as ``/research/team/<x>``
        while accepting article slugs such as ``/research/<slug>``.
        """
        path = urlparse(canonical_url).path
        normalized_prefix = prefix if prefix.endswith("/") else prefix + "/"
        if not path.startswith(normalized_prefix):
            return False
        remainder = path[len(normalized_prefix) :].strip("/")
        if not remainder:
            return False
        return "/" not in remainder

    def _load_text(self) -> str:
        fixture = read_fixture_text(self.source, self.fixture_base_dir)
        if fixture is not None:
            return fixture
        if self.source.feed_url is None:
            raise ValueError("feed_url is required for live html_scrape sources")
        headers: dict[str, str] = {}
        if etag := self.state.get("etag"):
            headers["If-None-Match"] = etag
        if last_modified := self.state.get("last_modified"):
            headers["If-Modified-Since"] = last_modified
        response = self.session.get(
            str(self.source.feed_url), headers=headers, timeout=DEFAULT_HTTP_TIMEOUT
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

    def _build_event(
        self, canonical_url: str, anchor_text: str
    ) -> IntelligenceEvent | None:
        slug = self._slug_from_url(canonical_url)
        if not slug:
            return None
        title = self._title_from_slug(slug) or self._title_from_anchor_text(anchor_text)
        if not title:
            return None
        published_at = self._published_at_from_anchor_text(anchor_text)
        summary = anchor_text[:500]
        source_native_id = slug
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
            item_type="html_scrape_item",
            canonical_url=canonical_url,
            title=title,
            retrieved_at=utc_now_iso(),
            collection_method=AccessMethod.HTML_SCRAPE,
            content_hash=content_hash_for(
                title, canonical_url, published_at, summary[:240]
            ),
            dedup_key=dedup_key,
            published_at=published_at,
            source_native_id=source_native_id,
            summary_hint=summary,
            excerpt=summary,
            topics=(topic_id_for_title(title),),
            artifact_links=(canonical_url,),
            confidence="medium",
        )

    @staticmethod
    def _extract_spaced_text(element: lxml_html.HtmlElement) -> str:
        """Return text from *element* with whitespace between text fragments.

        ``lxml``'s ``text_content()`` concatenates text from descendant
        elements without inserting separators, which can glue tokens together
        across sibling block elements (e.g. ``<div>Interpretability</div>``
        followed by ``<div>Apr 2, 2026</div>`` becomes
        ``InterpretabilityApr 2, 2026``). ``itertext()`` yields each text
        and tail node separately; joining them with a single space and then
        collapsing whitespace preserves natural word boundaries.
        """
        parts = [segment.strip() for segment in element.itertext()]
        return " ".join(part for part in parts if part)

    @staticmethod
    def _slug_from_url(canonical_url: str) -> str:
        path = urlparse(canonical_url).path.rstrip("/")
        if not path:
            return ""
        return path.rsplit("/", 1)[-1]

    @staticmethod
    def _title_from_slug(slug: str) -> str:
        return " ".join(part.capitalize() for part in re.split(r"[-_]+", slug) if part)

    @staticmethod
    def _title_from_anchor_text(text: str) -> str:
        if not text:
            return ""
        match = _DATE_INLINE_RE.search(text)
        stripped = (text[match.end() :] if match else text).strip()
        if not stripped:
            return ""
        if len(stripped) > 240:
            stripped = stripped[:240].rsplit(" ", 1)[0]
        return stripped

    @staticmethod
    def _published_at_from_anchor_text(text: str) -> str | None:
        match = _DATE_INLINE_RE.search(text)
        if not match:
            return None
        raw = match.group("date").strip().rstrip(",")
        for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(raw, fmt)
            except ValueError:
                continue
            return parsed.strftime("%Y-%m-%dT00:00:00Z")
        return None
