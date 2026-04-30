"""Tests for HtmlScrapeSource adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
)
from research_pipeline.briefing.sources.html_scrape import HtmlScrapeSource

FIXTURE_BASE = Path(__file__).parents[1] / "fixtures" / "briefing"


def _source(
    *,
    fixture_path: str,
    link_path_prefix: str = "/research/",
    feed_url: str = "https://www.example.com/research",
    max_events: int = 10,
) -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="html.example",
        source_name="Example HTML Scrape",
        source_class=SourceClass.PRIMARY_ARTIFACT,
        access_method=AccessMethod.HTML_SCRAPE,
        feed_url=feed_url,
        link_path_prefix=link_path_prefix,
        fixture_path=fixture_path,
        max_events_per_run=max_events,
        enabled=True,
    )


def test_html_scrape_extracts_matching_links_and_dedupes() -> None:
    source = _source(fixture_path="html_scrape/listing_normal.html")
    adapter = HtmlScrapeSource(source, fixture_base_dir=FIXTURE_BASE)

    events = adapter.poll()

    canonicals = [event.canonical_url for event in events]
    assert (
        "https://www.example.com/research/automated-alignment-researchers" in canonicals
    )
    assert "https://www.example.com/research/constitutional-classifiers" in canonicals
    assert (
        canonicals.count(
            "https://www.example.com/research/automated-alignment-researchers"
        )
        == 1
    )
    assert "https://www.example.com/research/" not in canonicals
    assert all("/news/" not in url for url in canonicals)
    assert all("other.example.com" not in url for url in canonicals)


def test_html_scrape_extracts_title_and_published_at() -> None:
    source = _source(fixture_path="html_scrape/listing_normal.html")
    adapter = HtmlScrapeSource(source, fixture_base_dir=FIXTURE_BASE)

    events = {event.source_native_id: event for event in adapter.poll()}
    aar = events["automated-alignment-researchers"]

    assert aar.title == "Automated Alignment Researchers"
    assert aar.summary_hint and "Automated Alignment Researchers" in aar.summary_hint
    assert aar.published_at == "2026-04-14T00:00:00Z"
    assert aar.collection_method == AccessMethod.HTML_SCRAPE
    assert aar.item_type == "html_scrape_item"
    assert aar.confidence == "medium"


def test_html_scrape_caps_events() -> None:
    source = _source(fixture_path="html_scrape/listing_normal.html", max_events=1)
    adapter = HtmlScrapeSource(source, fixture_base_dir=FIXTURE_BASE)

    events = adapter.poll()

    assert len(events) == 1


def test_html_scrape_requires_link_path_prefix() -> None:
    with pytest.raises(ValueError, match="link_path_prefix"):
        BriefingSourceConfig(
            source_id="html.bad",
            source_name="Bad",
            source_class=SourceClass.PRIMARY_ARTIFACT,
            access_method=AccessMethod.HTML_SCRAPE,
            feed_url="https://www.example.com/research",
        )
