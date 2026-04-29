from __future__ import annotations

from pathlib import Path

import pytest
from lxml import etree

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
)
from research_pipeline.briefing.sources.rss_atom import RssAtomSource

FIXTURE_BASE = Path(__file__).parents[1] / "fixtures" / "briefing"


def _source(fixture_path: str) -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="rss.example",
        source_name="RSS Example",
        source_class=SourceClass.TECHNICAL_DISCUSSION,
        access_method=AccessMethod.RSS_ATOM,
        fixture_path=fixture_path,
        max_events_per_run=10,
        enabled=True,
    )


def test_rss_normal_fixture_maps_to_events() -> None:
    source = _source("rss/rss_normal.xml")
    adapter = RssAtomSource(source, fixture_base_dir=FIXTURE_BASE)

    events = adapter.poll()

    assert len(events) == 2
    first = events[0]
    assert first.source_id == "rss.example"
    assert first.item_type == "rss_atom_item"
    assert first.source_native_id == "rss-item-1"
    assert first.canonical_url == "https://example.com/releases/mcp-adapter-1"


def test_atom_normal_fixture_maps_to_events() -> None:
    source = _source("atom/atom_normal.xml")
    adapter = RssAtomSource(source, fixture_base_dir=FIXTURE_BASE)

    events = adapter.poll()

    assert len(events) == 1
    event = events[0]
    assert event.source_native_id == "atom-item-1"
    assert event.canonical_url == "https://example.com/atom/releases/1"
    assert event.title == "Atom Entry Release"


def test_rss_empty_fixture_returns_no_events() -> None:
    source = _source("rss/rss_empty.xml")
    adapter = RssAtomSource(source, fixture_base_dir=FIXTURE_BASE)

    assert adapter.poll() == []


def test_rss_malformed_fixture_raises_parser_error() -> None:
    source = _source("rss/rss_malformed.xml")
    adapter = RssAtomSource(source, fixture_base_dir=FIXTURE_BASE)

    with pytest.raises(etree.XMLSyntaxError):
        adapter.poll()
