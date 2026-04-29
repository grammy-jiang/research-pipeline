"""Unit tests for Phase F academic enrichment."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
)
from research_pipeline.briefing.sources.academic_enrichment import (
    enrich_event,
    enrich_with_crossref,
    enrich_with_openalex,
    enrich_with_semantic_scholar,
    load_enrichment_fixture,
)
from research_pipeline.briefing.sources.papers import map_arxiv_candidate

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "briefing"


def _source() -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="arxiv-mcp",
        source_name="arXiv MCP",
        source_class=SourceClass.ACADEMIC_SOURCE,
        access_method=AccessMethod.ARXIV,
        fixture_path="x",
        enabled=False,
    )


def _event() -> object:
    return map_arxiv_candidate(
        {
            "arxiv_id": "2503.12345",
            "title": "Constitutional MCP",
            "summary": "abstract",
            "url": "https://arxiv.org/abs/2503.12345",
        },
        _source(),
    )


class TestEnrichWithOpenAlex:
    def test_adds_doi_and_topics(self) -> None:
        event = _event()
        record = load_enrichment_fixture(_FIXTURES, "openalex_2503_12345.json")
        out = enrich_with_openalex(event, record)
        assert out.identifiers["doi"] == "10.48550/arXiv.2503.12345"
        assert "Authentication" in out.topics
        assert out.raw_metadata["openalex"]["cited_by_count"] == 12
        assert out.dedup_key == event.dedup_key
        assert out.content_hash == event.content_hash

    def test_none_record_passthrough(self) -> None:
        event = _event()
        out = enrich_with_openalex(event, None)
        assert out == event


class TestEnrichWithSemanticScholar:
    def test_adds_paper_id_and_citation_count(self) -> None:
        event = _event()
        record = load_enrichment_fixture(_FIXTURES, "semantic_scholar_2503_12345.json")
        out = enrich_with_semantic_scholar(event, record)
        assert out.identifiers["s2_paper_id"] == "abc123"
        assert out.raw_metadata["semantic_scholar"]["citation_count"] == 8

    def test_passthrough_when_missing(self) -> None:
        event = _event()
        assert enrich_with_semantic_scholar(event, None) == event


class TestEnrichWithCrossref:
    def test_adds_venue_and_count(self) -> None:
        event = _event()
        record = load_enrichment_fixture(_FIXTURES, "crossref_2503_12345.json")
        out = enrich_with_crossref(event, record)
        assert out.raw_metadata["crossref"]["venue"] == "arXiv preprint"
        assert out.identifiers["doi"] == "10.48550/arXiv.2503.12345"


class TestEnrichEventComposite:
    def test_all_three_compose(self) -> None:
        event = _event()
        oa = load_enrichment_fixture(_FIXTURES, "openalex_2503_12345.json")
        s2 = load_enrichment_fixture(_FIXTURES, "semantic_scholar_2503_12345.json")
        cr = load_enrichment_fixture(_FIXTURES, "crossref_2503_12345.json")
        out = enrich_event(event, openalex=oa, semantic_scholar=s2, crossref=cr)
        assert "openalex_id" in out.identifiers
        assert "s2_paper_id" in out.identifiers
        assert "doi" in out.identifiers
        # original dedup key unchanged
        assert out.dedup_key == event.dedup_key

    def test_does_not_overwrite_existing_identifier(self) -> None:
        event = _event()
        # event already has arxiv_id; openalex sets a different one — ensure
        # the original arxiv_id is preserved.
        out = enrich_with_openalex(event, {"id": "W999"})
        assert out.identifiers["arxiv_id"] == "2503.12345"
        assert out.identifiers["openalex_id"] == "W999"
