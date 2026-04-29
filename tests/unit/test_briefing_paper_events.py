"""Unit tests for Phase F paper-event mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    IntelligenceEvent,
    SourceClass,
)
from research_pipeline.briefing.sources.papers import (
    PaperEventsSource,
    map_arxiv_candidate,
    map_hf_paper,
)

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "briefing" / "papers"


def _arxiv_source() -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="arxiv-mcp",
        source_name="arXiv MCP candidates",
        source_class=SourceClass.ACADEMIC_SOURCE,
        access_method=AccessMethod.ARXIV,
        fixture_path="arxiv_candidates.jsonl",
        enabled=False,
        max_events_per_run=10,
    )


def _hf_source() -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="hf-papers",
        source_name="Hugging Face papers",
        source_class=SourceClass.ACADEMIC_SOURCE,
        access_method=AccessMethod.HUGGINGFACE_PAPERS,
        fixture_path="hf_papers.json",
        enabled=False,
        max_events_per_run=10,
    )


class TestMapArxivCandidate:
    def test_happy_path(self) -> None:
        event = map_arxiv_candidate(
            {
                "arxiv_id": "2503.12345",
                "title": "Constitutional MCP",
                "summary": "abstract text",
                "authors": ["A. Alpha", "B. Beta"],
                "published": "2026-04-29T00:00:00Z",
                "url": "https://arxiv.org/abs/2503.12345",
            },
            _arxiv_source(),
        )
        assert isinstance(event, IntelligenceEvent)
        assert event.item_type == "arxiv_paper"
        assert event.collection_method == AccessMethod.ARXIV
        assert event.source_native_id == "2503.12345"
        assert event.identifiers["arxiv_id"] == "2503.12345"
        assert event.author_or_org == "A. Alpha, B. Beta"
        assert "arxiv.org" in event.canonical_url

    def test_dedup_key_is_stable_across_calls(self) -> None:
        record = {
            "arxiv_id": "2503.99",
            "title": "Same paper",
            "url": "https://arxiv.org/abs/2503.99",
        }
        a = map_arxiv_candidate(record, _arxiv_source())
        b = map_arxiv_candidate(record, _arxiv_source())
        assert a.dedup_key == b.dedup_key

    def test_missing_arxiv_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="arxiv_id"):
            map_arxiv_candidate({"title": "x"}, _arxiv_source())

    def test_missing_title_rejected(self) -> None:
        with pytest.raises(ValueError, match="title"):
            map_arxiv_candidate({"arxiv_id": "x"}, _arxiv_source())


class TestMapHFPaper:
    def test_happy_path(self) -> None:
        event = map_hf_paper(
            {
                "paper": {
                    "id": "2503.55555",
                    "title": "Adaptive Retrieval",
                    "summary": "summary",
                    "publishedAt": "2026-04-30",
                    "upvotes": 42,
                }
            },
            _hf_source(),
        )
        assert event.item_type == "huggingface_paper"
        assert event.collection_method == AccessMethod.HUGGINGFACE_PAPERS
        assert event.identifiers["hf_paper_id"] == "2503.55555"
        assert event.raw_metadata["upvotes"] == 42

    def test_accepts_flat_record(self) -> None:
        event = map_hf_paper(
            {"id": "abc", "title": "T", "summary": "s"},
            _hf_source(),
        )
        assert event.source_native_id == "abc"


class TestPaperEventsSource:
    def test_arxiv_jsonl_fixture(self) -> None:
        events = PaperEventsSource(_arxiv_source(), fixture_base_dir=_FIXTURES).poll()
        assert len(events) == 2
        assert all(e.collection_method == AccessMethod.ARXIV for e in events)
        assert {e.source_native_id for e in events} == {"2503.12345", "2503.67890"}

    def test_hf_json_fixture(self) -> None:
        events = PaperEventsSource(_hf_source(), fixture_base_dir=_FIXTURES).poll()
        assert len(events) == 2
        assert all(
            e.collection_method == AccessMethod.HUGGINGFACE_PAPERS for e in events
        )

    def test_unsupported_access_method_rejected(self) -> None:
        bad = BriefingSourceConfig(
            source_id="x",
            source_name="x",
            source_class=SourceClass.ACADEMIC_SOURCE,
            access_method=AccessMethod.GITHUB_RELEASES,
            fixture_path="x",
            repo_owner="o",
            repo_name="r",
        )
        with pytest.raises(ValueError, match="only supports"):
            PaperEventsSource(bad)

    def test_max_events_respected(self) -> None:
        source = _arxiv_source().model_copy(update={"max_events_per_run": 1})
        events = PaperEventsSource(source, fixture_base_dir=_FIXTURES).poll()
        assert len(events) == 1
