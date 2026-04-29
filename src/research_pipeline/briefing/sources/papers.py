"""Phase F paper-event mapping.

Provides ``PaperEventsSource``, an offline/online adapter that consumes
paper-candidate JSON/JSONL fixtures (such as arXiv search candidates or
Hugging Face daily-paper exports) and yields ``IntelligenceEvent``s.

Pure functions ``map_arxiv_candidate`` and ``map_hf_paper`` are exposed for
unit testing and reuse in the academic-enrichment helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    IntelligenceEvent,
    SourceClass,
)
from research_pipeline.briefing.normalize import (
    canonicalize_url,
    content_hash_for,
    dedup_key_for,
    event_id_for,
    utc_now_iso,
)
from research_pipeline.briefing.sources.base import read_fixture_text


def _coerce_str(value: object, default: str = "") -> str:
    return str(value) if value is not None else default


def map_arxiv_candidate(
    record: dict[str, Any],
    source: BriefingSourceConfig,
    *,
    retrieved_at: str | None = None,
) -> IntelligenceEvent:
    """Map a single arXiv candidate dict into an ``IntelligenceEvent``."""
    arxiv_id = _coerce_str(record.get("arxiv_id") or record.get("id"))
    if not arxiv_id:
        raise ValueError("arxiv candidate missing arxiv_id/id")
    title = _coerce_str(record.get("title")).strip()
    if not title:
        raise ValueError("arxiv candidate missing title")
    summary = _coerce_str(record.get("summary") or record.get("abstract"))
    abs_url = _coerce_str(record.get("url") or f"https://arxiv.org/abs/{arxiv_id}")
    pdf_url = _coerce_str(record.get("pdf_url"))
    canonical_url = canonicalize_url(abs_url)
    published_at = (
        _coerce_str(record.get("published") or record.get("published_at")) or None
    )
    authors_raw = record.get("authors") or ()
    if isinstance(authors_raw, list):
        authors = tuple(str(a) for a in authors_raw if a)
    else:
        authors = (str(authors_raw),) if authors_raw else ()
    author_org = ", ".join(authors[:3]) if authors else None
    identifiers = {"arxiv_id": arxiv_id}
    artifacts: tuple[str, ...] = (pdf_url,) if pdf_url else ()

    return IntelligenceEvent(
        event_id=event_id_for(source, canonical_url, arxiv_id),
        source_name=source.source_name,
        source_id=source.source_id,
        source_type=SourceClass.ACADEMIC_SOURCE,
        item_type="arxiv_paper",
        canonical_url=canonical_url,
        title=title,
        retrieved_at=retrieved_at or utc_now_iso(),
        collection_method=AccessMethod.ARXIV,
        content_hash=content_hash_for(title, canonical_url, published_at, summary),
        dedup_key=dedup_key_for(
            source=source,
            title=title,
            canonical_url=canonical_url,
            source_native_id=arxiv_id,
            identifiers=identifiers,
        ),
        author_or_org=author_org,
        published_at=published_at,
        source_native_id=arxiv_id,
        identifiers=identifiers,
        summary_hint=summary[:280],
        excerpt=summary[:600],
        artifact_links=artifacts,
        inferred_entities=authors[:10],
        raw_metadata={k: v for k, v in record.items() if k not in {"summary"}},
    )


def map_hf_paper(
    record: dict[str, Any],
    source: BriefingSourceConfig,
    *,
    retrieved_at: str | None = None,
) -> IntelligenceEvent:
    """Map a Hugging Face daily-papers JSON record into an event."""
    paper_field = record.get("paper")
    paper: dict[str, Any] = paper_field if isinstance(paper_field, dict) else record
    paper_id = _coerce_str(paper.get("id") or paper.get("arxiv_id"))
    if not paper_id:
        raise ValueError("hf paper record missing id/arxiv_id")
    title = _coerce_str(paper.get("title")).strip()
    if not title:
        raise ValueError("hf paper record missing title")
    summary = _coerce_str(paper.get("summary") or paper.get("abstract"))
    url = _coerce_str(paper.get("url") or f"https://huggingface.co/papers/{paper_id}")
    canonical_url = canonicalize_url(url)
    published_at = (
        _coerce_str(paper.get("publishedAt") or paper.get("published_at")) or None
    )
    upvotes = paper.get("upvotes")
    identifiers = {"hf_paper_id": paper_id}
    if isinstance(paper.get("arxiv_id"), str):
        identifiers["arxiv_id"] = str(paper["arxiv_id"])

    return IntelligenceEvent(
        event_id=event_id_for(source, canonical_url, paper_id),
        source_name=source.source_name,
        source_id=source.source_id,
        source_type=SourceClass.ACADEMIC_SOURCE,
        item_type="huggingface_paper",
        canonical_url=canonical_url,
        title=title,
        retrieved_at=retrieved_at or utc_now_iso(),
        collection_method=AccessMethod.HUGGINGFACE_PAPERS,
        content_hash=content_hash_for(title, canonical_url, published_at, summary),
        dedup_key=dedup_key_for(
            source=source,
            title=title,
            canonical_url=canonical_url,
            source_native_id=paper_id,
            identifiers=identifiers,
        ),
        published_at=published_at,
        source_native_id=paper_id,
        identifiers=identifiers,
        summary_hint=summary[:280],
        excerpt=summary[:600],
        raw_metadata={
            "upvotes": upvotes if isinstance(upvotes, int) else None,
            "raw": record,
        },
    )


class PaperEventsSource:
    """Offline-friendly source loading paper events from JSONL or JSON fixtures.

    The ``access_method`` of ``source`` selects the mapper:

    * ``AccessMethod.ARXIV`` reads a JSONL fixture, one candidate per line
    * ``AccessMethod.HUGGINGFACE_PAPERS`` reads a JSON array fixture
    """

    def __init__(
        self,
        source: BriefingSourceConfig,
        *,
        fixture_base_dir: Path | None = None,
    ) -> None:
        if source.access_method not in (
            AccessMethod.ARXIV,
            AccessMethod.HUGGINGFACE_PAPERS,
        ):
            raise ValueError(
                "PaperEventsSource only supports arxiv or huggingface_papers "
                f"access methods (got {source.access_method})"
            )
        self.source = source
        self.fixture_base_dir = fixture_base_dir

    def poll(self) -> list[IntelligenceEvent]:
        """Read the configured fixture and return mapped events."""
        text = read_fixture_text(self.source, self.fixture_base_dir)
        if text is None:
            return []
        records = self._load_records(text)
        retrieved_at = utc_now_iso()
        max_events = self.source.max_events_per_run
        mapper = (
            map_arxiv_candidate
            if self.source.access_method == AccessMethod.ARXIV
            else map_hf_paper
        )
        return [
            mapper(record, self.source, retrieved_at=retrieved_at)
            for record in records[:max_events]
        ]

    def _load_records(self, text: str) -> list[dict[str, Any]]:
        if self.source.access_method == AccessMethod.ARXIV:
            return self._load_jsonl(text)
        return self._load_json_array(text)

    @staticmethod
    def _load_jsonl(text: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError("arxiv jsonl entries must be JSON objects")
            records.append(payload)
        return records

    @staticmethod
    def _load_json_array(text: str) -> list[dict[str, Any]]:
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("hf papers fixture must be a JSON array")
        return [item for item in payload if isinstance(item, dict)]
