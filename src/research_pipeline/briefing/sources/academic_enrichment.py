"""Phase F academic-enrichment helpers.

Pure-function metadata enrichment for paper events using offline OpenAlex,
Semantic Scholar, and Crossref fixtures. The enrichers add citation counts,
DOIs, OA URLs, and topic tags into ``IntelligenceEvent.identifiers`` and
``raw_metadata``. They do NOT change ``dedup_key`` or ``content_hash``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_pipeline.briefing.models import IntelligenceEvent


def _merge_identifiers(
    base: dict[str, str], extra: dict[str, str | None]
) -> dict[str, str]:
    merged = dict(base)
    for key, value in extra.items():
        if value is not None and value != "":
            merged.setdefault(key, str(value))
    return merged


def enrich_with_openalex(
    event: IntelligenceEvent, record: dict[str, Any] | None
) -> IntelligenceEvent:
    """Merge OpenAlex metadata into ``event``.

    Recognised fields: ``id`` (OpenAlex work ID), ``doi``, ``cited_by_count``,
    ``open_access`` (with ``oa_url``), ``concepts`` (list of {display_name}).
    Missing or ``None`` ``record`` returns the event unchanged.
    """
    if not record:
        return event
    new_ids: dict[str, str | None] = {}
    if "id" in record:
        new_ids["openalex_id"] = str(record["id"])
    if "doi" in record:
        new_ids["doi"] = str(record["doi"]).replace("https://doi.org/", "")
    oa = record.get("open_access") or {}
    oa_url = oa.get("oa_url") if isinstance(oa, dict) else None

    citations = record.get("cited_by_count")
    concepts_raw = record.get("concepts") or ()
    concepts: tuple[str, ...] = tuple(
        str(c.get("display_name"))
        for c in concepts_raw
        if isinstance(c, dict) and c.get("display_name")
    )
    raw_metadata = dict(event.raw_metadata)
    raw_metadata["openalex"] = {
        "cited_by_count": citations,
        "oa_url": oa_url,
        "concepts": list(concepts),
    }
    return event.model_copy(
        update={
            "identifiers": _merge_identifiers(event.identifiers, new_ids),
            "topics": event.topics
            + tuple(c for c in concepts if c not in event.topics),
            "raw_metadata": raw_metadata,
        }
    )


def enrich_with_semantic_scholar(
    event: IntelligenceEvent, record: dict[str, Any] | None
) -> IntelligenceEvent:
    """Merge Semantic Scholar metadata into ``event``.

    Recognised fields: ``paperId``, ``citationCount``, ``influentialCitationCount``,
    ``venue``, ``fieldsOfStudy``, ``externalIds`` (with ``DOI``).
    """
    if not record:
        return event
    new_ids: dict[str, str | None] = {}
    if record.get("paperId"):
        new_ids["s2_paper_id"] = str(record["paperId"])
    external = record.get("externalIds") or {}
    if isinstance(external, dict) and external.get("DOI"):
        new_ids["doi"] = str(external["DOI"])

    fields = record.get("fieldsOfStudy") or ()
    if not isinstance(fields, list):
        fields = []
    raw_metadata = dict(event.raw_metadata)
    raw_metadata["semantic_scholar"] = {
        "citation_count": record.get("citationCount"),
        "influential_citation_count": record.get("influentialCitationCount"),
        "venue": record.get("venue"),
        "fields_of_study": list(fields),
    }
    return event.model_copy(
        update={
            "identifiers": _merge_identifiers(event.identifiers, new_ids),
            "raw_metadata": raw_metadata,
        }
    )


def enrich_with_crossref(
    event: IntelligenceEvent, record: dict[str, Any] | None
) -> IntelligenceEvent:
    """Merge Crossref metadata into ``event``.

    Recognised fields: ``DOI``, ``container-title``, ``publisher``,
    ``is-referenced-by-count``.
    """
    if not record:
        return event
    new_ids: dict[str, str | None] = {}
    if record.get("DOI"):
        new_ids["doi"] = str(record["DOI"])
    container = record.get("container-title")
    if isinstance(container, list) and container:
        venue = str(container[0])
    elif isinstance(container, str):
        venue = container
    else:
        venue = ""
    raw_metadata = dict(event.raw_metadata)
    raw_metadata["crossref"] = {
        "doi": record.get("DOI"),
        "venue": venue,
        "publisher": record.get("publisher"),
        "is_referenced_by_count": record.get("is-referenced-by-count"),
    }
    return event.model_copy(
        update={
            "identifiers": _merge_identifiers(event.identifiers, new_ids),
            "raw_metadata": raw_metadata,
        }
    )


def enrich_event(
    event: IntelligenceEvent,
    *,
    openalex: dict[str, Any] | None = None,
    semantic_scholar: dict[str, Any] | None = None,
    crossref: dict[str, Any] | None = None,
) -> IntelligenceEvent:
    """Apply OpenAlex, Semantic Scholar, and Crossref enrichers in order."""
    out = enrich_with_openalex(event, openalex)
    out = enrich_with_semantic_scholar(out, semantic_scholar)
    out = enrich_with_crossref(out, crossref)
    return out


def load_enrichment_fixture(base_dir: Path, name: str) -> dict[str, Any]:
    """Load an enrichment JSON fixture under ``<base_dir>/enrichment/<name>``."""
    path = base_dir / "enrichment" / name
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
