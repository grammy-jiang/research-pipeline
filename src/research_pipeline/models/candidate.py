"""Candidate record model representing a paper found via search.

Originally arXiv-only; extended to support multiple sources (Semantic Scholar,
OpenAlex, DBLP) with optional enrichment fields.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class CandidateRecord(BaseModel):
    """A single paper record from any search source."""

    arxiv_id: str = Field(description='Base arXiv ID, e.g. "2501.12345".')
    version: str = Field(description='Version string, e.g. "v2".')
    title: str = Field(description="Paper title.")
    authors: list[str] = Field(description="List of author names.")
    published: datetime = Field(description="v1 submission time (UTC).")
    updated: datetime = Field(description="This version's submission time (UTC).")
    categories: list[str] = Field(description="All arXiv categories.")
    primary_category: str = Field(description="Primary arXiv category.")
    abstract: str = Field(description="Paper abstract text.")
    abs_url: str = Field(description="URL to the abstract page.")
    pdf_url: str = Field(description="URL to the PDF.")

    # Multi-source fields (Phase 1.3 — all optional, backward-compatible)
    source: str = Field(default="arxiv", description="Which source found this paper.")
    doi: str | None = Field(default=None, description="Digital Object Identifier.")
    semantic_scholar_id: str | None = Field(
        default=None, description="Semantic Scholar paper ID."
    )
    openalex_id: str | None = Field(default=None, description="OpenAlex work ID.")
    citation_count: int | None = Field(
        default=None, description="Total citation count."
    )
    influential_citation_count: int | None = Field(
        default=None, description="Influential citation count (Semantic Scholar)."
    )
    venue: str | None = Field(default=None, description="Publication venue name.")
    year: int | None = Field(default=None, description="Publication year.")
