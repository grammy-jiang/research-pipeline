"""Candidate record model representing a paper found on arXiv."""

from datetime import datetime

from pydantic import BaseModel, Field


class CandidateRecord(BaseModel):
    """A single paper record parsed from an arXiv Atom feed entry."""

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
