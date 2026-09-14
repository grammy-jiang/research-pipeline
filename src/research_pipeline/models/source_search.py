"""Structured source coverage outcomes shared by CLI and MCP."""

from typing import Literal

from pydantic import BaseModel, Field

from research_pipeline.models.candidate import CandidateRecord


class SourceAttempt(BaseModel):
    """Application-level query attempt; HTTP dispatches have separate telemetry."""

    source: str
    window_months: int
    date_from: str | None = None
    date_to: str | None = None
    query_strategy: str
    planned_queries: list[str] = Field(default_factory=list)
    attempted_queries: list[str] = Field(default_factory=list)
    status: Literal["ok", "empty", "partial", "failed", "unavailable", "cooldown"]
    candidate_count: int = 0
    error_type: str | None = None
    detail: str | None = None
    not_before: float | None = None


class SourceSearchReport(BaseModel):
    """Distinguish empty search evidence, lost coverage and usable partial output."""

    status: Literal["ok", "empty", "partial", "failed"]
    sources: list[str]
    attempts: list[SourceAttempt] = Field(default_factory=list)
    candidates: list[CandidateRecord] = Field(default_factory=list)
    date_window_basis: str = "months multiplied by 30 days; not calendar months"
    http_telemetry: str = "source_http dispatch/response/deferred logger events"
