"""Serializable observations from a bounded public search API probe."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ProbeSource(StrEnum):
    """Fixed public endpoints available to the diagnostic."""

    ALL = "all"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    DBLP = "dblp"
    OPENALEX = "openalex"


ProbeStatus = Literal[
    "ok",
    "cooldown",
    "rate_limited",
    "bot_challenge",
    "auth_required",
    "forbidden",
    "redirect",
    "http_error",
    "unexpected_response",
    "timeout",
    "tls_error",
    "dns_error",
    "connection_error",
]


class ProbeResult(BaseModel):
    """An observation, not a claim about an IP-specific or permanent ban."""

    model_config = ConfigDict(frozen=True)

    source: ProbeSource
    endpoint: str
    checked_at: datetime
    status: ProbeStatus
    http_status: int | None = None
    request_sent: bool = Field(
        description="A network request was attempted; origin receipt is not guaranteed."
    )
    received_http_response: bool
    elapsed_seconds: float = 0
    retry_after_seconds: float | None = None
    cooldown_until: datetime | None = None
    response_bytes: int = 0
    body_truncated: bool = False
    returned_items: int | None = None


class ProbeReport(BaseModel):
    """Timestamped, anonymous search endpoint checks without raw server content."""

    model_config = ConfigDict(frozen=True)

    started_at: datetime
    finished_at: datetime
    results: list[ProbeResult]
    mode: Literal["anonymous"] = "anonymous"
    minimum_spacing_seconds: float = 30

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_available(self) -> bool:
        """All selected endpoints returned a usable search response."""
        return bool(self.results) and all(item.status == "ok" for item in self.results)
