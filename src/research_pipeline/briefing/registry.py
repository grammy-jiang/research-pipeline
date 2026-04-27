"""Source registry loading and governance validation."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_pipeline.briefing.io import model_to_jsonable, write_json
from research_pipeline.briefing.models import AccessMethod, BriefingSourceConfig


class SourceRegistry(BaseModel):
    """Validated briefing source registry."""

    model_config = ConfigDict(frozen=True)

    sources: tuple[BriefingSourceConfig, ...] = ()
    watchlist_terms: tuple[str, ...] = ()
    max_sources_per_run: int = Field(default=50, ge=1, le=200)

    @model_validator(mode="after")
    def validate_unique_source_ids(self) -> SourceRegistry:
        """Prevent ambiguous source IDs."""
        ids = [source.source_id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source IDs must be unique")
        return self

    def enabled_sources(self) -> list[BriefingSourceConfig]:
        """Return enabled sources within the configured source budget."""
        return [source for source in self.sources if source.enabled][
            : self.max_sources_per_run
        ]

    def snapshot(self, path: Path) -> None:
        """Write a source registry snapshot for a run."""
        write_json(path, model_to_jsonable(self))


def _coerce_registry_data(data: dict[str, Any]) -> dict[str, Any]:
    if "sources" in data:
        return data
    if "briefing" in data and isinstance(data["briefing"], dict):
        briefing = data["briefing"]
        if "sources" in briefing:
            return briefing
    return {"sources": []}


def load_source_registry(path: Path | None) -> SourceRegistry:
    """Load a source registry from JSON or TOML."""
    if path is None:
        return SourceRegistry()
    if not path.exists():
        raise FileNotFoundError(f"source registry not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    return SourceRegistry.model_validate(_coerce_registry_data(data))


def assert_phase_a_source_boundary(source: BriefingSourceConfig) -> None:
    """Reject source expansion without explicit review metadata."""
    phase_a = {
        AccessMethod.GITHUB_RELEASES,
        AccessMethod.RSS_ATOM,
        AccessMethod.MANUAL,
    }
    if (
        source.enabled
        and source.access_method not in phase_a
        and source.last_reviewed_at is None
    ):
        raise ValueError(
            f"source {source.source_id} uses {source.access_method}; "
            "source expansion requires an explicit later-phase enablement review"
        )
