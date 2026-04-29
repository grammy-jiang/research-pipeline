from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
)
from research_pipeline.briefing.registry import (
    SourceRegistry,
    assert_phase_a_source_boundary,
    load_source_registry,
)


def test_load_source_registry_none_returns_empty_registry() -> None:
    registry = load_source_registry(None)

    assert isinstance(registry, SourceRegistry)
    assert registry.sources == ()


def test_load_source_registry_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.json"

    with pytest.raises(FileNotFoundError, match="source registry not found"):
        load_source_registry(missing)


def test_source_registry_rejects_duplicate_source_ids() -> None:
    source = BriefingSourceConfig(
        source_id="github.valid",
        source_name="GitHub Valid",
        source_class=SourceClass.PRIMARY_ARTIFACT,
        access_method=AccessMethod.GITHUB_RELEASES,
        fixture_path="fixtures/gh.json",
        enabled=True,
    )

    with pytest.raises(ValidationError, match="source IDs must be unique"):
        SourceRegistry(sources=(source, source))


def test_load_source_registry_json_and_toml(tmp_path: Path) -> None:
    json_path = tmp_path / "registry.json"
    json_path.write_text(
        """
{
  "sources": [
    {
      "source_id": "github.valid",
      "source_name": "GitHub Valid",
      "source_class": "primary_artifact",
      "access_method": "github_releases",
      "fixture_path": "fixtures/gh.json",
      "enabled": true
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    toml_path = tmp_path / "registry.toml"
    toml_path.write_text(
        """
[briefing]
max_sources_per_run = 1

[[briefing.sources]]
source_id = "rss.valid"
source_name = "RSS Valid"
source_class = "technical_discussion"
access_method = "rss_atom"
fixture_path = "fixtures/rss.xml"
enabled = true
""".strip(),
        encoding="utf-8",
    )

    json_registry = load_source_registry(json_path)
    toml_registry = load_source_registry(toml_path)

    assert len(json_registry.sources) == 1
    assert json_registry.sources[0].source_id == "github.valid"
    assert len(toml_registry.sources) == 1
    assert toml_registry.sources[0].source_id == "rss.valid"


def test_enabled_sources_respects_max_sources_per_run() -> None:
    source_a = BriefingSourceConfig(
        source_id="source.a",
        source_name="Source A",
        source_class=SourceClass.PRIMARY_ARTIFACT,
        access_method=AccessMethod.GITHUB_RELEASES,
        fixture_path="fixtures/a.json",
        enabled=True,
    )
    source_b = BriefingSourceConfig(
        source_id="source.b",
        source_name="Source B",
        source_class=SourceClass.TECHNICAL_DISCUSSION,
        access_method=AccessMethod.RSS_ATOM,
        fixture_path="fixtures/b.xml",
        enabled=True,
    )

    registry = SourceRegistry(sources=(source_a, source_b), max_sources_per_run=1)

    assert [source.source_id for source in registry.enabled_sources()] == ["source.a"]


def test_phase_a_boundary_rejects_unreviewed_non_phase_a_source() -> None:
    source = BriefingSourceConfig(
        source_id="hn.new",
        source_name="Hacker News",
        source_class=SourceClass.TECHNICAL_DISCUSSION,
        access_method=AccessMethod.HACKER_NEWS,
        query="mcp",
        enabled=True,
    )

    with pytest.raises(ValueError, match="source expansion requires"):
        assert_phase_a_source_boundary(source)


def test_phase_a_boundary_allows_reviewed_non_phase_a_source() -> None:
    source = BriefingSourceConfig(
        source_id="hn.reviewed",
        source_name="Hacker News Reviewed",
        source_class=SourceClass.TECHNICAL_DISCUSSION,
        access_method=AccessMethod.HACKER_NEWS,
        query="mcp",
        enabled=True,
        last_reviewed_at="2026-04-29",
    )

    assert_phase_a_source_boundary(source)
