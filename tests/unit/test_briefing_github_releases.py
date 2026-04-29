from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
)
from research_pipeline.briefing.sources.github_releases import GitHubReleasesSource

FIXTURE_BASE = Path(__file__).parents[1] / "fixtures" / "briefing"


def _source(fixture_name: str) -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="github.example",
        source_name="GitHub Example",
        source_class=SourceClass.PRIMARY_ARTIFACT,
        access_method=AccessMethod.GITHUB_RELEASES,
        repo_owner="example",
        repo_name="repo",
        fixture_path=f"github/{fixture_name}",
        max_events_per_run=10,
        enabled=True,
    )


def test_github_releases_normal_fixture_maps_to_events() -> None:
    source = _source("releases_normal.json")
    adapter = GitHubReleasesSource(source, fixture_base_dir=FIXTURE_BASE)

    events = adapter.poll()

    assert len(events) == 2
    first = events[0]
    assert first.source_id == "github.example"
    assert first.item_type == "github_release"
    assert first.canonical_url == "https://github.com/example/repo/releases/tag/v1.2.3"
    assert first.identifiers["repo"] == "example/repo"
    assert first.identifiers["tag"] == "v1.2.3"
    assert first.source_native_id == "1001"


def test_github_releases_empty_fixture_returns_no_events() -> None:
    source = _source("releases_empty.json")
    adapter = GitHubReleasesSource(source, fixture_base_dir=FIXTURE_BASE)

    assert adapter.poll() == []


def test_github_releases_malformed_fixture_is_rejected() -> None:
    source = _source("releases_malformed.json")
    adapter = GitHubReleasesSource(source, fixture_base_dir=FIXTURE_BASE)

    with pytest.raises(ValueError, match="JSON array"):
        adapter.poll()


def test_github_releases_rate_limited_fixture_is_rejected() -> None:
    source = _source("releases_rate_limited.json")
    adapter = GitHubReleasesSource(source, fixture_base_dir=FIXTURE_BASE)

    with pytest.raises(ValueError, match="JSON array"):
        adapter.poll()
