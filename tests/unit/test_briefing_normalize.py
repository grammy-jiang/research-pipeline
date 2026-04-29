from __future__ import annotations

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
)
from research_pipeline.briefing.normalize import (
    canonicalize_url,
    cluster_id_for,
    content_hash_for,
    dedup_key_for,
    event_id_for,
    normalize_title,
    stable_hash,
)


def _github_source() -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="github.example",
        source_name="GitHub Example",
        source_class=SourceClass.PRIMARY_ARTIFACT,
        access_method=AccessMethod.GITHUB_RELEASES,
        repo_owner="example",
        repo_name="repo",
        fixture_path="fixtures/gh.json",
        enabled=True,
    )


def _rss_source() -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="rss.example",
        source_name="RSS Example",
        source_class=SourceClass.TECHNICAL_DISCUSSION,
        access_method=AccessMethod.RSS_ATOM,
        fixture_path="fixtures/rss.xml",
        enabled=True,
    )


def _api_source() -> BriefingSourceConfig:
    return BriefingSourceConfig(
        source_id="api.example",
        source_name="API Example",
        source_class=SourceClass.IMPLEMENTATION_SOURCE,
        access_method=AccessMethod.API,
        enabled=True,
    )


def test_normalize_title_and_canonicalize_url() -> None:
    assert normalize_title("  New   Release   Notes  ") == "new release notes"
    assert (
        canonicalize_url("HTTPS://Example.COM/foo/?utm_source=x&b=2&a=1#frag")
        == "https://example.com/foo?a=1&b=2"
    )


def test_stable_hash_and_event_id_are_deterministic() -> None:
    value_a = stable_hash("part-a", "part-b", prefix="id_")
    value_b = stable_hash("part-a", "part-b", prefix="id_")
    value_c = stable_hash("part-a", "different", prefix="id_")

    assert value_a == value_b
    assert value_a != value_c

    source = _github_source()
    event_a = event_id_for(source, "https://example.com/a", "native-1")
    event_b = event_id_for(source, "https://example.com/a", "native-1")
    fallback_event = event_id_for(source, "https://example.com/a", None)

    assert event_a == event_b
    assert event_a.startswith("event_")
    assert fallback_event.startswith("event_")
    assert fallback_event != event_a


def test_content_hash_changes_when_content_changes() -> None:
    hash_a = content_hash_for(
        "Release",
        "https://example.com/release",
        "2026-04-29T00:00:00Z",
        "summary-a",
    )
    hash_b = content_hash_for(
        "Release",
        "https://example.com/release",
        "2026-04-29T00:00:00Z",
        "summary-a",
    )
    hash_c = content_hash_for(
        "Release",
        "https://example.com/release",
        "2026-04-29T00:00:00Z",
        "summary-b",
    )

    assert hash_a == hash_b
    assert hash_a != hash_c


def test_dedup_key_precedence_for_github_and_rss() -> None:
    github_key = dedup_key_for(
        source=_github_source(),
        title="Release",
        canonical_url="https://example.com/release",
        source_native_id="native-1",
        identifiers={"repo": "example/repo", "tag": "v1.0.0"},
    )
    rss_key = dedup_key_for(
        source=_rss_source(),
        title="Feed Item",
        canonical_url="https://example.com/feed/item",
        source_native_id="guid-1",
    )

    assert github_key == "github-release:example/repo:v1.0.0"
    assert rss_key == "rss-guid:rss.example:guid-1"


def test_dedup_key_fallback_order_and_cluster_id() -> None:
    source = _api_source()

    native_key = dedup_key_for(
        source=source,
        title="API Entry",
        canonical_url="https://example.com/api/item",
        source_native_id="native-1",
    )
    url_key = dedup_key_for(
        source=source,
        title="API Entry",
        canonical_url="https://example.com/api/item",
        source_native_id=None,
    )
    title_key = dedup_key_for(
        source=source,
        title="  API   Entry  ",
        canonical_url="",
        source_native_id=None,
    )

    assert native_key == "native:api.example:native-1"
    assert url_key == "url:https://example.com/api/item"
    assert title_key == "title:api entry"

    cluster_a = cluster_id_for(native_key)
    cluster_b = cluster_id_for(native_key)

    assert cluster_a == cluster_b
    assert cluster_a.startswith("cluster_")
