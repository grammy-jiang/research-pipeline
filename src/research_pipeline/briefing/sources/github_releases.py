"""GitHub releases source adapter for daily briefings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    IntelligenceEvent,
)
from research_pipeline.briefing.normalize import (
    canonicalize_url,
    content_hash_for,
    dedup_key_for,
    event_id_for,
    topic_id_for_title,
    utc_now_iso,
)
from research_pipeline.briefing.sources.base import build_session, read_fixture_text


class GitHubReleasesSource:
    """Poll GitHub releases for one whitelisted repository."""

    def __init__(
        self,
        source: BriefingSourceConfig,
        *,
        state: dict[str, str] | None = None,
        fixture_base_dir: Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.source = source
        self.state = state if state is not None else {}
        self.fixture_base_dir = fixture_base_dir
        self.session = session or build_session()

    def poll(self) -> list[IntelligenceEvent]:
        """Poll releases and normalize them into events."""
        payload = self._load_payload()
        events = [
            self._event_from_release(item)
            for item in payload[: self.source.max_events_per_run]
        ]
        return events

    def _load_payload(self) -> list[dict[str, Any]]:
        fixture = read_fixture_text(self.source, self.fixture_base_dir)
        if fixture is not None:
            loaded = json.loads(fixture)
            if not isinstance(loaded, list):
                raise ValueError("GitHub release fixture must be a JSON array")
            return [item for item in loaded if isinstance(item, dict)]
        url = (
            str(self.source.api_url) if self.source.api_url else self._default_api_url()
        )
        headers: dict[str, str] = {}
        if etag := self.state.get("etag"):
            headers["If-None-Match"] = etag
        if last_modified := self.state.get("last_modified"):
            headers["If-Modified-Since"] = last_modified
        response = self.session.get(url, headers=headers, timeout=20)
        self.state["status_code"] = str(response.status_code)
        if response.status_code == 304:
            return []
        response.raise_for_status()
        if etag := response.headers.get("ETag"):
            self.state["etag"] = etag
        if last_modified := response.headers.get("Last-Modified"):
            self.state["last_modified"] = last_modified
        loaded = response.json()
        if not isinstance(loaded, list):
            raise ValueError("GitHub releases response must be a JSON array")
        return [item for item in loaded if isinstance(item, dict)]

    def _default_api_url(self) -> str:
        if self.source.repo_owner is None or self.source.repo_name is None:
            raise ValueError("repo_owner and repo_name are required")
        return (
            "https://api.github.com/repos/"
            f"{self.source.repo_owner}/{self.source.repo_name}/releases"
        )

    def _event_from_release(self, item: dict[str, Any]) -> IntelligenceEvent:
        release_label = str(
            item.get("name") or item.get("tag_name") or "GitHub release"
        )
        repo_label = (
            f"{self.source.repo_owner}/{self.source.repo_name}"
            if self.source.repo_owner and self.source.repo_name
            else self.source.source_name
        )
        title = (
            f"{repo_label} {release_label}"
            if release_label and repo_label not in release_label
            else release_label or repo_label
        )
        tag = str(item.get("tag_name") or "")
        html_url = str(
            item.get("html_url") or item.get("url") or self.source.official_url
        )
        canonical_url = canonicalize_url(html_url)
        published_at = item.get("published_at") or item.get("created_at")
        body = str(item.get("body") or "")
        identifiers = {
            "repo": f"{self.source.repo_owner}/{self.source.repo_name}",
            "tag": tag,
        }
        source_native_id = str(item.get("id") or tag or canonical_url)
        dedup_key = dedup_key_for(
            source=self.source,
            title=title,
            canonical_url=canonical_url,
            source_native_id=source_native_id,
            identifiers=identifiers,
        )
        content_hash = content_hash_for(title, canonical_url, published_at, body[:240])
        return IntelligenceEvent(
            event_id=event_id_for(self.source, canonical_url, source_native_id),
            source_name=self.source.source_name,
            source_id=self.source.source_id,
            source_type=self.source.source_class,
            item_type="github_release",
            canonical_url=canonical_url,
            title=title,
            retrieved_at=utc_now_iso(),
            collection_method=AccessMethod.GITHUB_RELEASES,
            content_hash=content_hash,
            dedup_key=dedup_key,
            author_or_org=self.source.repo_owner,
            published_at=str(published_at) if published_at else None,
            source_native_id=source_native_id,
            identifiers=identifiers,
            summary_hint=body[:1200],
            excerpt=body[:1200],
            topics=(topic_id_for_title(release_label or title),),
            artifact_links=(canonical_url,),
            confidence="high",
            raw_metadata={
                "draft": str(bool(item.get("draft", False))),
                "prerelease": str(bool(item.get("prerelease", False))),
            },
        )
