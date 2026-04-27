"""Normalization and stable ID helpers for briefing events."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from research_pipeline.briefing.models import AccessMethod, BriefingSourceConfig

_WHITESPACE_RE = re.compile(r"\s+")
_NON_TOPIC_RE = re.compile(r"[^a-z0-9]+")


def utc_now_iso() -> str:
    """Return a UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def today_utc() -> str:
    """Return the current UTC date."""
    return datetime.now(UTC).date().isoformat()


def stable_hash(*parts: object, prefix: str = "", length: int = 16) -> str:
    """Hash normalized parts with SHA-256 and return a stable short ID."""
    joined = "\x1f".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}" if prefix else digest


def normalize_title(title: str) -> str:
    """Normalize a title for matching and hashing."""
    return _WHITESPACE_RE.sub(" ", title.strip().lower())


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL while preserving meaningful query parameters."""
    split = urlsplit(url.strip())
    scheme = split.scheme.lower() or "https"
    netloc = split.netloc.lower()
    path = re.sub(r"/{2,}", "/", split.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(split.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def topic_id_for_title(title: str) -> str:
    """Create a deterministic coarse topic ID from a title."""
    tokens = [
        token
        for token in _NON_TOPIC_RE.sub(" ", normalize_title(title)).split()
        if len(token) > 2
    ][:5]
    slug = "-".join(tokens) or "general"
    return f"topic_{slug[:64]}"


def event_id_for(
    source: BriefingSourceConfig,
    canonical_url: str,
    source_native_id: str | None,
) -> str:
    """Generate a stable event ID following the Phase A policy."""
    key = source_native_id or canonical_url
    return stable_hash(source.source_id, key, prefix="event_")


def content_hash_for(
    title: str,
    canonical_url: str,
    published_at: str | None,
    summary_hint: str,
) -> str:
    """Generate a stable event content hash."""
    return stable_hash(
        normalize_title(title), canonical_url, published_at, summary_hint
    )


def dedup_key_for(
    *,
    source: BriefingSourceConfig,
    title: str,
    canonical_url: str,
    source_native_id: str | None,
    identifiers: dict[str, str] | None = None,
) -> str:
    """Select the strongest exact deduplication key available."""
    identifiers = identifiers or {}
    if source.access_method == AccessMethod.GITHUB_RELEASES:
        repo = identifiers.get("repo")
        tag = identifiers.get("tag")
        if repo and tag:
            return f"github-release:{repo}:{tag}".lower()
    if source.access_method == AccessMethod.RSS_ATOM and source_native_id:
        return f"rss-guid:{source.source_id}:{source_native_id}"
    if source_native_id:
        return f"native:{source.source_id}:{source_native_id}"
    if canonical_url:
        return f"url:{canonical_url}"
    return f"title:{normalize_title(title)}"


def cluster_id_for(dedup_key: str) -> str:
    """Generate a stable cluster ID from a primary dedup key."""
    return stable_hash(dedup_key, prefix="cluster_")
