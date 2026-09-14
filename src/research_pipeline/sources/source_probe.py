"""Gentle, anonymous reachability checks for fixed public search endpoints."""

import html
import json
import logging
import socket
import time
from collections.abc import Callable
from datetime import UTC, datetime

import requests
from lxml import etree  # type: ignore[import-untyped]
from requests.auth import AuthBase
from urllib3.exceptions import NameResolutionError, ReadTimeoutError

from research_pipeline.config.models import PipelineConfig
from research_pipeline.infra.http import SourceSession
from research_pipeline.infra.request_budget import SourceCooldown
from research_pipeline.infra.retry import parse_retry_after
from research_pipeline.models.source_probe import (
    ProbeReport,
    ProbeResult,
    ProbeSource,
    ProbeStatus,
)

logger = logging.getLogger(__name__)
_MIN_SPACING = 30.0
_MAX_BODY = 65_536
_TARGETS: dict[ProbeSource, tuple[str, dict[str, str]]] = {
    ProbeSource.ARXIV: (
        "https://export.arxiv.org/api/query",
        {"search_query": "all:email", "start": "0", "max_results": "1"},
    ),
    ProbeSource.SEMANTIC_SCHOLAR: (
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {"query": "email", "limit": "1", "fields": "title"},
    ),
    ProbeSource.DBLP: (
        "https://dblp.org/search/publ/api",
        {"q": "email", "format": "json", "h": "1"},
    ),
    ProbeSource.OPENALEX: (
        "https://api.openalex.org/works",
        {"search": "email", "per-page": "1", "select": "id,title"},
    ),
}


class _AnonymousAuth(AuthBase):
    """Suppress implicit netrc credentials while retaining normal network routing."""

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        return request


def _item_count(source: ProbeSource, body: bytes) -> int | None:
    """Accept expected search shapes, including a valid empty result."""
    if source == ProbeSource.ARXIV:
        try:
            root = etree.fromstring(
                body, parser=etree.XMLParser(resolve_entities=False, no_network=True)
            )
        except (etree.XMLSyntaxError, ValueError):
            return None
        atom = "{http://www.w3.org/2005/Atom}"
        if root.tag != f"{atom}feed":
            return None
        entries = root.findall(f"{atom}entry")
        identifiers = [entry.findtext(f"{atom}id", "") for entry in entries]
        if any(not value or "/api/errors" in value for value in identifiers):
            return None
        return len(entries)
    try:
        data = json.loads(body)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(data, dict):
        return None
    if source == ProbeSource.DBLP:
        result = data.get("result")
        hits = result.get("hits") if isinstance(result, dict) else None
        if not isinstance(hits, dict):
            return None
        total = str(hits.get("@total", ""))
        items = hits.get("hit", [])
        if not total.isdigit() or not isinstance(items, list):
            return None
        if int(total) > 0 and not items:
            return None
        return len(items) if all(isinstance(item, dict) for item in items) else None
    key = "data" if source == ProbeSource.SEMANTIC_SCHOLAR else "results"
    id_key = "paperId" if source == ProbeSource.SEMANTIC_SCHOLAR else "id"
    items = data.get(key)
    if not isinstance(items, list) or any(
        not isinstance(item, dict) or not item.get(id_key) for item in items
    ):
        return None
    return len(items)


def _classify(
    source: ProbeSource, status: int, body: bytes, truncated: bool
) -> tuple[ProbeStatus, int | None]:
    text = html.unescape(body.decode("utf-8", errors="replace")).lower()
    is_html = "<html" in text or "<!doctype html" in text
    if status == 429:
        return "rate_limited", None
    if 300 <= status < 400:
        return "redirect", None
    if is_html and any(
        marker in text
        for marker in (
            "not a bot",
            "verify you are human",
            "verify that you are human",
            "checking your browser",
            "cf-chl-",
            "captcha",
            "anubis",
        )
    ):
        return "bot_challenge", None
    if status == 401 or (
        status == 403
        and any(
            marker in text
            for marker in (
                "api key required",
                "api_key required",
                "authentication required",
            )
        )
    ):
        return "auth_required", None
    if status == 403:
        return "forbidden", None
    if not 200 <= status < 300:
        return "http_error", None
    count = None if truncated or is_html else _item_count(source, body)
    return ("ok", count) if count is not None else ("unexpected_response", None)


def _transport_status(exc: requests.RequestException) -> ProbeStatus:
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.SSLError):
        return "tls_error"
    # requests/urllib3 wrap DNS errors; SourceSession also preserves the original
    # exception as context when redacting its text. Inspect types, not raw strings.
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending and len(seen) < 32:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ReadTimeoutError):
            return "timeout"
        if isinstance(current, (socket.gaierror, NameResolutionError)):
            return "dns_error"
        for nested in (
            current.__cause__,
            current.__context__,
            getattr(current, "reason", None),
            *current.args,
        ):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return "connection_error"


def _cooldown_result(
    source: ProbeSource,
    endpoint: str,
    checked_at: datetime,
    exc: SourceCooldown,
    elapsed: float = 0,
) -> ProbeResult:
    response = exc.response
    attempted = response is not None
    retry_after = (
        parse_retry_after(response.headers.get("Retry-After"))
        if response is not None
        else None
    )
    if response is not None:
        response.close()
    return ProbeResult(
        source=source,
        endpoint=endpoint,
        checked_at=checked_at,
        status="rate_limited" if attempted else "cooldown",
        http_status=response.status_code if response is not None else None,
        request_sent=attempted,
        received_http_response=attempted,
        elapsed_seconds=elapsed,
        retry_after_seconds=retry_after,
        cooldown_until=datetime.fromtimestamp(exc.not_before, UTC),
    )


def _probe_one(
    session: SourceSession,
    source: ProbeSource,
    endpoint: str,
    params: dict[str, str],
) -> ProbeResult:
    checked_at = datetime.now(UTC)
    started = time.monotonic()
    response: requests.Response | None = None
    body = bytearray()
    truncated = False
    try:
        response = session.get(
            endpoint,
            params=params,
            timeout=(10, 15),
            stream=True,
            allow_redirects=False,
        )
        # Redirects are observations, never another request.
        if not 300 <= response.status_code < 400:
            for chunk in response.iter_content(chunk_size=4096):
                available = _MAX_BODY - len(body)
                body.extend(chunk[:available])
                if len(chunk) > available:
                    truncated = True
                    break
        status, count = _classify(source, response.status_code, bytes(body), truncated)
        return ProbeResult(
            source=source,
            endpoint=endpoint,
            checked_at=checked_at,
            status=status,
            http_status=response.status_code,
            request_sent=True,
            received_http_response=True,
            elapsed_seconds=round(time.monotonic() - started, 3),
            response_bytes=len(body),
            body_truncated=truncated,
            returned_items=count,
            retry_after_seconds=parse_retry_after(response.headers.get("Retry-After")),
        )
    except SourceCooldown as exc:
        return _cooldown_result(
            source, endpoint, checked_at, exc, round(time.monotonic() - started, 3)
        )
    except requests.RequestException as exc:
        return ProbeResult(
            source=source,
            endpoint=endpoint,
            checked_at=checked_at,
            status=_transport_status(exc),
            http_status=response.status_code if response is not None else None,
            request_sent=True,
            received_http_response=response is not None,
            elapsed_seconds=round(time.monotonic() - started, 3),
            response_bytes=len(body),
            body_truncated=truncated,
        )
    finally:
        if response is not None:
            response.close()


def probe_sources(
    source: ProbeSource | str = ProbeSource.ALL,
    config: PipelineConfig | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> ProbeReport:
    """Make at most one request per source, preserving shared rate-limit state.

    All probes are anonymous. API keys and contact details from config are not
    sent. Active cooldowns skip network access; there are no retries, redirects
    or cooldown-reset switches. Calls across sources are sequential and spaced.
    """
    selected = ProbeSource(source)
    sources = list(_TARGETS) if selected == ProbeSource.ALL else [selected]
    config = config or PipelineConfig()
    intervals = {
        ProbeSource.ARXIV: config.arxiv.min_interval_seconds,
        ProbeSource.SEMANTIC_SCHOLAR: config.sources.semantic_scholar_min_interval,
        ProbeSource.DBLP: config.sources.dblp_min_interval,
        ProbeSource.OPENALEX: config.sources.openalex_min_interval,
    }
    started_at = datetime.now(UTC)
    last_attempt_end: float | None = None
    results: list[ProbeResult] = []

    def notify(done: int, message: str) -> None:
        logger.info("source_probe %s", message)
        if progress is not None:
            progress(done, len(sources), message)

    for index, name in enumerate(sources):
        endpoint, params = _TARGETS[name]
        with SourceSession(
            name.value, min_interval=max(_MIN_SPACING, intervals[name])
        ) as session:
            session.auth = _AnonymousAuth()
            try:
                session.budget.check_available()
            except SourceCooldown as exc:
                item = _cooldown_result(name, endpoint, datetime.now(UTC), exc)
            else:
                if last_attempt_end is not None:
                    delay = max(0.0, last_attempt_end + _MIN_SPACING - time.monotonic())
                    if delay:
                        notify(
                            index, f"Waiting {delay:.0f}s before checking {name.value}"
                        )
                        time.sleep(delay)
                notify(index, f"Checking {name.value}")
                item = _probe_one(session, name, endpoint, params)
                if item.request_sent:
                    last_attempt_end = time.monotonic()
        results.append(item)
        notify(index + 1, f"{name.value}: {item.status}")
    return ProbeReport(
        started_at=started_at, finished_at=datetime.now(UTC), results=results
    )
