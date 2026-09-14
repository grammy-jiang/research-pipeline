"""HTTP sessions with source budgets and credential-safe attempt diagnostics."""

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from research_pipeline import __version__
from research_pipeline.config.defaults import DEFAULT_SOURCE_INTERVAL
from research_pipeline.infra.request_budget import SharedRequestBudget, SourceCooldown
from research_pipeline.infra.retry import parse_retry_after, safe_exception

logger = logging.getLogger(__name__)
_USER_AGENT = f"research-pipeline/{__version__}"
_DEFAULT_RATE_LIMIT_COOLDOWN = 15 * 60.0


class SourceSession(requests.Session):
    """Serialize default source clients across processes and stop on HTTP 429.

    Injected custom sessions remain the caller's responsibility. The default
    state directory is independent of per-run artifact/cache directories.
    """

    def __init__(
        self,
        source: str,
        min_interval: float = DEFAULT_SOURCE_INTERVAL,
        state_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.budget = SharedRequestBudget(source, min_interval, state_dir)
        self.headers["User-Agent"] = _USER_AGENT

    def request(
        self, method: str | bytes, url: str | bytes, *args: Any, **kwargs: Any
    ) -> requests.Response:
        raw_url = url.decode() if isinstance(url, bytes) else url
        parsed = urlsplit(raw_url)
        attempt_id = uuid.uuid4().hex
        base = {
            "source": self.source,
            "attempt_id": attempt_id,
            "host": parsed.hostname,
            "method": (
                method.decode() if isinstance(method, bytes) else method
            ).upper(),
        }

        def record(event: str, **fields: Any) -> None:
            logger.info(
                "source_http %s",
                json.dumps(
                    {
                        **base,
                        "event": event,
                        "at": datetime.now(UTC).isoformat(),
                        **fields,
                    },
                    sort_keys=True,
                ),
            )

        try:
            with self.budget.slot():
                record("dispatch", min_interval_seconds=self.budget.min_interval)
                started = time.monotonic()
                try:
                    response = super().request(method, url, *args, **kwargs)
                except requests.RequestException as exc:
                    record(
                        "transport_failure",
                        duration_seconds=time.monotonic() - started,
                        error_type=type(exc).__name__,
                    )
                    raise type(exc)(
                        safe_exception(exc),
                        response=getattr(exc, "response", None),
                        request=getattr(exc, "request", None),
                    ) from None
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                content_type = response.headers.get("Content-Type", "").split(";")[0]
                # Only recognized media types are retained, never arbitrary values.
                if content_type not in {
                    "application/json",
                    "application/atom+xml",
                    "application/xml",
                    "text/xml",
                    "text/html",
                    "text/plain",
                    "application/pdf",
                }:
                    content_type = "unknown"
                record(
                    "response",
                    duration_seconds=time.monotonic() - started,
                    status=response.status_code,
                    content_type=content_type[:80],
                    retry_after_seconds=retry_after,
                    retry_after_present="Retry-After" in response.headers,
                )
                if response.status_code == 429:
                    delay = (
                        retry_after
                        if retry_after is not None
                        else _DEFAULT_RATE_LIMIT_COOLDOWN
                    )
                    raise self.budget.defer(delay, response)
                return response
        except SourceCooldown as exc:
            record("deferred", not_before=exc.not_before)
            raise


def create_session(
    contact_email: str = "", min_interval: float = DEFAULT_SOURCE_INTERVAL
) -> requests.Session:
    """Create a coordinated arXiv session without logging contact information."""
    session = SourceSession("arxiv", max(3.0, min_interval))
    if contact_email:
        session.headers["User-Agent"] = f"{_USER_AGENT} (contact: {contact_email})"
    logger.info("Created HTTP session with User-Agent: %s", _USER_AGENT)
    return session


@contextmanager
def sdk_operation(budget: SharedRequestBudget) -> Iterator[None]:
    """Coordinate SDK operations; an SDK may hide additional internal requests."""
    with budget.slot():
        logger.info("source_sdk source=%s event=dispatch", budget.source)
        try:
            yield
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                delay = parse_retry_after(exc.response.headers.get("Retry-After"))
                raise budget.defer(
                    delay if delay is not None else _DEFAULT_RATE_LIMIT_COOLDOWN,
                    exc.response,
                ) from None
            raise
