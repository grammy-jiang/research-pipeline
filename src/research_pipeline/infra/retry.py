"""Bounded retries with server-directed waiting and safe diagnostics."""

import functools
import logging
import math
import random
import time
from collections.abc import Callable
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


def parse_retry_after(header: str | None, *, now: float | None = None) -> float | None:
    """Parse delay seconds or an HTTP date without accepting invalid delays."""
    if not isinstance(header, str):
        return None
    try:
        delay = float(header)
    except ValueError:
        try:
            deadline = parsedate_to_datetime(header)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            return max(
                0.0, deadline.timestamp() - (time.time() if now is None else now)
            )
        except (ValueError, TypeError, OverflowError):
            return None
    return delay if math.isfinite(delay) and delay >= 0 else None


def _get_retry_after(exc: BaseException) -> float | None:
    """Return the response's waiting requirement, including HTTP-date values."""
    response: requests.Response | None = getattr(exc, "response", None)
    if response is None:
        return None
    return parse_retry_after(response.headers.get("Retry-After"))


def safe_exception(exc: BaseException) -> str:
    """Describe a request failure without echoing URLs, query data or secrets."""
    response: requests.Response | None = getattr(exc, "response", None)
    if response is None:
        return type(exc).__name__
    status = getattr(response, "status_code", None)
    url = getattr(response, "url", "")
    host = urlsplit(url if isinstance(url, str) else "").hostname or "unknown host"
    return f"{type(exc).__name__}: HTTP {status} from {host}"


def retryable_error(exc: BaseException) -> bool:
    """Separate transient failures from requests that need correction."""
    if getattr(exc, "retryable", True) is False:
        return False
    if isinstance(exc, requests.exceptions.JSONDecodeError):
        return False
    response: requests.Response | None = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500:
        return status in (408, 425, 429)
    return not (isinstance(status, int) and status in (501, 505))


def retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    jitter_fraction: float = 0.25,
    retryable_exceptions: tuple[type[BaseException], ...] = (
        requests.RequestException,
        requests.Timeout,
    ),
) -> Callable[[F], F]:
    """Retry transient failures; never shorten a server waiting deadline."""
    if max_attempts < 1 or backoff_base < 0 or not 0 <= jitter_fraction <= 1:
        raise ValueError("Invalid retry attempts, backoff or jitter")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    if not retryable_error(exc) or attempt == max_attempts:
                        logger.warning(
                            "Request stopped for %s after %d attempt(s): %s",
                            getattr(func, "__qualname__", type(func).__name__),
                            attempt,
                            safe_exception(exc),
                        )
                        raise
                    delay = backoff_base ** (attempt - 1)
                    jitter = delay * jitter_fraction
                    delay = max(0.0, delay + random.uniform(-jitter, jitter))  # nosec B311
                    retry_after = _get_retry_after(exc)
                    if retry_after is not None:
                        delay = max(delay, retry_after)
                    logger.info(
                        "Retry %d/%d for %s after %.2fs: %s",
                        attempt,
                        max_attempts,
                        getattr(func, "__qualname__", type(func).__name__),
                        delay,
                        safe_exception(exc),
                    )
                    time.sleep(delay)
            raise RuntimeError("Retry loop ended without an outcome")

        return wrapper  # type: ignore[return-value]

    return decorator
