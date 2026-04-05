"""Retry decorator with exponential backoff, jitter, and Retry-After support.

Provides a ``@retry`` decorator for resilient network calls.  It retries on
configurable exception types, backs off exponentially with jitter, and
respects the ``Retry-After`` HTTP header when a ``requests.Response`` is
available on the exception.
"""

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

import requests

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _get_retry_after(exc: BaseException) -> float | None:
    """Extract ``Retry-After`` seconds from a requests exception, if present."""
    response: requests.Response | None = getattr(exc, "response", None)
    if response is None:
        return None
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return float(header)
    except (ValueError, TypeError):
        return None


def retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    jitter_fraction: float = 0.25,
    retryable_exceptions: tuple[type[BaseException], ...] = (
        requests.RequestException,
        requests.Timeout,
    ),
) -> Callable[[F], F]:
    """Decorator that retries a function on transient failures.

    Args:
        max_attempts: Total number of attempts (including the first).
        backoff_base: Base for exponential backoff (seconds).
        jitter_fraction: Fraction of the computed delay to randomize (±).
        retryable_exceptions: Exception types that trigger a retry.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.warning(
                            "Retry exhausted for %s after %d attempts: %s",
                            func.__qualname__,
                            max_attempts,
                            exc,
                        )
                        raise

                    # Compute delay: exponential backoff
                    delay = backoff_base ** (attempt - 1)

                    # Respect Retry-After header if present
                    retry_after = _get_retry_after(exc)
                    if retry_after is not None and retry_after > delay:
                        delay = retry_after

                    # Add jitter
                    jitter = delay * jitter_fraction
                    delay += random.uniform(-jitter, jitter)  # noqa: S311  # nosec B311
                    delay = max(0.0, delay)

                    logger.info(
                        "Retry %d/%d for %s after %.2fs: %s",
                        attempt,
                        max_attempts,
                        func.__qualname__,
                        delay,
                        exc,
                    )
                    time.sleep(delay)

            # Should not reach here, but satisfy type checker
            assert last_exc is not None  # noqa: S101  # nosec B101
            raise last_exc

        return wrapper  # type: ignore[return-value]

    return decorator
