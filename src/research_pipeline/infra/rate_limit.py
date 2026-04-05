"""Generic, thread-safe rate limiter.

Provides a reusable ``RateLimiter`` that enforces a configurable minimum
interval between consecutive calls.  Unlike the arXiv-specific
``ArxivRateLimiter``, this class does **not** impose a hard floor — the
caller decides the appropriate interval for each service.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter with configurable minimum interval.

    Args:
        min_interval: Minimum seconds between consecutive calls.
            Must be non-negative.
        name: Optional label used in log messages.
    """

    def __init__(self, min_interval: float = 1.0, name: str = "") -> None:
        if min_interval < 0:
            min_interval = 0.0
        self.min_interval = min_interval
        self.name = name
        self._last_request_time: float = 0.0
        self._lock = threading.Lock()

    def wait(self) -> float:
        """Block until enough time has elapsed since the last call.

        Returns:
            The number of seconds actually waited (0 if no wait was needed).
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            wait_time = self.min_interval - elapsed
            if wait_time > 0:
                label = f" [{self.name}]" if self.name else ""
                logger.debug("Rate limiter%s: waiting %.2fs", label, wait_time)
                time.sleep(wait_time)
                self._last_request_time = time.monotonic()
                return wait_time
            self._last_request_time = now
            return 0.0
