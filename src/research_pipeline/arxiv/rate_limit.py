"""Global rate limiter for arXiv API compliance.

arXiv requires at minimum 3 seconds between requests and a single connection.
This module enforces a configurable minimum interval (default 5s, hard floor 3s).
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

_HARD_FLOOR_SECONDS = 3.0


class ArxivRateLimiter:
    """Thread-safe global rate limiter for arXiv requests.

    Enforces a minimum interval between consecutive requests.
    The interval can never be set below the hard floor of 3 seconds.
    """

    def __init__(self, min_interval: float = 5.0) -> None:
        if min_interval < _HARD_FLOOR_SECONDS:
            logger.warning(
                "Requested interval %.1fs is below hard floor %.1fs; "
                "clamping to %.1fs",
                min_interval,
                _HARD_FLOOR_SECONDS,
                _HARD_FLOOR_SECONDS,
            )
            min_interval = _HARD_FLOOR_SECONDS
        self.min_interval = min_interval
        self._last_request_time: float = 0.0
        self._lock = threading.Lock()

    def wait(self) -> float:
        """Block until enough time has elapsed since the last request.

        Returns:
            The number of seconds actually waited (0 if no wait was needed).
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            wait_time = self.min_interval - elapsed
            if wait_time > 0:
                logger.debug("Rate limiter: waiting %.2fs", wait_time)
                time.sleep(wait_time)
                self._last_request_time = time.monotonic()
                return wait_time
            self._last_request_time = now
            return 0.0
