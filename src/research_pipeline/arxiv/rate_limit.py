"""Global rate limiter for arXiv API compliance.

arXiv requires at minimum 3 seconds between requests and a single connection.
This module enforces a configurable minimum interval (default 5s, hard floor 3s).
"""

import logging

from research_pipeline.infra.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

_HARD_FLOOR_SECONDS = 3.0


class ArxivRateLimiter(RateLimiter):
    """Thread-safe rate limiter for arXiv requests.

    Extends the generic ``RateLimiter`` with a hard floor of 3 seconds,
    as required by arXiv's API terms.
    """

    def __init__(self, min_interval: float = 5.0) -> None:
        if min_interval < _HARD_FLOOR_SECONDS:
            logger.warning(
                "Requested interval %.1fs is below hard floor %.1fs; clamping to %.1fs",
                min_interval,
                _HARD_FLOOR_SECONDS,
                _HARD_FLOOR_SECONDS,
            )
            min_interval = _HARD_FLOOR_SECONDS
        super().__init__(min_interval=min_interval, name="arxiv")
