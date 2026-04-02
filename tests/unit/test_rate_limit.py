"""Unit tests for arxiv.rate_limit module."""

import time
from unittest.mock import patch

from arxiv_paper_pipeline.arxiv.rate_limit import ArxivRateLimiter


class TestArxivRateLimiter:
    def test_hard_floor_clamping(self) -> None:
        limiter = ArxivRateLimiter(min_interval=1.0)
        assert limiter.min_interval == 3.0

    def test_custom_interval_above_floor(self) -> None:
        limiter = ArxivRateLimiter(min_interval=5.0)
        assert limiter.min_interval == 5.0

    def test_exact_floor(self) -> None:
        limiter = ArxivRateLimiter(min_interval=3.0)
        assert limiter.min_interval == 3.0

    def test_first_call_no_wait(self) -> None:
        limiter = ArxivRateLimiter(min_interval=3.0)
        waited = limiter.wait()
        assert waited == 0.0

    def test_second_call_computes_wait(self) -> None:
        """Verify that the wait logic computes a positive wait time."""
        limiter = ArxivRateLimiter(min_interval=3.0)
        # Simulate first call just happened
        limiter._last_request_time = time.monotonic()
        with patch("time.sleep") as mock_sleep:
            limiter.wait()
        # Should have attempted to sleep for ~3 seconds
        assert mock_sleep.called
        call_args = mock_sleep.call_args[0][0]
        assert call_args > 2.5

    def test_no_wait_when_enough_time_elapsed(self) -> None:
        limiter = ArxivRateLimiter(min_interval=3.0)
        # Simulate last request was long ago
        limiter._last_request_time = time.monotonic() - 10.0
        actual_wait = limiter.wait()
        assert actual_wait == 0.0

    def test_negative_interval_clamped(self) -> None:
        limiter = ArxivRateLimiter(min_interval=-1.0)
        assert limiter.min_interval == 3.0

    def test_zero_interval_clamped(self) -> None:
        limiter = ArxivRateLimiter(min_interval=0.0)
        assert limiter.min_interval == 3.0
