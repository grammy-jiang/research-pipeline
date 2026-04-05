"""Unit tests for infra.rate_limit — generic RateLimiter."""

import time
from unittest.mock import patch

from research_pipeline.infra.rate_limit import RateLimiter


class TestRateLimiter:
    """Tests for the generic RateLimiter class."""

    def test_default_interval(self) -> None:
        limiter = RateLimiter()
        assert limiter.min_interval == 1.0

    def test_custom_interval(self) -> None:
        limiter = RateLimiter(min_interval=0.5)
        assert limiter.min_interval == 0.5

    def test_zero_interval(self) -> None:
        limiter = RateLimiter(min_interval=0.0)
        assert limiter.min_interval == 0.0

    def test_negative_interval_clamped_to_zero(self) -> None:
        limiter = RateLimiter(min_interval=-1.0)
        assert limiter.min_interval == 0.0

    def test_name_stored(self) -> None:
        limiter = RateLimiter(name="test-service")
        assert limiter.name == "test-service"

    def test_first_call_no_wait(self) -> None:
        limiter = RateLimiter(min_interval=1.0)
        waited = limiter.wait()
        assert waited == 0.0

    def test_second_call_waits(self) -> None:
        limiter = RateLimiter(min_interval=2.0)
        limiter._last_request_time = time.monotonic()
        with patch("time.sleep") as mock_sleep:
            limiter.wait()
        assert mock_sleep.called
        call_args = mock_sleep.call_args[0][0]
        assert call_args > 1.5

    def test_no_wait_when_enough_time_elapsed(self) -> None:
        limiter = RateLimiter(min_interval=1.0)
        limiter._last_request_time = time.monotonic() - 10.0
        waited = limiter.wait()
        assert waited == 0.0

    def test_no_hard_floor(self) -> None:
        """Generic limiter should NOT have a hard floor, unlike ArxivRateLimiter."""
        limiter = RateLimiter(min_interval=0.1)
        assert limiter.min_interval == 0.1
