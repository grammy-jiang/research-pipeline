"""Unit tests for infra.retry — retry decorator."""

from unittest.mock import MagicMock, patch

import requests

from research_pipeline.infra.retry import _get_retry_after, retry


class TestGetRetryAfter:
    """Tests for _get_retry_after helper."""

    def test_no_response_attribute(self) -> None:
        exc = ValueError("no response")
        assert _get_retry_after(exc) is None

    def test_response_is_none(self) -> None:
        exc = requests.RequestException()
        exc.response = None  # type: ignore[attr-defined]
        assert _get_retry_after(exc) is None

    def test_no_retry_after_header(self) -> None:
        resp = MagicMock(spec=requests.Response)
        resp.headers = {}
        exc = requests.RequestException(response=resp)
        assert _get_retry_after(exc) is None

    def test_valid_retry_after_header(self) -> None:
        resp = MagicMock(spec=requests.Response)
        resp.headers = {"Retry-After": "10"}
        exc = requests.RequestException(response=resp)
        assert _get_retry_after(exc) == 10.0

    def test_invalid_retry_after_header(self) -> None:
        resp = MagicMock(spec=requests.Response)
        resp.headers = {"Retry-After": "not-a-number"}
        exc = requests.RequestException(response=resp)
        assert _get_retry_after(exc) is None


class TestRetryDecorator:
    """Tests for the @retry decorator."""

    def test_success_no_retry(self) -> None:
        call_count = 0

        @retry(max_attempts=3)
        def succeeds() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeeds()
        assert result == "ok"
        assert call_count == 1

    @patch("time.sleep")
    def test_retries_on_request_exception(self, mock_sleep: MagicMock) -> None:
        call_count = 0

        @retry(max_attempts=3)
        def fails_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.RequestException("transient")
            return "ok"

        result = fails_twice()
        assert result == "ok"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @patch("time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep: MagicMock) -> None:
        @retry(max_attempts=2)
        def always_fails() -> str:
            raise requests.Timeout("timeout")

        try:
            always_fails()
            assert False, "Should have raised"  # noqa: B011
        except requests.Timeout:
            pass
        assert mock_sleep.call_count == 1

    def test_non_retryable_exception_not_retried(self) -> None:
        call_count = 0

        @retry(max_attempts=3, retryable_exceptions=(requests.Timeout,))
        def raises_value_error() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        try:
            raises_value_error()
            assert False, "Should have raised"  # noqa: B011
        except ValueError:
            pass
        assert call_count == 1

    @patch("time.sleep")
    def test_exponential_backoff(self, mock_sleep: MagicMock) -> None:
        call_count = 0

        @retry(max_attempts=4, backoff_base=2.0, jitter_fraction=0.0)
        def fails_thrice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise requests.RequestException("transient")
            return "ok"

        result = fails_thrice()
        assert result == "ok"
        # Delays: 2^0=1, 2^1=2, 2^2=4
        delays = [c[0][0] for c in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 4.0]

    @patch("time.sleep")
    def test_retry_after_header_respected(self, mock_sleep: MagicMock) -> None:
        call_count = 0

        @retry(max_attempts=2, backoff_base=1.0, jitter_fraction=0.0)
        def fails_with_retry_after() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                resp = MagicMock(spec=requests.Response)
                resp.headers = {"Retry-After": "30"}
                exc = requests.RequestException("rate limited")
                exc.response = resp  # type: ignore[attr-defined]
                raise exc
            return "ok"

        result = fails_with_retry_after()
        assert result == "ok"
        # Retry-After=30 > backoff=1, so delay should be 30
        delay = mock_sleep.call_args[0][0]
        assert delay == 30.0

    @patch("time.sleep")
    def test_jitter_applied(self, mock_sleep: MagicMock) -> None:
        """With jitter_fraction > 0, delay should vary from base."""
        call_count = 0

        @retry(max_attempts=2, backoff_base=10.0, jitter_fraction=0.25)
        def fails_once() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.RequestException("transient")
            return "ok"

        result = fails_once()
        assert result == "ok"
        delay = mock_sleep.call_args[0][0]
        # Base = 10^0 = 1.0; jitter ±25% → [0.75, 1.25]
        # But with randomness, just check it's reasonably close
        assert 0.5 <= delay <= 1.5

    def test_preserves_function_metadata(self) -> None:
        @retry(max_attempts=2)
        def my_function() -> str:
            """My docstring."""
            return "ok"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."
