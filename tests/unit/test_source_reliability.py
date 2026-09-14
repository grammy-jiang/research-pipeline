"""Regression coverage for the source failures observed in msgloom Topic 01."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from research_pipeline.arxiv.client import ArxivClient
from research_pipeline.config.models import PipelineConfig
from research_pipeline.infra.retry import _get_retry_after, retry
from research_pipeline.sources.openalex_source import OpenAlexSource


def response_error(status: int, retry_after: str | None = None) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response.url = "https://api.example.invalid/search?api_key=redacted-fixture"
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return requests.HTTPError("fixture", response=response)


def test_retry_after_is_a_floor_even_with_negative_jitter() -> None:
    attempts = 0

    @retry(max_attempts=2)
    def request() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise response_error(429, "60")
        return "ok"

    with (
        patch("time.sleep") as sleep,
        patch("random.uniform", side_effect=lambda low, high: low),
    ):
        assert request() == "ok"
    assert sleep.call_args.args[0] >= 60


def test_retry_after_supports_http_date() -> None:
    now = datetime(2026, 9, 14, tzinfo=UTC).timestamp()
    with patch("time.time", return_value=now):
        assert (
            _get_retry_after(response_error(429, "Mon, 14 Sep 2026 00:01:00 GMT")) == 60
        )


@pytest.mark.parametrize("header", ["-1", "nan", "inf"])
def test_retry_after_rejects_invalid_delays(header: str) -> None:
    assert _get_retry_after(response_error(429, header)) is None


def test_bad_request_is_not_retried() -> None:
    request = MagicMock(side_effect=response_error(400))
    with patch("time.sleep") as sleep, pytest.raises(requests.HTTPError):
        retry(max_attempts=3)(request)()
    assert request.call_count == 1
    sleep.assert_not_called()


def test_arxiv_terminal_rate_limit_does_not_sleep() -> None:
    session = MagicMock()
    session.get.return_value = response_error(429).response
    client = ArxivClient(session=session, rate_limiter=MagicMock(), max_retries=0)
    with patch("time.sleep") as sleep, pytest.raises(requests.HTTPError):
        client._fetch_page("https://export.arxiv.org/api/query", "fixture")
    assert session.get.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (
            "198704120000",
            "202609140000",
            "from_publication_date:1987-04-12,to_publication_date:2026-09-14",
        ),
        (
            "2024-02-29",
            "2026-09-14",
            "from_publication_date:2024-02-29,to_publication_date:2026-09-14",
        ),
        (
            "2024",
            "2026",
            "from_publication_date:2024-01-01,to_publication_date:2026-12-31",
        ),
    ],
)
def test_openalex_dates_are_provider_valid(start: str, end: str, expected: str) -> None:
    source = OpenAlexSource()
    with patch.object(source, "_api_get", return_value={"results": []}) as get:
        source.search("email", ["email"], [], date_from=start, date_to=end)
    assert get.call_args.args[1]["filter"] == expected


def test_source_defaults_are_conservative() -> None:
    config = PipelineConfig()
    assert config.arxiv.min_interval_seconds == 30
    for name in (
        "scholar",
        "serpapi",
        "semantic_scholar",
        "openalex",
        "dblp",
        "huggingface",
    ):
        assert getattr(config.sources, f"{name}_min_interval") == 30
