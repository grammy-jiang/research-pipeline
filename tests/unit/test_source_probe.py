"""Offline public-source probe contracts; no live requests are permitted."""

import asyncio
import io
import json
import socket
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests
from typer.testing import CliRunner

from research_pipeline.cli.app import app
from research_pipeline.config.models import PipelineConfig
from research_pipeline.models.source_probe import ProbeReport, ProbeResult, ProbeSource
from research_pipeline.sources.source_probe import probe_sources

MODULE = "research_pipeline.sources.source_probe"
ATOM = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1234.5678</id></entry></feed>'


def response(
    status=200, body=b'{"results":[{"id":"W1"}]}', mime="application/json", headers=None
):
    value = requests.Response()
    value.status_code = status
    value.headers["Content-Type"] = mime
    value.headers.update(headers or {})
    value.raw = io.BytesIO(body)
    return value


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock():
    value = Clock()
    with patch(f"{MODULE}.time", value):
        yield value


@pytest.fixture
def transport(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_PIPELINE_REQUEST_STATE_DIR", str(tmp_path / "budgets"))
    with patch("requests.adapters.HTTPAdapter.send") as send:
        yield send


@pytest.mark.parametrize(
    ("source", "http", "body", "mime", "status"),
    [
        ("arxiv", 200, ATOM, "application/atom+xml", "ok"),
        (
            "semantic_scholar",
            200,
            b'{"data":[{"paperId":"p1"}]}',
            "application/json",
            "ok",
        ),
        ("openalex", 200, b'{"results":[]}', "application/json", "ok"),
        (
            "dblp",
            200,
            b'{"result":{"hits":{"@total":"1","hit":[{"info":{"title":"Paper"}}]}}}',
            "application/json",
            "ok",
        ),
        (
            "dblp",
            200,
            b"<html><title>Making sure you&#39;re not a bot!</title></html>",
            "text/html",
            "bot_challenge",
        ),
        (
            "dblp",
            403,
            b"<html>Verify you are human</html>",
            "text/html",
            "bot_challenge",
        ),
        (
            "openalex",
            401,
            b'{"error":"Unauthorized"}',
            "application/json",
            "auth_required",
        ),
        ("openalex", 403, b'{"error":"Forbidden"}', "application/json", "forbidden"),
        (
            "openalex",
            403,
            b'{"message":"API key required"}',
            "application/json",
            "auth_required",
        ),
        ("openalex", 503, b"unavailable", "text/plain", "http_error"),
        (
            "openalex",
            200,
            b'{"error":"not results"}',
            "application/json",
            "unexpected_response",
        ),
        (
            "arxiv",
            200,
            b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#incorrect_id_format_for</id></entry></feed>',
            "application/atom+xml",
            "unexpected_response",
        ),
    ],
)
def test_classifies_single_response(transport, clock, source, http, body, mime, status):
    transport.return_value = response(http, body, mime)
    report = probe_sources(source)
    item = report.results[0]
    assert item.status == status
    assert item.http_status == http
    assert item.received_http_response is True
    assert item.request_sent is True
    assert report.all_available is (status == "ok")
    assert transport.call_count == 1
    assert transport.call_args.kwargs["stream"] is True
    assert ProbeReport.model_validate_json(report.model_dump_json()) == report


def test_429_persists_cooldown_and_next_probe_skips_without_network(transport, clock):
    transport.return_value = response(
        429, b"rate limited", "text/plain", {"Retry-After": "120"}
    )
    first = probe_sources("semantic_scholar").results[0]
    second = probe_sources("semantic_scholar").results[0]
    assert first.status == "rate_limited"
    assert first.retry_after_seconds == 120
    assert first.cooldown_until > datetime.now(UTC)
    assert second.status == "cooldown"
    assert second.request_sent is False
    assert second.received_http_response is False
    assert second.cooldown_until == first.cooldown_until
    assert transport.call_count == 1
    assert clock.sleeps == []


def test_redirect_is_not_followed(transport, clock):
    transport.return_value = response(
        302, b"", "text/html", {"Location": "https://example.invalid/secret"}
    )
    item = probe_sources("openalex").results[0]
    assert item.status == "redirect"
    assert transport.call_count == 1
    assert "example.invalid" not in item.model_dump_json()


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (requests.Timeout("private details"), "timeout"),
        (requests.exceptions.SSLError("private details"), "tls_error"),
        (requests.ConnectionError(socket.gaierror(-2, "private details")), "dns_error"),
        (requests.ConnectionError("private details"), "connection_error"),
    ],
)
def test_transport_failures_are_distinct_and_do_not_leak_details(
    transport, clock, error, status
):
    transport.side_effect = error
    report = probe_sources("openalex")
    assert report.results[0].status == status
    assert not report.results[0].received_http_response
    assert "private details" not in report.model_dump_json()
    assert transport.call_count == 1


def test_all_sources_are_serial_and_anonymous_with_no_retry(transport, clock):
    transport.side_effect = [
        response(200, ATOM, "application/atom+xml"),
        response(200, b'{"data":[]}'),
        response(200, b'{"result":{"hits":{"@total":"0"}}}'),
        response(),
    ]
    with patch(
        "requests.sessions.get_netrc_auth",
        side_effect=AssertionError("netrc must not be used"),
    ):
        report = probe_sources()
    assert report.all_available
    assert len(report.results) == 4
    assert clock.sleeps == [30, 30, 30]
    for call in transport.call_args_list:
        assert "Authorization" not in call.args[0].headers
        assert "x-api-key" not in call.args[0].headers
        assert call.kwargs["timeout"] == (10, 15)


def test_configured_source_interval_is_not_shortened(transport, clock):
    from research_pipeline.infra.http import SourceSession

    transport.return_value = response()
    config = PipelineConfig()
    config.sources.openalex_min_interval = 75
    with patch(f"{MODULE}.SourceSession", wraps=SourceSession) as session:
        probe_sources("openalex", config=config)
    assert session.call_args.kwargs["min_interval"] == 75


def test_unknown_source_fails_before_network(transport, clock):
    with pytest.raises(ValueError):
        probe_sources("https://example.invalid")
    transport.assert_not_called()


def test_oversized_body_is_bounded_and_not_accepted(transport, clock):
    raw = io.BytesIO(b'{"results":[' + b" " * 100_000 + b"]}")
    value = response()
    value.raw = raw
    transport.return_value = value
    item = probe_sources("openalex").results[0]
    assert item.status == "unexpected_response"
    assert item.body_truncated
    assert value.raw.closed
    assert item.response_bytes <= 65536


def report_fixture(status="ok"):
    return ProbeReport(
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        results=[
            ProbeResult(
                source=ProbeSource.OPENALEX,
                endpoint="https://api.openalex.org/works",
                checked_at=datetime.now(UTC),
                status=status,
                http_status=200 if status == "ok" else 429,
                request_sent=True,
                received_http_response=True,
            )
        ],
    )


def test_cli_json_and_exit_status_match_probe_result(tmp_path):
    report = report_fixture("rate_limited")
    path = tmp_path / "probe.json"
    with patch(
        "research_pipeline.cli.cmd_probe_sources.probe_sources", return_value=report
    ):
        result = CliRunner().invoke(
            app,
            ["probe-sources", "--source", "openalex", "--json", "--output", str(path)],
        )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == json.loads(path.read_text())
    assert json.loads(result.stdout)["results"][0]["status"] == "rate_limited"


def test_mcp_adapter_returns_observations_and_awaits_progress():
    from research_pipeline.mcp_server.schemas import ProbeSourcesInput
    from research_pipeline.mcp_server.tools.source_probe import probe_sources_tool

    ctx = Mock()
    ctx.report_progress = AsyncMock()

    def fake_probe(*args, **kwargs):
        kwargs["progress"](0, 1, "Checking openalex")
        return report_fixture("rate_limited")

    with patch(
        "research_pipeline.mcp_server.tools.source_probe.probe_sources",
        side_effect=fake_probe,
    ):
        result = asyncio.run(
            probe_sources_tool(ProbeSourcesInput(source="openalex"), ctx)
        )
    assert result.success
    assert result.artifacts["probe"]["results"][0]["status"] == "rate_limited"
    ctx.report_progress.assert_awaited()


def test_mcp_probe_is_registered_as_networked_and_mutating():
    from research_pipeline.mcp_server.server import mcp

    tool = next(
        t for t in asyncio.run(mcp.list_tools()) if t.name == "tool_probe_sources"
    )
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.openWorldHint is True
    assert tool.annotations.idempotentHint is False
    assert set(tool.inputSchema["properties"]) == {"source", "config_path"}


def test_missing_retry_after_uses_existing_shared_fifteen_minute_cooldown(
    transport, clock
):
    value = response(429, b"rate limited", "text/plain")
    transport.return_value = value
    before = datetime.now(UTC)
    item = probe_sources("arxiv").results[0]
    assert item.retry_after_seconds is None
    assert 899 <= (item.cooldown_until - before).total_seconds() <= 905
    assert value.raw.closed


def test_all_cooling_sources_skip_without_requests_or_spacing_waits(transport, clock):
    from research_pipeline.infra.request_budget import SharedRequestBudget

    for name in ("arxiv", "semantic_scholar", "dblp", "openalex"):
        budget = SharedRequestBudget(name, 30)
        with budget.slot():
            budget.defer(120)
    report = probe_sources()
    assert {item.status for item in report.results} == {"cooldown"}
    assert not report.all_available
    transport.assert_not_called()
    assert clock.sleeps == []


def test_timeout_while_reading_retains_http_observation_and_closes_body(
    transport, clock
):
    from urllib3.exceptions import ReadTimeoutError

    value = response()
    value.iter_content = Mock(
        side_effect=requests.ConnectionError(
            ReadTimeoutError(None, None, "private details")
        )
    )
    transport.return_value = value
    item = probe_sources("openalex").results[0]
    assert item.status == "timeout"
    assert item.http_status == 200
    assert item.received_http_response
    assert value.raw.closed
    assert "private details" not in item.model_dump_json()


def test_arxiv_external_entity_is_not_resolved(transport, clock, tmp_path):
    private = tmp_path / "private.txt"
    private.write_text("PRIVATE_XML_SENTINEL")
    body = (
        '<!DOCTYPE feed [<!ENTITY external SYSTEM "' + private.as_uri() + '">]>'
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>&external;</id></entry></feed>'
    ).encode()
    transport.return_value = response(200, body, "application/atom+xml")
    report = probe_sources("arxiv")
    assert report.results[0].status == "unexpected_response"
    assert "PRIVATE_XML_SENTINEL" not in report.model_dump_json()


def test_cli_rejects_arbitrary_url_without_network(transport):
    result = CliRunner().invoke(
        app, ["probe-sources", "--source", "https://example.invalid"]
    )
    assert result.exit_code == 2
    transport.assert_not_called()


def test_mcp_adapter_failure_is_sanitized():
    from research_pipeline.mcp_server.schemas import ProbeSourcesInput
    from research_pipeline.mcp_server.tools.source_probe import probe_sources_tool

    with patch(
        "research_pipeline.mcp_server.tools.source_probe.probe_sources",
        side_effect=ValueError("private details"),
    ):
        result = asyncio.run(probe_sources_tool(ProbeSourcesInput(source="openalex")))
    assert not result.success
    assert "private details" not in result.model_dump_json()


def test_progress_notification_failure_does_not_discard_probe_observations():
    from research_pipeline.mcp_server.schemas import ProbeSourcesInput
    from research_pipeline.mcp_server.tools.source_probe import probe_sources_tool

    ctx = Mock()
    ctx.report_progress = AsyncMock(side_effect=RuntimeError("private details"))

    def fake_probe(*args, **kwargs):
        kwargs["progress"](0, 1, "Checking openalex")
        return report_fixture()

    with patch(
        "research_pipeline.mcp_server.tools.source_probe.probe_sources",
        side_effect=fake_probe,
    ):
        result = asyncio.run(
            probe_sources_tool(ProbeSourcesInput(source="openalex"), ctx)
        )
    assert result.success
    assert result.artifacts["probe"]["all_available"]
    assert "private details" not in result.model_dump_json()
