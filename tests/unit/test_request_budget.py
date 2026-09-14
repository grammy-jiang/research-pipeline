"""Shared source budgets must survive client/process restarts without traffic."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from research_pipeline.infra.http import SourceSession
from research_pipeline.infra.request_budget import SharedRequestBudget, SourceCooldown


def test_budget_serializes_clients_without_real_sleep(tmp_path: Path) -> None:
    clock = [1000.0]
    delays: list[float] = []

    def advance(seconds: float) -> None:
        delays.append(seconds)
        clock[0] += seconds

    first = SharedRequestBudget("arxiv", 30, tmp_path)
    second = SharedRequestBudget("arxiv", 30, tmp_path)
    with (
        patch("time.time", side_effect=lambda: clock[0]),
        patch("time.sleep", side_effect=advance),
    ):
        with first.slot():
            pass
        with second.slot():
            pass
    assert delays == [30.0]


def test_429_stops_and_persists_server_deadline(tmp_path: Path) -> None:
    response = requests.Response()
    response.status_code = 429
    response.url = "https://api.openalex.org/works"
    response.headers["Retry-After"] = "3600"
    with patch("requests.Session.request", return_value=response) as request:
        first = SourceSession("openalex", state_dir=tmp_path)
        with pytest.raises(SourceCooldown):
            first.get(response.url)
        second = SourceSession("openalex", state_dir=tmp_path)
        with pytest.raises(SourceCooldown):
            second.get(response.url)
    assert request.call_count == 1

    code = (
        "from pathlib import Path; "
        "from research_pipeline.infra.request_budget import SharedRequestBudget; "
        "budget=SharedRequestBudget('openalex',30,Path(__import__('sys').argv[1])); "
        "budget.check_available()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert "SourceCooldown" in result.stderr


def test_http_diagnostics_do_not_echo_credential_query(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    response = requests.Response()
    response.status_code = 400
    response.url = "https://api.openalex.org/works?api_key=private-fixture-value"
    response.headers["Content-Type"] = "text/html"
    caplog.set_level("INFO")
    with patch("requests.Session.request", return_value=response):
        session = SourceSession("openalex", state_dir=tmp_path)
        result = session.get(response.url)
        with pytest.raises(requests.HTTPError):
            result.raise_for_status()
    assert "private-fixture-value" not in caplog.text
    events = [
        json.loads(record.message.removeprefix("source_http "))
        for record in caplog.records
        if record.message.startswith("source_http ")
    ]
    assert [event["event"] for event in events] == ["dispatch", "response"]
    assert events[-1]["status"] == 400
    assert events[-1]["content_type"] == "text/html"
