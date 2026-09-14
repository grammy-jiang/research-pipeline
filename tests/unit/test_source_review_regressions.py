"""Regression cases found by independent offline source verification."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from research_pipeline.arxiv.client import ArxivClient
from research_pipeline.arxiv.parser import parse_atom_response
from research_pipeline.sources.scholar_source import SerpAPISource


def test_arxiv_rejects_successful_html_block_page():
    with pytest.raises(ValueError, match="Atom"):
        parse_atom_response("<html><body>temporarily blocked</body></html>")


def test_arxiv_keeps_prior_pages_on_failure(monkeypatch):
    client = ArxivClient(rate_limiter=MagicMock(), session=MagicMock())
    item = MagicMock()
    monkeypatch.setattr(
        "research_pipeline.arxiv.client.parse_atom_response", lambda _: [item] * 100
    )
    monkeypatch.setattr(
        "research_pipeline.arxiv.client.parse_total_results", lambda _: 101
    )
    client._fetch_page = MagicMock(side_effect=["feed", requests.HTTPError("fixture")])
    with pytest.raises(requests.HTTPError):
        client.search("all:email", max_results=101)
    assert client.partial_candidates == [item] * 100


def test_serp_error_is_visible_and_safe(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("RESEARCH_PIPELINE_REQUEST_STATE_DIR", str(tmp_path))
    search = MagicMock()
    search.return_value.get_dict.return_value = {"error": "nonsecret-fixture-marker"}
    monkeypatch.setitem(sys.modules, "serpapi", SimpleNamespace(GoogleSearch=search))
    source = SerpAPISource(api_key="fixture", min_interval=0)
    assert source.search("email", ["email"], []) == []
    assert isinstance(source.last_error, RuntimeError)
    assert "nonsecret-fixture-marker" not in caplog.text


def test_serp_queries_share_a_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_PIPELINE_REQUEST_STATE_DIR", str(tmp_path))
    now = [1000.0]
    sleeps = []
    monkeypatch.setattr(
        "research_pipeline.infra.request_budget.time.time", lambda: now[0]
    )

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr("research_pipeline.infra.request_budget.time.sleep", sleep)
    search = MagicMock()
    search.return_value.get_dict.return_value = {
        "organic_results": [
            {
                "title": "A paper",
                "snippet": "email",
                "link": "https://example.org/paper",
            }
        ]
    }
    monkeypatch.setitem(sys.modules, "serpapi", SimpleNamespace(GoogleSearch=search))
    for _ in range(2):
        result = SerpAPISource(api_key="fixture", min_interval=30).search(
            "email", [], []
        )
        assert result[0].source == "scholar"
    assert sleeps == [30.0]


def test_resume_does_not_query_or_replace_evidence(monkeypatch, tmp_path):
    from datetime import UTC, datetime

    from research_pipeline.cli.cmd_search import run_search
    from research_pipeline.config.models import PipelineConfig
    from research_pipeline.models.candidate import CandidateRecord

    run = tmp_path / "saved"
    search = run / "search"
    search.mkdir(parents=True)
    candidate = CandidateRecord(
        arxiv_id="fixture",
        version="v1",
        title="paper",
        authors=[],
        published=datetime.now(UTC),
        updated=datetime.now(UTC),
        categories=[],
        primary_category="",
        abstract="email",
        abs_url="https://example.org/paper",
        pdf_url="",
    )
    data = candidate.model_dump_json() + "\n"
    (search / "candidates.jsonl").write_text(data)
    cfg = PipelineConfig(workspace=str(tmp_path))
    monkeypatch.setattr("research_pipeline.cli.cmd_search.load_config", lambda _: cfg)
    call = MagicMock(side_effect=AssertionError("resume must not query"))
    monkeypatch.setattr("research_pipeline.cli.cmd_search.execute_search", call)
    run_search(resume=True, workspace=tmp_path, run_id="saved")
    call.assert_not_called()
    assert (search / "candidates.jsonl").read_text() == data
