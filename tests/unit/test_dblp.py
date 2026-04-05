"""Unit tests for sources.dblp_source."""

from unittest.mock import MagicMock, patch

import requests

from research_pipeline.sources.dblp_source import DBLPSource


def _make_dblp_hit(
    key: str = "conf/nips/Author24",
    title: str = "Test Paper",
    year: str = "2024",
    doi: str = "10.1234/test",
) -> dict:  # type: ignore[type-arg]
    """Build a minimal DBLP hit dict for testing."""
    return {
        "info": {
            "key": key,
            "title": title,
            "authors": {
                "author": [
                    {"text": "Author One"},
                    {"text": "Author Two"},
                ]
            },
            "year": year,
            "doi": doi,
            "venue": "NeurIPS",
            "url": f"https://dblp.org/rec/{key}",
            "ee": f"https://doi.org/{doi}",
        }
    }


class TestDBLPSource:
    """Tests for DBLPSource."""

    def test_name(self) -> None:
        source = DBLPSource()
        assert source.name == "dblp"

    def test_parse_hit_full(self) -> None:
        source = DBLPSource()
        info = _make_dblp_hit()["info"]
        candidate = source._parse_hit(info)

        assert candidate.title == "Test Paper"
        assert candidate.source == "dblp"
        assert candidate.doi == "10.1234/test"
        assert candidate.venue == "NeurIPS"
        assert candidate.year == 2024
        assert candidate.abstract == ""  # DBLP has no abstracts
        assert len(candidate.authors) == 2

    def test_parse_hit_trailing_period(self) -> None:
        source = DBLPSource()
        info = {"title": "Test Paper.", "authors": {"author": []}}
        candidate = source._parse_hit(info)
        assert candidate.title == "Test Paper"

    def test_parse_hit_arxiv_url(self) -> None:
        source = DBLPSource()
        info = _make_dblp_hit()["info"]
        info["ee"] = "https://arxiv.org/abs/2401.12345"
        candidate = source._parse_hit(info)
        assert candidate.arxiv_id == "2401.12345"

    def test_parse_hit_no_arxiv(self) -> None:
        source = DBLPSource()
        info = _make_dblp_hit()["info"]
        candidate = source._parse_hit(info)
        assert candidate.arxiv_id.startswith("dblp-")

    def test_parse_hit_single_author_dict(self) -> None:
        source = DBLPSource()
        info = _make_dblp_hit()["info"]
        info["authors"]["author"] = {"text": "Solo Author"}
        candidate = source._parse_hit(info)
        assert candidate.authors == ["Solo Author"]

    @patch.object(DBLPSource, "_api_get")
    def test_search_returns_candidates(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "result": {
                "hits": {
                    "hit": [
                        _make_dblp_hit("k1", "Paper One"),
                        _make_dblp_hit("k2", "Paper Two"),
                    ]
                }
            }
        }
        source = DBLPSource()
        results = source.search(
            topic="transformers",
            must_terms=["attention"],
            nice_terms=[],
            max_results=10,
        )

        assert len(results) == 2

    @patch.object(DBLPSource, "_api_get")
    def test_search_empty(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"result": {"hits": {"hit": []}}}
        source = DBLPSource()
        results = source.search(topic="test", must_terms=[], nice_terms=[])
        assert results == []

    @patch.object(DBLPSource, "_api_get")
    def test_search_handles_api_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.RequestException("fail")
        source = DBLPSource()
        results = source.search(topic="test", must_terms=["x"], nice_terms=[])
        assert results == []
