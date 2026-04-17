"""Tests for cli/cmd_watch.py — watch mode for topics."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research_pipeline.cli.cmd_watch import (
    _load_queries,
    _load_watch_state,
    _save_watch_state,
    watch_command,
)


class TestLoadWatchState:
    """Tests for _load_watch_state."""

    def test_returns_empty_dict_when_no_file(self, tmp_path: Path) -> None:
        """Returns empty dict when state file doesn't exist."""
        state = _load_watch_state(tmp_path / "nonexistent.json")
        assert state == {}

    def test_loads_existing_state(self, tmp_path: Path) -> None:
        """Loads state from existing JSON file."""
        state_path = tmp_path / "state.json"
        state_path.write_text('{"topic1": "2024-01-15T00:00:00+00:00"}')
        state = _load_watch_state(state_path)
        assert "topic1" in state
        assert "2024-01-15" in state["topic1"]


class TestSaveWatchState:
    """Tests for _save_watch_state."""

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Creates parent directories if needed."""
        state_path = tmp_path / "sub" / "dir" / "state.json"
        _save_watch_state(state_path, {"test": "value"})
        assert state_path.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Save then load produces same state."""
        state_path = tmp_path / "state.json"
        original = {
            "q1": "2024-06-01T00:00:00+00:00",
            "q2": "2024-06-15T12:00:00+00:00",
        }
        _save_watch_state(state_path, original)
        loaded = _load_watch_state(state_path)
        assert loaded == original


class TestLoadQueries:
    """Tests for _load_queries."""

    def test_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        """Returns empty list when file doesn't exist."""
        queries = _load_queries(tmp_path / "nonexistent.json")
        assert queries == []

    def test_loads_query_list(self, tmp_path: Path) -> None:
        """Loads list of query dicts."""
        qf = tmp_path / "queries.json"
        data = [
            {"name": "transformers", "query": "transformer architecture"},
            {"name": "rag", "query": "retrieval augmented generation"},
        ]
        qf.write_text(json.dumps(data))
        queries = _load_queries(qf)
        assert len(queries) == 2
        assert queries[0]["name"] == "transformers"

    def test_non_list_returns_empty(self, tmp_path: Path) -> None:
        """Non-list JSON returns empty list."""
        qf = tmp_path / "queries.json"
        qf.write_text('{"not": "a list"}')
        queries = _load_queries(qf)
        assert queries == []


class TestWatchCommand:
    """Tests for watch_command."""

    def test_exits_when_no_queries(self, tmp_path: Path) -> None:
        """Exits with error when no queries file found."""
        from click.exceptions import Exit

        with pytest.raises(Exit):
            watch_command(
                queries_file=tmp_path / "missing.json",
                config_path=tmp_path / "config.toml",
            )

    def test_exits_when_empty_queries(self, tmp_path: Path) -> None:
        """Exits with error when queries file is empty list."""
        from click.exceptions import Exit

        qf = tmp_path / "queries.json"
        qf.write_text("[]")
        with pytest.raises(Exit):
            watch_command(
                queries_file=qf,
                config_path=tmp_path / "config.toml",
            )

    @patch("research_pipeline.cli.cmd_watch.ArxivClient")
    @patch("research_pipeline.cli.cmd_watch.create_session")
    @patch("research_pipeline.cli.cmd_watch.ArxivRateLimiter")
    def test_searches_each_query(
        self,
        mock_limiter_cls: MagicMock,
        mock_session_fn: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Runs a search for each configured query."""
        qf = tmp_path / "queries.json"
        qf.write_text(
            json.dumps([
                {"name": "q1", "query": "machine learning"},
                {"name": "q2", "query": "deep learning"},
            ])
        )

        mock_client = MagicMock()
        mock_client.search.return_value = []
        mock_client_cls.return_value = mock_client

        watch_command(
            queries_file=qf,
            config_path=tmp_path / "config.toml",
        )

        assert mock_client.search.call_count == 2

    @patch("research_pipeline.cli.cmd_watch.ArxivClient")
    @patch("research_pipeline.cli.cmd_watch.create_session")
    @patch("research_pipeline.cli.cmd_watch.ArxivRateLimiter")
    def test_filters_by_publish_date(
        self,
        mock_limiter_cls: MagicMock,
        mock_session_fn: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Only papers published after last check are reported."""
        qf = tmp_path / "queries.json"
        qf.write_text(json.dumps([{"name": "q1", "query": "test"}]))

        # Set last check to 3 days ago
        state_path = tmp_path / "watch_state.json"
        three_days_ago = datetime.now(tz=UTC) - timedelta(days=3)
        state_path.write_text(
            json.dumps({"q1": three_days_ago.isoformat()})
        )

        # Create mock papers: one old, one new
        old_paper = MagicMock()
        old_paper.arxiv_id = "2401.00001"
        old_paper.title = "Old Paper"
        old_paper.published = three_days_ago - timedelta(days=5)
        old_paper.authors = ["Author"]

        new_paper = MagicMock()
        new_paper.arxiv_id = "2401.00002"
        new_paper.title = "New Paper"
        new_paper.published = datetime.now(tz=UTC) - timedelta(hours=1)
        new_paper.authors = ["Author"]

        mock_client = MagicMock()
        mock_client.search.return_value = [old_paper, new_paper]
        mock_client_cls.return_value = mock_client

        output_file = tmp_path / "output.json"
        watch_command(
            queries_file=qf,
            output=output_file,
            config_path=tmp_path / "config.toml",
        )

        # Output should only contain the new paper
        result = json.loads(output_file.read_text())
        assert "q1" in result
        assert len(result["q1"]) == 1
        assert result["q1"][0]["arxiv_id"] == "2401.00002"

    @patch("research_pipeline.cli.cmd_watch.ArxivClient")
    @patch("research_pipeline.cli.cmd_watch.create_session")
    @patch("research_pipeline.cli.cmd_watch.ArxivRateLimiter")
    def test_updates_state_after_check(
        self,
        mock_limiter_cls: MagicMock,
        mock_session_fn: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Watch state is updated after each check."""
        qf = tmp_path / "queries.json"
        qf.write_text(json.dumps([{"name": "q1", "query": "test"}]))

        mock_client = MagicMock()
        mock_client.search.return_value = []
        mock_client_cls.return_value = mock_client

        watch_command(
            queries_file=qf,
            config_path=tmp_path / "config.toml",
        )

        state_path = tmp_path / "watch_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert "q1" in state

    @patch("research_pipeline.cli.cmd_watch.ArxivClient")
    @patch("research_pipeline.cli.cmd_watch.create_session")
    @patch("research_pipeline.cli.cmd_watch.ArxivRateLimiter")
    def test_handles_search_failure(
        self,
        mock_limiter_cls: MagicMock,
        mock_session_fn: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Search failure for one query doesn't stop others."""
        qf = tmp_path / "queries.json"
        qf.write_text(
            json.dumps([
                {"name": "q1", "query": "fails"},
                {"name": "q2", "query": "works"},
            ])
        )

        mock_client = MagicMock()
        mock_client.search.side_effect = [
            Exception("network error"),
            [],
        ]
        mock_client_cls.return_value = mock_client

        # Should not raise — graceful degradation
        watch_command(
            queries_file=qf,
            config_path=tmp_path / "config.toml",
        )

    @patch("research_pipeline.cli.cmd_watch.ArxivClient")
    @patch("research_pipeline.cli.cmd_watch.create_session")
    @patch("research_pipeline.cli.cmd_watch.ArxivRateLimiter")
    def test_lookback_days_default(
        self,
        mock_limiter_cls: MagicMock,
        mock_session_fn: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """First run uses lookback_days to set initial window."""
        qf = tmp_path / "queries.json"
        qf.write_text(json.dumps([{"name": "q1", "query": "test"}]))

        # Paper from 3 days ago (within 7-day lookback)
        recent_paper = MagicMock()
        recent_paper.arxiv_id = "2401.00001"
        recent_paper.title = "Recent Paper"
        recent_paper.published = datetime.now(tz=UTC) - timedelta(days=3)
        recent_paper.authors = ["Author"]

        mock_client = MagicMock()
        mock_client.search.return_value = [recent_paper]
        mock_client_cls.return_value = mock_client

        output_file = tmp_path / "output.json"
        watch_command(
            queries_file=qf,
            lookback_days=7,
            output=output_file,
            config_path=tmp_path / "config.toml",
        )

        result = json.loads(output_file.read_text())
        assert len(result.get("q1", [])) == 1

    @patch("research_pipeline.cli.cmd_watch.ArxivClient")
    @patch("research_pipeline.cli.cmd_watch.create_session")
    @patch("research_pipeline.cli.cmd_watch.ArxivRateLimiter")
    def test_skips_empty_query(
        self,
        mock_limiter_cls: MagicMock,
        mock_session_fn: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Queries with empty query text are skipped."""
        qf = tmp_path / "queries.json"
        qf.write_text(
            json.dumps([
                {"name": "empty", "query": ""},
                {"name": "valid", "query": "test"},
            ])
        )

        mock_client = MagicMock()
        mock_client.search.return_value = []
        mock_client_cls.return_value = mock_client

        watch_command(
            queries_file=qf,
            config_path=tmp_path / "config.toml",
        )

        # Only called once (the valid query)
        assert mock_client.search.call_count == 1

    @patch("research_pipeline.cli.cmd_watch.ArxivClient")
    @patch("research_pipeline.cli.cmd_watch.create_session")
    @patch("research_pipeline.cli.cmd_watch.ArxivRateLimiter")
    def test_no_output_file_when_no_new_papers(
        self,
        mock_limiter_cls: MagicMock,
        mock_session_fn: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """No output file created when there are no new papers."""
        qf = tmp_path / "queries.json"
        qf.write_text(json.dumps([{"name": "q1", "query": "test"}]))

        mock_client = MagicMock()
        mock_client.search.return_value = []
        mock_client_cls.return_value = mock_client

        output_file = tmp_path / "output.json"
        watch_command(
            queries_file=qf,
            output=output_file,
            config_path=tmp_path / "config.toml",
        )

        assert not output_file.exists()
