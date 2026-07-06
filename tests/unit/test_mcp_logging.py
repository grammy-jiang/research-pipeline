"""Tests for MCP logging capability + level gating (#41)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from mcp.server.lowlevel.server import NotificationOptions
from mcp.types import SetLevelRequest

from research_pipeline.mcp_server import logging_state
from research_pipeline.mcp_server.server import mcp


@pytest.fixture(autouse=True)
def _restore_min_level() -> Iterator[None]:
    """Restore the module-global min level after each test."""
    saved = logging_state.get_min_level()
    try:
        yield
    finally:
        logging_state.set_min_level(saved)


class TestLoggingCapability:
    def test_setlevel_handler_registered(self) -> None:
        """Registering the handler is what advertises the capability."""
        assert SetLevelRequest in mcp._mcp_server.request_handlers

    def test_logging_capability_declared(self) -> None:
        """initialize must advertise `logging` since the server emits logs."""
        caps = mcp._mcp_server.get_capabilities(NotificationOptions(), {})
        assert caps.logging is not None


class TestLevelGating:
    def test_default_emits_everything(self) -> None:
        assert logging_state.get_min_level() == "debug"
        for level in ("debug", "info", "warning", "error", "emergency"):
            assert logging_state.should_emit(level) is True

    def test_setlevel_narrows_emissions(self) -> None:
        logging_state.set_min_level("warning")
        assert logging_state.should_emit("debug") is False
        assert logging_state.should_emit("info") is False
        assert logging_state.should_emit("warning") is True
        assert logging_state.should_emit("error") is True

    def test_unknown_level_treated_as_info(self) -> None:
        logging_state.set_min_level("error")
        # An unrecognised level maps to "info" severity, below "error".
        assert logging_state.should_emit("bogus") is False
