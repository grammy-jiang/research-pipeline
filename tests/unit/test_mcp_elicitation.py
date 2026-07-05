"""Tests for MCP elicitation governance gates (#37)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from research_pipeline.mcp_server.workflow.research import (
    _ApprovalDecision,
    _elicit_approval,
)


def _telemetry() -> MagicMock:
    t = MagicMock()
    t.log_user_decision = MagicMock()
    return t


def _ctx(action: str, approved: bool | None = None) -> MagicMock:
    data = _ApprovalDecision(approved=approved) if approved is not None else None
    ctx = MagicMock()
    ctx.elicit = AsyncMock(return_value=SimpleNamespace(action=action, data=data))
    return ctx


def test_schema_is_basemodel_not_dict() -> None:
    """The SDK requires a BaseModel subclass; a dict raised AttributeError."""
    ctx = _ctx("accept", approved=True)
    asyncio.run(_elicit_approval(ctx, "msg", _telemetry(), "plan"))
    _, kwargs = ctx.elicit.call_args
    assert kwargs["schema"] is _ApprovalDecision


def test_accept_approved_true_proceeds() -> None:
    ctx = _ctx("accept", approved=True)
    decision = asyncio.run(_elicit_approval(ctx, "msg", _telemetry(), "plan"))
    assert decision == {"approved": True}


def test_accept_approved_false_stops() -> None:
    ctx = _ctx("accept", approved=False)
    decision = asyncio.run(_elicit_approval(ctx, "msg", _telemetry(), "plan"))
    assert decision["approved"] is False


def test_cancel_fails_closed() -> None:
    """Explicit cancel must not silently proceed (fail-closed)."""
    ctx = _ctx("cancel")
    decision = asyncio.run(
        _elicit_approval(ctx, "msg", _telemetry(), "iterate", iteration=1)
    )
    assert decision["approved"] is False
    assert decision["cancelled"] is True


def test_decline_stops() -> None:
    ctx = _ctx("decline")
    decision = asyncio.run(_elicit_approval(ctx, "msg", _telemetry(), "screen"))
    assert decision["approved"] is False


def test_elicitation_unavailable_proceeds() -> None:
    """A client without elicitation capability cannot gate; proceed + log."""
    ctx = MagicMock()
    ctx.elicit = AsyncMock(side_effect=RuntimeError("no elicitation capability"))
    decision = asyncio.run(_elicit_approval(ctx, "msg", _telemetry(), "plan"))
    assert decision == {"approved": True}
