"""Shared MCP logging-level state for the packaged server.

The server emits ``notifications/message`` via ``ctx.info/.warning/.error``.
Per the MCP specification a server that emits log notifications MUST declare
the ``logging`` capability and honour the client-set minimum level
(``logging/setLevel``). This module holds the client-configured minimum level
and the severity ordering used to gate emissions. See issue #41.
"""

from __future__ import annotations

from mcp.types import LoggingLevel

# Syslog severity ordering per the MCP spec (RFC 5424), least → most severe.
_SEVERITY: dict[str, int] = {
    "debug": 0,
    "info": 1,
    "notice": 2,
    "warning": 3,
    "error": 4,
    "critical": 5,
    "alert": 6,
    "emergency": 7,
}

# Default: emit everything until a client narrows the level via setLevel.
_min_level: LoggingLevel = "debug"


def set_min_level(level: LoggingLevel) -> None:
    """Record the client-requested minimum log level (``logging/setLevel``)."""
    global _min_level
    _min_level = level


def get_min_level() -> LoggingLevel:
    """Return the current minimum log level."""
    return _min_level


def should_emit(level: str) -> bool:
    """Return True if a message at *level* meets the client-set minimum."""
    return _SEVERITY.get(level, _SEVERITY["info"]) >= _SEVERITY.get(
        _min_level, _SEVERITY["debug"]
    )
