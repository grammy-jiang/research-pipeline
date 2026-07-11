"""Tests for research_pipeline.infra.logging."""

from __future__ import annotations

import json
import logging

from research_pipeline.infra.logging import JSONLFormatter


def _record(msg: str, args: object = None) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


# Obviously-fake, low-entropy tokens that still match the redactor's shapes.
_FAKE_OPENAI = "sk-" + "0" * 24
_FAKE_GITHUB = "ghp_" + "0" * 36


def test_formatter_redacts_secret_in_message() -> None:
    """A credential surfaced in a log message must not be persisted (#125)."""
    out = JSONLFormatter().format(_record(f"calling API with {_FAKE_OPENAI}"))
    entry = json.loads(out)
    assert _FAKE_OPENAI not in entry["message"]
    assert "[REDACTED]" in entry["message"]


def test_formatter_preserves_ordinary_message() -> None:
    out = JSONLFormatter().format(_record("screened 42 candidates"))
    assert json.loads(out)["message"] == "screened 42 candidates"


def test_formatter_redacts_exception_text() -> None:
    try:
        raise RuntimeError(f"bad token {_FAKE_GITHUB}")
    except RuntimeError:
        import sys

        rec = logging.LogRecord(
            name="t",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=None,
            exc_info=sys.exc_info(),
        )
    entry = json.loads(JSONLFormatter().format(rec))
    assert _FAKE_GITHUB not in entry["exception"]
