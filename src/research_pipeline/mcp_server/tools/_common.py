"""MCP tool implementations that wrap core pipeline services.

Each function accepts a typed input schema and returns a ToolResult.
These are pure adapter functions — no business logic.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


logger = logging.getLogger(__name__)


def _report_progress(
    ctx: Context | None, progress: float, total: float, message: str = ""
) -> None:
    """Safely report progress via MCP context if available."""
    if ctx is not None:
        with contextlib.suppress(Exception):
            ctx.report_progress(progress, total, message)


def _log_info(ctx: Context | None, message: str) -> None:
    """Log via MCP context if available, always log via Python logger."""
    logger.info(message)
    if ctx is not None:
        from research_pipeline.mcp_server.logging_state import should_emit

        if should_emit("info"):
            with contextlib.suppress(Exception):
                ctx.info(message)


def _scrub_exc(exc: object) -> str:
    """Redact absolute filesystem paths from an exception message.

    Tool errors are returned to the client verbatim; raw exception text can
    leak internal directory structure (and the OS username via the home dir),
    and — since backends are built with live API keys — a credential embedded in
    an exception (#125, HC6). Replace the home/working-directory prefixes with an
    ellipsis, then redact credential-shaped substrings.
    """
    from research_pipeline.infra.sanitize import redact_secrets

    text = str(exc)
    for base in (str(Path.home()), str(Path.cwd())):
        if base and base != "/":
            text = text.replace(base, "…")
    return redact_secrets(text)


class McpToolError(RuntimeError):
    """An MCP tool wrapper caught an unexpected error (#108).

    Subclasses :class:`RuntimeError` so existing handlers keep working, but is a
    distinct type; the triggering exception is preserved as ``__cause__`` so
    callers and logs can still discriminate the underlying failure mode.
    """


def _raise_tool_error(label: str, exc: Exception) -> NoReturn:
    """Single MCP-tool error boundary (#108): log, path/secret-scrub, re-raise.

    This log + redaction + re-raise contract was copy-pasted at ~63 tool call
    sites; centralising it means a change to the contract touches one place, and
    every boundary now raises the same :class:`McpToolError` type.

    Args:
        label: Human phrase for the failing operation (used in the log line).
        exc: The caught exception (preserved as ``__cause__``).
    """
    logger.error("%s failed: %s", label, exc)
    raise McpToolError(_scrub_exc(exc)) from exc


def _backend_kwargs(
    backend_name: str,
    config: object,
) -> dict[str, object]:
    """Build constructor kwargs for a converter backend from config."""
    if backend_name == "docling":
        return {"timeout_seconds": config.conversion.timeout_seconds}  # type: ignore[union-attr]
    if backend_name == "marker":
        mc = config.conversion.marker  # type: ignore[union-attr]
        kwargs: dict[str, object] = {"force_ocr": mc.force_ocr}
        if mc.use_llm:
            kwargs["use_llm"] = True
            if mc.llm_service:
                kwargs["llm_service"] = mc.llm_service
            if mc.llm_api_key:
                kwargs["llm_api_key"] = mc.llm_api_key
        return kwargs
    if backend_name == "mathpix":
        mp = config.conversion.mathpix  # type: ignore[union-attr]
        return {"app_id": mp.app_id, "app_key": mp.app_key}
    if backend_name == "datalab":
        dl = config.conversion.datalab  # type: ignore[union-attr]
        return {"api_key": dl.api_key, "mode": dl.mode}
    if backend_name == "llamaparse":
        lp = config.conversion.llamaparse  # type: ignore[union-attr]
        return {"api_key": lp.api_key}
    if backend_name == "mistral_ocr":
        mo = config.conversion.mistral_ocr  # type: ignore[union-attr]
        return {"api_key": mo.api_key, "model": mo.model}
    if backend_name == "openai_vision":
        ov = config.conversion.openai_vision  # type: ignore[union-attr]
        return {"api_key": ov.api_key, "model": ov.model}
    return {}


def _resolve_workspace(workspace: str) -> Path:
    """Resolve a workspace path.

    An empty string falls back to the shared default so every tool resolves
    an omitted ``workspace=`` to the same location (#43).
    """
    return Path(workspace or "./workspace").expanduser().resolve()


def _resolve_run_id(run_id: str) -> str:
    """Generate a run ID if not provided."""
    if run_id:
        return run_id
    from research_pipeline.infra.clock import utc_now

    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def _get_run_root(ws: Path, rid: str) -> Path:
    """Compute the run root directory from workspace and run ID."""
    from research_pipeline.infra.paths import run_dir

    return run_dir(ws, rid)


def _resolve_latest_run_id(ws: Path, run_id: str) -> str:
    """Resolve an inspect/read tool's ``run_id``: explicit id, else latest run.

    Read tools document ``run_id=""`` as "latest". Unlike ``_resolve_run_id``
    (which auto-GENERATES a new id for write tools), this resolves the empty
    value to the most recent existing run — never joins ``""`` onto the
    workspace and silently reads the workspace root (#110). Returns ``""`` only
    when the workspace holds no run.
    """
    if run_id:
        return run_id
    from research_pipeline.infra.paths import latest_run_id

    return latest_run_id(ws)


def _sanitize_candidates(records: list) -> None:  # type: ignore[type-arg]
    """Sanitize untrusted scraped title/abstract in place at the stage boundary.

    ``docs/security-model.md`` §6.3 requires the content-sanitization gate at
    each stage boundary, but the MCP ``search`` / ``enrich`` path persisted
    scraped candidate fields raw — unlike the CLI orchestrator (issue #104).
    Neutralize prompt-injection patterns before the records are written and
    flow into downstream summarization / synthesis prompts.
    """
    from research_pipeline.infra.sanitize import sanitize_candidate_fields

    for r in records:
        r.title, r.abstract = sanitize_candidate_fields(r.title, r.abstract)


__all__ = [
    "McpToolError",
    "_backend_kwargs",
    "_get_run_root",
    "_log_info",
    "_raise_tool_error",
    "_report_progress",
    "_resolve_latest_run_id",
    "_resolve_run_id",
    "_resolve_workspace",
    "_sanitize_candidates",
    "_scrub_exc",
    "logger",
]
