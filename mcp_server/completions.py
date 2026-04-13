"""MCP completion handlers for argument auto-complete.

Provides completions for resource template URIs and prompt arguments,
enabling clients to offer auto-complete for run IDs, paper IDs,
backend names, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.types import (
    Completion,
    CompletionArgument,
    PromptReference,
    ResourceTemplateReference,
)

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIRS = ["./runs", "./workspace"]


def _list_run_ids(prefix: str = "") -> list[str]:
    """List available run IDs matching an optional prefix."""
    run_ids: list[str] = []
    for base in DEFAULT_RUNS_DIRS:
        runs_dir = Path(base).resolve()
        if not runs_dir.is_dir():
            continue
        for entry in sorted(runs_dir.iterdir()):
            if (
                entry.is_dir()
                and not entry.name.startswith(".")
                and entry.name.startswith(prefix)
            ):
                run_ids.append(entry.name)
    return run_ids


def _list_paper_ids(run_id: str, prefix: str = "") -> list[str]:
    """List paper IDs within a run matching an optional prefix."""
    paper_ids: list[str] = []
    for base in DEFAULT_RUNS_DIRS:
        pdf_dir = Path(base).resolve() / run_id / "download" / "pdf"
        if not pdf_dir.is_dir():
            continue
        for pdf_file in sorted(pdf_dir.glob("*.pdf")):
            paper_id = pdf_file.stem
            if paper_id.startswith(prefix):
                paper_ids.append(paper_id)
    return paper_ids


def _list_backends() -> list[str]:
    """List available converter backend names."""
    try:
        from research_pipeline.conversion.registry import (
            _ensure_builtins_registered,
            list_backends,
        )

        _ensure_builtins_registered()
        return list_backends()
    except Exception:
        return ["pymupdf4llm", "docling", "marker"]


DIRECTION_VALUES = ["citations", "references", "both"]
SOURCE_VALUES = ["arxiv", "scholar", "semantic_scholar", "openalex", "dblp", "all"]


async def handle_completion(
    ref: PromptReference | ResourceTemplateReference,
    argument: CompletionArgument,
    context: dict | None = None,
) -> Completion | None:
    """Route completion requests to appropriate handlers."""
    partial = argument.value or ""
    name = argument.name

    if name == "run_id":
        values = _list_run_ids(partial)
        return Completion(values=values[:100], hasMore=len(values) > 100)

    if name == "paper_id":
        # Need run_id from context to scope paper lookup
        run_id = ""
        if context:
            run_id = context.get("run_id", "")
        if isinstance(ref, ResourceTemplateReference):
            # Try to extract run_id from the URI
            uri = str(ref.uri)
            parts = uri.split("/")
            for i, part in enumerate(parts):
                if part == "" and i > 0:
                    run_id = parts[i - 1]
                    break
        if run_id:
            values = _list_paper_ids(run_id, partial)
            return Completion(values=values[:100], hasMore=len(values) > 100)
        return Completion(values=[])

    if name == "backend":
        backends = _list_backends()
        values = [b for b in backends if b.startswith(partial)]
        return Completion(values=values)

    if name == "direction":
        values = [d for d in DIRECTION_VALUES if d.startswith(partial)]
        return Completion(values=values)

    if name == "source":
        values = [s for s in SOURCE_VALUES if s.startswith(partial)]
        return Completion(values=values)

    if name == "topic":
        # Suggest recent topics from existing runs
        topics: list[str] = []
        import json

        for base in DEFAULT_RUNS_DIRS:
            runs_dir = Path(base).resolve()
            if not runs_dir.is_dir():
                continue
            for entry in sorted(runs_dir.iterdir(), reverse=True):
                if not entry.is_dir():
                    continue
                plan_path = entry / "plan" / "query_plan.json"
                if plan_path.exists():
                    try:
                        data = json.loads(plan_path.read_text())
                        topic = data.get("topic", "")
                        if topic and topic.startswith(partial) and topic not in topics:
                            topics.append(topic)
                    except (json.JSONDecodeError, OSError):
                        continue
                if len(topics) >= 20:
                    break
        return Completion(values=topics)

    return None
