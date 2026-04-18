"""CLI handler for the 'cite-context' command.

Extracts citation contexts from converted Markdown papers.
"""

import json
import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.extraction.citation_context import (
    contexts_to_dicts,
    extract_citation_contexts,
    group_by_marker,
)
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def cite_context_command(
    run_id: str = typer.Option(..., help="Run ID to extract contexts from"),
    context_window: int = typer.Option(
        1,
        "--window",
        help="Extra sentences before/after citation (0=citing sentence only)",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output JSON file path (default: <convert_dir>/citation_contexts.json)",
    ),
    config_path: Path = typer.Option(
        Path("config.toml"),
        "--config",
        help="Path to config file",
    ),
) -> None:
    """Extract citation contexts from converted Markdown papers."""
    config = load_config(config_path)
    ws = Path(config.workspace)
    _run_id, run_dir = init_run(ws, run_id)

    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        raise typer.Exit(code=1)

    convert_dir = get_stage_dir(run_dir, "convert")
    md_files = sorted(convert_dir.glob("**/*.md"))

    if not md_files:
        logger.warning("No Markdown files found in %s", convert_dir)
        raise typer.Exit(code=1)

    all_contexts: dict[str, list[dict[str, object]]] = {}
    total_count = 0

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        contexts = extract_citation_contexts(text, context_window=context_window)

        if contexts:
            paper_key = md_file.stem
            all_contexts[paper_key] = contexts_to_dicts(contexts)
            total_count += len(contexts)

            groups = group_by_marker(contexts)
            logger.info(
                "%s: %d contexts from %d unique markers",
                md_file.name,
                len(contexts),
                len(groups),
            )

    output_path = output or (convert_dir / "citation_contexts.json")
    output_path.write_text(json.dumps(all_contexts, indent=2, ensure_ascii=False))

    logger.info(
        "Extracted %d citation contexts from %d/%d papers → %s",
        total_count,
        len(all_contexts),
        len(md_files),
        output_path,
    )
