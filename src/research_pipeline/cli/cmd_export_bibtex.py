"""CLI command: export-bibtex — export screened papers as a BibTeX file."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.models.screening import RelevanceDecision
from research_pipeline.storage.workspace import get_stage_dir, init_run
from research_pipeline.summarization.bibtex_export import (
    export_candidates_bibtex,
    load_candidates_from_jsonl,
)

logger = logging.getLogger(__name__)


def export_bibtex_cmd(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        help="Pipeline run ID.",
    ),
    stage: str = typer.Option(
        "screen",
        "--stage",
        help="Stage to export from: search, screen, or download.",
    ),
    output: str = typer.Option(
        "",
        "-o",
        "--output",
        help="Output .bib file path (default: auto in run dir).",
    ),
) -> None:
    """Export papers from a pipeline stage as a BibTeX file.

    Reads candidate JSONL files from the specified stage (search, screen,
    or download) and produces a .bib file suitable for LaTeX workflows.

    Example::

        research-pipeline export-bibtex --run-id <RUN_ID>
        research-pipeline export-bibtex --run-id <RUN_ID> --stage search
        research-pipeline export-bibtex --run-id <RUN_ID> -o refs.bib
    """
    cfg = load_config()
    ws = Path(cfg.workspace)
    _run_id, run_root = init_run(ws, run_id)
    stage_dir = get_stage_dir(run_root, stage)

    candidates = []
    shortlist_path = stage_dir / "shortlist.json"
    if stage == "screen" and shortlist_path.exists():
        raw = json.loads(shortlist_path.read_text(encoding="utf-8"))
        decisions = [RelevanceDecision.model_validate(item) for item in raw]
        candidates = [decision.paper for decision in decisions]
        logger.info("Loading candidates from %s", shortlist_path)
    else:
        # Find the candidates JSONL file
        jsonl_candidates = [
            f for f in stage_dir.glob("*.jsonl") if f.stem.startswith("candidates")
        ]
        if not jsonl_candidates:
            # Fallback: look for screened_candidates.jsonl or any .jsonl
            jsonl_candidates = list(stage_dir.glob("*.jsonl"))

        if not jsonl_candidates:
            logger.error(
                "No candidate JSONL files found in %s. Run the %s stage first.",
                stage_dir,
                stage,
            )
            raise typer.Exit(1)

        # Use the first (or most recent) JSONL file
        jsonl_path = sorted(jsonl_candidates)[-1]
        logger.info("Loading candidates from %s", jsonl_path)
        candidates = load_candidates_from_jsonl(jsonl_path)

    if not candidates:
        logger.warning("No candidates found in %s", stage_dir)
        raise typer.Exit(1)

    out_path = Path(output) if output else stage_dir / "references.bib"

    count = export_candidates_bibtex(candidates, out_path)
    logger.info("Exported %d BibTeX entries to %s", count, out_path)
