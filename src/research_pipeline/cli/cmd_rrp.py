"""CLI command for the Recall / Reasoning / Presentation (RRP) diagnostic."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from research_pipeline.evaluation.recall_diagnostic import (
    compute_rrp_diagnostic,
)

logger = logging.getLogger(__name__)


def _load_shortlist_ids(shortlist_path: Path) -> list[str]:
    """Load paper IDs from a shortlist JSON (accepts list or {papers: [...]})."""
    data = json.loads(shortlist_path.read_text(encoding="utf-8"))
    entries: list[object]
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "papers" in data:
        entries = data["papers"]
    else:
        raise typer.BadParameter(
            "Shortlist must be a JSON list or {papers: [...]} object."
        )
    ids: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            ids.append(entry)
        elif isinstance(entry, dict):
            for key in ("paper_id", "arxiv_id", "id"):
                value = entry.get(key)
                if isinstance(value, str) and value:
                    ids.append(value)
                    break
    return ids


def rrp_cmd(
    report: Path = typer.Option(
        ..., "--report", "-r", help="Path to the synthesis report (md/txt)."
    ),
    shortlist: Path = typer.Option(
        ..., "--shortlist", "-s", help="Path to shortlist JSON with paper IDs."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON diagnostic to this path."
    ),
) -> None:
    """Compute Recall / Reasoning / Presentation diagnostic for a report.

    Operationalizes the DeepResearch Bench II finding (Theme 16): Information
    Recall is typically the bottleneck; Presentation is usually near-saturated.
    """
    if not report.exists():
        typer.echo(f"Report not found: {report}")
        raise typer.Exit(1)
    if not shortlist.exists():
        typer.echo(f"Shortlist not found: {shortlist}")
        raise typer.Exit(1)

    report_text = report.read_text(encoding="utf-8")
    ids = _load_shortlist_ids(shortlist)

    diagnostic = compute_rrp_diagnostic(report_text, ids)
    payload = diagnostic.to_dict()
    text = json.dumps(payload, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote RRP diagnostic to {output}")
    else:
        typer.echo(text)
    typer.echo(
        f"\nRecall={diagnostic.recall:.3f}  "
        f"Reasoning={diagnostic.reasoning:.3f}  "
        f"Presentation={diagnostic.presentation:.3f}  "
        f"(bottleneck: {diagnostic.bottleneck})"
    )
