"""CLI command for KG quality evaluation."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import typer

logger = logging.getLogger(__name__)


def kg_quality_command(
    db_path: str = typer.Option(
        "",
        "--db",
        help=(
            "Path to KG SQLite database."
            " Default: ~/.cache/research-pipeline/knowledge_graph.db"
        ),
    ),
    staleness_days: float = typer.Option(
        365.0,
        "--staleness-days",
        help="Threshold in days for a triple to be considered stale.",
    ),
    sample_size: int = typer.Option(
        0,
        "--sample",
        help="If > 0, also run TWCS sampling and print sample.",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON.",
    ),
) -> None:
    """Evaluate knowledge graph quality across 5 dimensions.

    Computes structural, consistency, completeness, timeliness, and
    redundancy metrics using the three-layer composable architecture
    (TKDE 2022 + Text2KGBench + LLM-KG-Bench).
    """
    from research_pipeline.quality.kg_quality import (
        evaluate_kg_quality,
        sample_triples_twcs,
    )
    from research_pipeline.storage.knowledge_graph import (
        DEFAULT_KG_PATH,
    )

    path = Path(db_path) if db_path else DEFAULT_KG_PATH

    if not path.exists():
        logger.error("KG database not found: %s", path)
        raise typer.Exit(code=1)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    try:
        score = evaluate_kg_quality(conn, staleness_days=staleness_days)

        if output_json:
            typer.echo(json.dumps(score.to_dict(), indent=2))
        else:
            typer.echo(f"=== KG Quality Evaluation: {path.name} ===")
            typer.echo(f"Composite Score: {score.composite:.4f}")
            typer.echo(f"  Accuracy:      {score.accuracy:.4f}")
            typer.echo(f"  Consistency:   {score.consistency:.4f}")
            typer.echo(f"  Completeness:  {score.completeness:.4f}")
            typer.echo(f"  Timeliness:    {score.timeliness:.4f}")
            typer.echo(f"  Redundancy:    {score.redundancy:.4f}")
            typer.echo("")
            s = score.structural
            typer.echo(
                f"Structural: {s.num_entities} entities, "
                f"{s.num_triples} triples, ICR={s.icr:.2f}, "
                f"density={s.density:.2f}, "
                f"components={s.connected_components}"
            )
            c = score.consistency_detail
            typer.echo(
                f"Consistency: IC={c.ic_score:.2f}, "
                f"EC={c.ec_score:.2f}, "
                f"contradictions={c.ic_contradiction_count}, "
                f"duplicates={c.duplicate_triples}"
            )
            comp = score.completeness_detail
            typer.echo(
                f"Completeness: types={comp.entity_type_coverage:.2f}, "
                f"relations={comp.relation_type_coverage:.2f}, "
                f"orphans={comp.orphan_entities}"
            )

        if sample_size > 0:
            sample = sample_triples_twcs(conn, sample_size=sample_size)
            if output_json:
                typer.echo(json.dumps(sample, indent=2))
            else:
                typer.echo(f"\n--- TWCS Sample ({len(sample)} triples) ---")
                for t in sample:
                    typer.echo(
                        f"  {t['subject_id']} --[{t['relation']}]--> {t['object_id']}"
                    )

        logger.info("KG quality evaluation complete: composite=%.4f", score.composite)
    finally:
        conn.close()
