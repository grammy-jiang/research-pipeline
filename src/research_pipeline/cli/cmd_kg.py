"""CLI handlers for knowledge graph commands (kg-stats, kg-query, kg-ingest)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from research_pipeline.storage.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def run_kg_stats(db_path: Path | None = None) -> None:
    """Show knowledge graph statistics.

    Args:
        db_path: Optional path to the KG database.
    """
    kg = KnowledgeGraph(db_path=db_path)
    try:
        s = kg.stats()
        typer.echo(f"Total entities: {s['total_entities']}")
        typer.echo(f"Total triples: {s['total_triples']}")

        entities: dict[str, int] = s.get("entities", {})  # type: ignore[assignment]
        if entities:
            typer.echo("\nEntities by type:")
            for etype, count in sorted(entities.items()):
                typer.echo(f"  {etype}: {count}")

        triples: dict[str, int] = s.get("triples", {})  # type: ignore[assignment]
        if triples:
            typer.echo("\nTriples by relation:")
            for rtype, count in sorted(triples.items()):
                typer.echo(f"  {rtype}: {count}")
    finally:
        kg.close()


def run_kg_query(entity_id: str, db_path: Path | None = None) -> None:
    """Query knowledge graph for an entity and its neighbors.

    Args:
        entity_id: Entity identifier to look up.
        db_path: Optional path to the KG database.
    """
    kg = KnowledgeGraph(db_path=db_path)
    try:
        entity = kg.get_entity(entity_id)
        if entity is None:
            typer.echo(f"Entity not found: {entity_id}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Entity: {entity.name}")
        typer.echo(f"Type: {entity.entity_type.value}")
        if entity.properties:
            typer.echo(f"Properties: {entity.properties}")

        neighbors = kg.get_neighbors(entity_id)
        if neighbors:
            typer.echo(f"\nRelations ({len(neighbors)}):")
            for t in neighbors:
                direction = "→" if t.subject_id == entity_id else "←"
                other = t.object_id if t.subject_id == entity_id else t.subject_id
                typer.echo(
                    f"  {direction} [{t.relation.value}] {other}"
                    f" (conf: {t.confidence:.2f})"
                )
        else:
            typer.echo("\nNo relations found.")
    finally:
        kg.close()


def run_kg_ingest(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Ingest pipeline results into the knowledge graph.

    Args:
        config_path: Path to config TOML file.
        workspace: Workspace root directory.
        run_id: Pipeline run ID.
        db_path: Optional path to the KG database.
    """
    from research_pipeline.config.loader import load_config
    from research_pipeline.models.candidate import CandidateRecord
    from research_pipeline.models.claim import ClaimDecomposition
    from research_pipeline.storage.manifests import read_jsonl
    from research_pipeline.storage.workspace import get_stage_dir, init_run

    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id_str, run_root = init_run(ws, run_id)

    kg = KnowledgeGraph(db_path=db_path)
    try:
        total = 0

        # Ingest candidates from screen shortlist
        screen_dir = get_stage_dir(run_root, "screen")
        shortlist_path = screen_dir / "shortlist.jsonl"
        if shortlist_path.exists():
            raw = read_jsonl(shortlist_path)
            candidates = [CandidateRecord.model_validate(d) for d in raw]
            added = kg.ingest_from_candidates(candidates, run_id=run_id_str)
            typer.echo(f"Ingested {added} paper entities from candidates")
            total += added

        # Ingest claims from claim decomposition
        claims_dir = get_stage_dir(run_root, "summarize") / "claims"
        claims_path = claims_dir / "claim_decomposition.jsonl"
        if claims_path.exists():
            raw = read_jsonl(claims_path)
            for d in raw:
                decomp = ClaimDecomposition.model_validate(d)
                added = kg.ingest_from_claims(decomp, run_id=run_id_str)
                total += added
            typer.echo(f"Ingested claims from {len(raw)} papers")

        typer.echo(f"\nTotal ingested: {total} items")
        s = kg.stats()
        typer.echo(
            f"KG now has {s['total_entities']} entities,"
            f" {s['total_triples']} triples"
        )
    finally:
        kg.close()
