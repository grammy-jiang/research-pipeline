from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from research_pipeline.mcp_server.schemas import (
    KGIngestInput,
    KGQualityInput,
    KGQueryInput,
    KGStatsInput,
    ToolResult,
)
from research_pipeline.mcp_server.tools._common import (
    _raise_tool_error,
    _report_progress,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


def kg_quality_tool(
    params: KGQualityInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Evaluate knowledge graph quality across 5 dimensions.

    Uses the three-layer composable architecture (structural metrics,
    IC+EC consistency, TWCS sampling) to produce a composite score.
    """
    try:
        import sqlite3

        from research_pipeline.quality.kg_quality import (
            evaluate_kg_quality,
            sample_triples_twcs,
        )
        from research_pipeline.storage.knowledge_graph import DEFAULT_KG_PATH

        db_path = Path(params.db_path) if params.db_path else DEFAULT_KG_PATH
        if not db_path.exists():
            return ToolResult(
                success=False,
                message=f"KG database not found: {db_path}",
            )

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        try:
            score = evaluate_kg_quality(conn, staleness_days=params.staleness_days)

            result: dict[str, Any] = score.to_dict()

            if params.sample_size > 0:
                sample = sample_triples_twcs(conn, sample_size=params.sample_size)
                result["twcs_sample"] = sample

            return ToolResult(
                success=True,
                message=(
                    f"KG quality: composite={score.composite:.4f}, "
                    f"accuracy={score.accuracy:.4f}, "
                    f"consistency={score.consistency:.4f}, "
                    f"completeness={score.completeness:.4f}"
                ),
                artifacts=result,
            )
        finally:
            conn.close()

    except Exception as exc:
        _raise_tool_error("KG quality evaluation", exc)


def kg_stats_tool(params: KGStatsInput, ctx: Context | None = None) -> ToolResult:
    """Show knowledge graph statistics."""
    try:
        from research_pipeline.storage.knowledge_graph import KnowledgeGraph

        db_path = Path(params.db_path) if params.db_path else None
        kg = KnowledgeGraph(db_path=db_path)
        try:
            stats = kg.stats()
        finally:
            kg.close()

        return ToolResult(
            success=True,
            message=(
                f"KG has {stats['total_entities']} entities, "
                f"{stats['total_triples']} triples."
            ),
            artifacts=stats,
        )
    except Exception as exc:
        _raise_tool_error("kg_stats", exc)


def kg_query_tool(params: KGQueryInput, ctx: Context | None = None) -> ToolResult:
    """Query an entity and its relations in the knowledge graph."""
    try:
        from research_pipeline.storage.knowledge_graph import KnowledgeGraph

        db_path = Path(params.db_path) if params.db_path else None
        kg = KnowledgeGraph(db_path=db_path)
        try:
            entity = kg.get_entity(params.entity_id)
            if entity is None:
                return ToolResult(
                    success=False,
                    message=f"Entity not found: {params.entity_id}",
                )

            neighbors = kg.get_neighbors(params.entity_id)

            entity_data = {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "type": entity.entity_type.value,
                "properties": entity.properties,
            }

            relations = []
            for t in neighbors:
                direction = (
                    "outgoing" if t.subject_id == params.entity_id else "incoming"
                )
                other = (
                    t.object_id if t.subject_id == params.entity_id else t.subject_id
                )
                relations.append(
                    {
                        "direction": direction,
                        "relation": t.relation.value,
                        "other_entity": other,
                        "confidence": t.confidence,
                    }
                )
        finally:
            kg.close()

        return ToolResult(
            success=True,
            message=(
                f"Entity '{entity.name}' ({entity.entity_type.value}) "
                f"with {len(relations)} relations."
            ),
            artifacts={
                "entity": entity_data,
                "relations": relations,
            },
        )
    except Exception as exc:
        _raise_tool_error("kg_query", exc)


def kg_ingest_tool(params: KGIngestInput, ctx: Context | None = None) -> ToolResult:
    """Ingest pipeline results into the knowledge graph."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.models.candidate import CandidateRecord
        from research_pipeline.models.claim import ClaimDecomposition
        from research_pipeline.storage.knowledge_graph import KnowledgeGraph
        from research_pipeline.storage.manifests import read_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        config = load_config()
        ws = Path(params.workspace) if params.workspace else Path(config.workspace)
        run_id_str, run_root = init_run(ws, params.run_id or None)

        db_path = Path(params.db_path) if params.db_path else None
        kg = KnowledgeGraph(db_path=db_path)

        try:
            total = 0
            claim_papers = 0

            _report_progress(ctx, 0, 2, "Ingesting candidates")

            screen_dir = get_stage_dir(run_root, "screen")
            shortlist_path = screen_dir / "shortlist.jsonl"
            if shortlist_path.exists():
                raw = read_jsonl(shortlist_path)
                candidates = [CandidateRecord.model_validate(d) for d in raw]
                added = kg.ingest_from_candidates(candidates, run_id=run_id_str)
                total += added

            _report_progress(ctx, 1, 2, "Ingesting claims")

            claims_dir = get_stage_dir(run_root, "summarize") / "claims"
            claims_path = claims_dir / "claim_decomposition.jsonl"
            if claims_path.exists():
                raw = read_jsonl(claims_path)
                for d in raw:
                    decomp = ClaimDecomposition.model_validate(d)
                    added = kg.ingest_from_claims(decomp, run_id=run_id_str)
                    total += added
                    claim_papers += 1

            stats = kg.stats()
        finally:
            kg.close()

        _report_progress(ctx, 2, 2, "Done")
        return ToolResult(
            success=True,
            message=(
                f"Ingested {total} items. KG now has "
                f"{stats['total_entities']} entities, "
                f"{stats['total_triples']} triples."
            ),
            artifacts={
                "total_ingested": total,
                "claim_papers": claim_papers,
                "kg_entities": stats["total_entities"],
                "kg_triples": stats["total_triples"],
            },
        )
    except Exception as exc:
        _raise_tool_error("kg_ingest", exc)
