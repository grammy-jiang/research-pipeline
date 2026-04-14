"""CLI handler for the 'expand' command.

Fetches citation and reference graphs for specified papers using the
Semantic Scholar API.  The user (or agent) must explicitly provide
paper IDs — the command does not autonomously select papers to expand.
"""

import logging
from pathlib import Path

from research_pipeline.config.loader import load_config
from research_pipeline.infra.rate_limit import RateLimiter
from research_pipeline.sources.citation_graph import CitationGraphClient
from research_pipeline.storage.manifests import write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_expand(
    paper_ids: list[str],
    direction: str = "both",
    limit_per_paper: int = 50,
    reference_boost: float = 1.0,
    bfs_depth: int = 0,
    bfs_top_k: int = 10,
    query_terms: list[str] | None = None,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute citation-graph expansion for the given paper IDs.

    Args:
        paper_ids: arXiv IDs or S2 paper IDs to expand.
        direction: "citations", "references", or "both".
        limit_per_paper: Max related papers per seed paper.
        reference_boost: Multiplier for reference (backward) limit when
            direction is "both".  E.g. 2.0 fetches twice as many
            references as citations.  Default 1.0 (equal).
        bfs_depth: BFS expansion depth (0 = single-hop only).
        bfs_top_k: Papers to keep per BFS hop after BM25 pruning.
        query_terms: Query terms for BFS BM25 hop pruning.
        config_path: Path to config TOML file.
        workspace: Workspace root directory.
        run_id: Pipeline run ID.
    """
    if not paper_ids:
        logger.error("No paper IDs provided. Use --paper-ids.")
        return

    config = load_config(config_path)
    ws = workspace or Path(config.workspace)

    if not run_id:
        logger.error("--run-id is required for the expand command.")
        return

    _run_id, run_root = init_run(ws, run_id)
    expand_dir = get_stage_dir(run_root, "expand")
    expand_dir.mkdir(parents=True, exist_ok=True)

    s2_api_key = config.sources.semantic_scholar_api_key
    rate_limiter = RateLimiter(
        min_interval=config.sources.semantic_scholar_min_interval,
        name="s2_expand",
    )

    client = CitationGraphClient(
        api_key=s2_api_key,
        rate_limiter=rate_limiter,
    )

    logger.info(
        "Expanding citation graph for %d papers (direction=%s, limit=%d)",
        len(paper_ids),
        direction,
        limit_per_paper,
    )

    candidates = client.fetch_related(
        paper_ids=paper_ids,
        direction=direction,
        limit_per_paper=limit_per_paper,
        reference_boost=reference_boost,
    )

    # BFS multi-hop expansion if requested
    if bfs_depth > 0 and query_terms:
        logger.info(
            "BFS expansion: depth=%d, top_k=%d, terms=%s",
            bfs_depth,
            bfs_top_k,
            query_terms,
        )
        bfs_candidates = client.bfs_expand(
            seed_ids=paper_ids,
            query_terms=query_terms,
            max_depth=bfs_depth,
            limit_per_paper=limit_per_paper,
            top_k_per_hop=bfs_top_k,
            direction=direction,
        )
        # Merge and deduplicate
        seen_ids = {c.arxiv_id for c in candidates}
        for bc in bfs_candidates:
            if bc.arxiv_id not in seen_ids:
                candidates.append(bc)
                seen_ids.add(bc.arxiv_id)
        logger.info(
            "BFS added %d unique papers",
            len(candidates) - len(seen_ids) + len(bfs_candidates),
        )

    # Write expanded candidates
    output_path = expand_dir / "expanded_candidates.jsonl"
    records = [c.model_dump(mode="json") for c in candidates]
    write_jsonl(records, output_path)

    logger.info(
        "Expansion complete: %d related papers written to %s",
        len(candidates),
        output_path,
    )
