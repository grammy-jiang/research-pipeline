"""CLI command: cluster — group screened papers by topic similarity."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.screening.clustering import cluster_candidates
from research_pipeline.storage.workspace import get_stage_dir
from research_pipeline.summarization.bibtex_export import load_candidates_from_jsonl

logger = logging.getLogger(__name__)


def cluster_cmd(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        help="Pipeline run ID.",
    ),
    stage: str = typer.Option(
        "screen",
        "--stage",
        help="Stage to cluster from: search or screen.",
    ),
    threshold: float = typer.Option(
        0.15,
        "--threshold",
        "-t",
        help="Cosine similarity threshold (0-1). Lower = fewer, larger clusters.",
    ),
    output: str = typer.Option(
        "",
        "-o",
        "--output",
        help="Output JSON file path (default: auto in run dir).",
    ),
) -> None:
    """Cluster papers by topic similarity using TF-IDF.

    Groups screened candidates into topically coherent clusters for
    better organization before synthesis.

    Example::

        research-pipeline cluster --run-id <RUN_ID>
        research-pipeline cluster --run-id <RUN_ID> --threshold 0.2
    """
    cfg = load_config()
    stage_dir = get_stage_dir(cfg.runs_dir, run_id, stage)

    jsonl_candidates = sorted(stage_dir.glob("*.jsonl"))
    if not jsonl_candidates:
        logger.error("No candidate JSONL files found in %s", stage_dir)
        raise typer.Exit(1)

    jsonl_path = jsonl_candidates[-1]
    logger.info("Loading candidates from %s", jsonl_path)

    candidates = load_candidates_from_jsonl(jsonl_path)
    if not candidates:
        logger.warning("No candidates found in %s", jsonl_path)
        raise typer.Exit(1)

    clusters = cluster_candidates(candidates, threshold=threshold)

    result = {
        "run_id": run_id,
        "threshold": threshold,
        "num_papers": len(candidates),
        "num_clusters": len(clusters),
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "paper_count": len(c.paper_ids),
                "paper_ids": c.paper_ids,
                "top_terms": c.top_terms,
            }
            for c in clusters
        ],
    }

    out_path = Path(output) if output else stage_dir / "clusters.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Clustered %d papers into %d groups → %s",
        len(candidates),
        len(clusters),
        out_path,
    )

    # Print summary
    for c in clusters:
        logger.info(
            "  Cluster %d (%s): %d papers",
            c.cluster_id,
            c.label,
            len(c.paper_ids),
        )
