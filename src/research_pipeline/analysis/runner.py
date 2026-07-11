"""Claim-analysis stage runner (#109).

Core stage logic for the `analyze-claims` step, shared by the CLI wrapper
(`cli/cmd_analyze_claims.py`) and the pipeline orchestrator. Typer-free: it uses
``logging`` for progress and raises :class:`FileNotFoundError` on missing input;
the CLI wrapper maps that to ``typer.Exit`` so the presentation concern stays in
the presentation layer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from research_pipeline.analysis.decomposer import decompose_paper
from research_pipeline.config.loader import load_config
from research_pipeline.models.summary import PaperSummary
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_analyze_claims(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute claim decomposition on paper summaries.

    Args:
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with summaries.

    Raises:
        FileNotFoundError: If no paper summaries exist for the run.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id_str, run_root = init_run(ws, run_id)

    # Read paper summaries
    summary_dir = get_stage_dir(run_root, "summarize")
    summary_path = summary_dir / "paper_summaries.jsonl"
    if not summary_path.exists():
        # Also try individual summary files
        summary_files = list(summary_dir.glob("*.summary.json"))
        if not summary_files:
            raise FileNotFoundError("no paper summaries found. Run 'summarize' first.")
        raw = []
        for sf in summary_files:
            raw.append(json.loads(sf.read_text(encoding="utf-8")))
    else:
        raw = read_jsonl(summary_path)

    summaries = [PaperSummary.model_validate(d) for d in raw]

    # Get markdown directory for evidence retrieval
    md_dir = get_stage_dir(run_root, "convert")

    results = []
    for summary in summaries:
        md_path = md_dir / f"{summary.arxiv_id}.md"
        if not md_path.exists():
            md_path = md_dir / f"{summary.arxiv_id}{summary.version}.md"

        markdown_path_str = str(md_path) if md_path.exists() else None

        decomp = decompose_paper(
            summary=summary,
            markdown_path=markdown_path_str,
        )
        results.append(decomp)

        supported = decomp.evidence_summary.get("supported", 0)
        total = decomp.total_claims
        logger.info(
            "Decomposed %s: %d claims, %d supported",
            summary.arxiv_id,
            total,
            supported,
        )

    # Write results
    analysis_dir = get_stage_dir(run_root, "summarize") / "claims"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / "claim_decomposition.jsonl"
    write_jsonl(output_path, [r.model_dump(mode="json") for r in results])

    total_claims = sum(r.total_claims for r in results)
    total_supported = sum(r.evidence_summary.get("supported", 0) for r in results)

    logger.info("Papers analyzed: %d", len(results))
    logger.info("Total claims: %d", total_claims)
    logger.info("Supported: %d", total_supported)
    logger.info("Output: %s", output_path)
