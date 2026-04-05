"""CLI handler for the 'quality' command.

Computes multi-dimensional quality scores for paper candidates
using citation metrics, venue reputation, and author credibility.
"""

import logging
from pathlib import Path

from research_pipeline.config.loader import load_config
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.quality.composite import compute_quality_score
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_quality(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute quality scoring for candidates in a pipeline run.

    Args:
        config_path: Path to config TOML file.
        workspace: Workspace root directory.
        run_id: Pipeline run ID.
    """
    if not run_id:
        logger.error("--run-id is required for the quality command.")
        return

    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    qc = config.quality

    weights = {
        "citation_weight": qc.citation_weight,
        "venue_weight": qc.venue_weight,
        "author_weight": qc.author_weight,
        "recency_weight": qc.recency_weight,
    }

    _run_id, run_root = init_run(ws, run_id)

    # Try to load candidates from screen stage, then search stage
    screen_dir = get_stage_dir(run_root, "screen")
    search_dir = get_stage_dir(run_root, "search")

    candidates_path = screen_dir / "shortlist.jsonl"
    if not candidates_path.exists():
        candidates_path = search_dir / "candidates.jsonl"

    if not candidates_path.exists():
        logger.error("No candidates found at %s", candidates_path)
        return

    raw_records = read_jsonl(candidates_path)
    candidates = [CandidateRecord(**r) for r in raw_records]
    logger.info("Scoring %d candidates from %s", len(candidates), candidates_path)

    quality_dir = get_stage_dir(run_root, "quality")
    quality_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    for candidate in candidates:
        qs = compute_quality_score(
            candidate,
            weights=weights,
            venue_data_path=qc.venue_data_path,
        )
        scores.append(qs.model_dump(mode="json"))

    output_path = quality_dir / "quality_scores.jsonl"
    write_jsonl(scores, output_path)

    logger.info(
        "Quality scoring complete: %d scores written to %s",
        len(scores),
        output_path,
    )
