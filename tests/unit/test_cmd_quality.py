"""Regression tests for the quality CLI stage (#28)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from research_pipeline.cli.cmd_quality import run_quality
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.screening import CheapScoreBreakdown, RelevanceDecision
from research_pipeline.storage.workspace import get_stage_dir, init_run


def _decision(arxiv_id: str) -> RelevanceDecision:
    candidate = CandidateRecord(
        arxiv_id=arxiv_id,
        version="v1",
        title=f"Paper {arxiv_id}",
        authors=["A"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
        categories=["cs.AI"],
        primary_category="cs.AI",
        abstract="Test abstract",
        abs_url="",
        pdf_url="",
        source="test",
    )
    cheap = CheapScoreBreakdown(
        bm25_title=0.0,
        bm25_abstract=0.0,
        cat_match=0.0,
        negative_penalty=0.0,
        recency_bonus=0.0,
        cheap_score=0.8,
    )
    return RelevanceDecision(
        paper=candidate,
        cheap=cheap,
        final_score=0.8,
        download=True,
        download_reason="score_threshold",
    )


def test_quality_scores_a_screen_shortlist(tmp_path: Path) -> None:
    """quality must score a screen-produced shortlist.json without crashing.

    Regression for #28: the stage parsed each RelevanceDecision-shaped entry
    as a CandidateRecord and raised a ValidationError on every normal run.
    """
    _run_id, run_root = init_run(tmp_path, "run-quality-28")
    screen_dir = get_stage_dir(run_root, "screen")
    screen_dir.mkdir(parents=True, exist_ok=True)
    shortlist = [_decision("2401.00001"), _decision("2401.00002")]
    (screen_dir / "shortlist.json").write_text(
        json.dumps([d.model_dump(mode="json") for d in shortlist], default=str),
        encoding="utf-8",
    )

    run_quality(workspace=tmp_path, run_id="run-quality-28")

    scores_path = get_stage_dir(run_root, "quality") / "quality_scores.jsonl"
    assert scores_path.exists()
    lines = [ln for ln in scores_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
