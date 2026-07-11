"""CLI handler for the 'screen' command."""

import json
import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.models.screening import RelevanceDecision
from research_pipeline.screening.heuristic import score_candidates, select_topk
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_screen(
    resume: bool = False,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    diversity: bool | None = None,
    diversity_lambda: float | None = None,
) -> None:
    """Execute the screen stage: two-pass relevance filtering.

    Args:
        resume: Skip if already completed.
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with search results.
        diversity: Enable diversity-aware MMR reranking. Overrides config.
        diversity_lambda: Relevance vs diversity balance. Overrides config.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id_str, run_root = init_run(ws, run_id)

    # Load plan
    plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
    if not plan_path.exists():
        typer.echo("Error: no query plan found. Run 'plan' first.", err=True)
        raise typer.Exit(1)
    plan = QueryPlan.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))

    # Load candidates
    candidates_path = get_stage_dir(run_root, "search") / "candidates.jsonl"
    if not candidates_path.exists():
        typer.echo("Error: no candidates found. Run 'search' first.", err=True)
        raise typer.Exit(1)
    raw = read_jsonl(candidates_path)
    candidates = [CandidateRecord.model_validate(d) for d in raw]

    # Score
    scores = score_candidates(
        candidates,
        must_terms=plan.must_terms,
        nice_terms=plan.nice_terms,
        negative_terms=plan.negative_terms,
        target_categories=plan.candidate_categories,
    )

    screen_dir = get_stage_dir(run_root, "screen")
    write_jsonl(
        screen_dir / "cheap_scores.jsonl",
        [s.model_dump(mode="json") for s in scores],
    )

    use_diversity = diversity if diversity is not None else config.screen.diversity
    use_lambda = (
        diversity_lambda
        if diversity_lambda is not None
        else config.screen.diversity_lambda
    )

    top = select_topk(
        candidates,
        scores,
        top_k=config.screen.cheap_top_k,
        diversity=use_diversity,
        diversity_lambda=use_lambda,
    )

    shortlist = []
    for candidate, score in top[: config.screen.download_top_n]:
        decision = RelevanceDecision(
            paper=candidate,
            cheap=score,
            llm=None,
            final_score=score.cheap_score,
            download=True,
            download_reason="score_threshold",
        )
        shortlist.append(decision)

    shortlist_path = screen_dir / "shortlist.json"
    shortlist_path.write_text(
        json.dumps(
            [d.model_dump(mode="json") for d in shortlist], indent=2, default=str
        ),
        encoding="utf-8",
    )

    # Query-typed stopping analysis — classify query intent and record
    # the stopping profile so downstream stages can adapt behaviour.
    from research_pipeline.screening.typed_stopping import (
        TypedStoppingEvaluator,
        classify_query_type,
        estimate_cost,
    )

    query_type = classify_query_type(plan.topic_raw)
    cost_est = estimate_cost(plan.topic_raw)
    evaluator = TypedStoppingEvaluator(
        query=plan.topic_raw,
        query_type=query_type,
        reclassify_after_n=config.screen.reclassify_after_n,
    )
    batch_scores = [s.cheap_score for s in scores]
    if batch_scores:
        # Feed retrieved titles + abstracts so opt-in re-classification can revise
        # the topic-only query type from actual evidence (#111). Inert unless
        # config.screen.reclassify_after_n > 0.
        batch_texts = [
            f"{candidate.title}. {candidate.abstract}" for candidate, _ in top
        ]
        evaluator.add_batch(batch_scores, texts=batch_texts)

    final_type = evaluator.query_type
    stopping_metadata = {
        "query_type": final_type.value,
        "stopping_profile": evaluator.profile.to_dict(),
        "cost_estimate": cost_est.to_dict(),
        "evaluation": evaluator.evaluate().to_dict(),
    }
    stopping_path = screen_dir / "typed_stopping.json"
    stopping_path.write_text(json.dumps(stopping_metadata, indent=2), encoding="utf-8")
    if evaluator.reclassified:
        logger.info(
            "Re-classified query type from retrieved evidence: %s → %s",
            query_type.value,
            final_type.value,
        )
    logger.info(
        "Query type: %s (profile: %s)", final_type.value, evaluator.profile.description
    )

    # Query refinement feedback — suggest terms for iterative improvement
    from research_pipeline.screening.query_feedback import compute_query_refinement

    top_papers = [c for c, _ in top[: config.screen.download_top_n]]
    # Pass the full screened pool so a right-but-unretrieved term (present in the
    # pool, absent from the top-K) is not pruned as a self-confirming loop (#123).
    screened_pool = [c for c, _ in top]
    refinement = compute_query_refinement(plan, top_papers, screened_pool=screened_pool)
    refinement_path = screen_dir / "query_refinement.json"
    refinement_path.write_text(refinement.model_dump_json(indent=2), encoding="utf-8")
    if refinement.suggested_additions:
        logger.info(
            "Query refinement: suggested additions: %s",
            ", ".join(refinement.suggested_additions),
        )
    if refinement.suggested_removals:
        logger.info(
            "Query refinement: low-coverage terms: %s",
            ", ".join(refinement.suggested_removals),
        )

    typer.echo(f"Screened {len(candidates)} → {len(shortlist)} shortlisted")
    typer.echo(f"Query type: {final_type.value}")
    typer.echo(f"Saved to: {shortlist_path}")
    if refinement.suggested_additions:
        typer.echo(
            f"Suggested query additions: {', '.join(refinement.suggested_additions)}"
        )
    logger.info("Screen stage complete: %d shortlisted", len(shortlist))
