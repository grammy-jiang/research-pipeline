"""Bidirectional citation snowball expansion with budget-aware stopping.

Implements iterative snowball sampling: each round uses newly discovered
high-relevance papers as seeds for the next round.  Stops when budget
is exhausted, marginal relevance decays below threshold, or category
diversity saturates.

References:
    - Deep Research Report §FD4: forward+backward bidirectional snowball
      with budget-aware stopping
    - Nogueira 2020: BM25→backward→rerank, +10–20pp Recall@1000
    - Arafat 2025: BFS expansion +24pp completeness at 27ms
"""

from __future__ import annotations

import logging

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.snowball import (
    SnowballBudget,
    SnowballResult,
    SnowballRound,
    StopReason,
)
from research_pipeline.sources.citation_graph import (
    CitationGraphClient,
    _bm25_score_text,
)

logger = logging.getLogger(__name__)


def _compute_median_score(
    candidates: list[CandidateRecord],
    query_terms: list[str],
) -> float:
    """Compute median BM25 score for a list of candidates.

    Args:
        candidates: Papers to score.
        query_terms: Terms for BM25 scoring.

    Returns:
        Median score, or 0.0 if the list is empty.
    """
    if not candidates or not query_terms:
        return 0.0
    scores = sorted(
        _bm25_score_text(f"{c.title} {c.abstract}", query_terms) for c in candidates
    )
    mid = len(scores) // 2
    if len(scores) % 2 == 0 and len(scores) > 1:
        return (scores[mid - 1] + scores[mid]) / 2
    return scores[mid]


def _extract_categories(candidates: list[CandidateRecord]) -> set[str]:
    """Extract all unique categories from candidates.

    Args:
        candidates: Papers to extract categories from.

    Returns:
        Set of unique category strings.
    """
    cats: set[str] = set()
    for c in candidates:
        if c.primary_category:
            cats.add(c.primary_category)
        cats.update(c.categories)
    return cats


def snowball_expand(
    client: CitationGraphClient,
    seed_ids: list[str],
    query_terms: list[str],
    budget: SnowballBudget | None = None,
    existing_candidates: list[CandidateRecord] | None = None,
) -> tuple[list[CandidateRecord], SnowballResult]:
    """Run bidirectional snowball expansion with budget-aware stopping.

    Each round:
    1. Fetch citations and/or references for current seeds
    2. Deduplicate against all previously seen papers
    3. Score new papers against query terms (BM25)
    4. Check stopping criteria:
       - Max rounds reached
       - Max total papers reached
       - Marginal relevance decay (consecutive low-relevance rounds)
       - Category diversity saturation
       - No new papers found
    5. Select top-scoring new papers as seeds for next round

    Args:
        client: Semantic Scholar citation graph client.
        seed_ids: Starting paper IDs.
        query_terms: Terms for relevance scoring.
        budget: Budget constraints (uses defaults if None).
        existing_candidates: Previously discovered candidates (for
            computing baseline median score).

    Returns:
        Tuple of (all discovered candidates, snowball result stats).
    """
    if budget is None:
        budget = SnowballBudget()

    if not seed_ids:
        return [], SnowballResult(
            seed_ids=[],
            query_terms=query_terms,
            budget=budget,
            stop_reason=StopReason.NO_NEW_PAPERS,
        )

    # Track all seen paper IDs and candidates
    seen_ids: set[str] = set(seed_ids)
    all_discovered: list[CandidateRecord] = []
    all_categories: set[str] = set()
    rounds: list[SnowballRound] = []
    api_calls = 0

    # Compute baseline median from existing candidates
    baseline_median = 0.0
    if existing_candidates and query_terms:
        baseline_median = _compute_median_score(existing_candidates, query_terms)
        all_categories = _extract_categories(existing_candidates)
        logger.info(
            "Baseline: median_score=%.3f, categories=%d",
            baseline_median,
            len(all_categories),
        )

    current_seeds = list(seed_ids)
    low_relevance_streak = 0
    no_new_category_streak = 0

    for round_num in range(1, budget.max_rounds + 1):
        logger.info(
            "Snowball round %d: %d seeds, %d total discovered so far",
            round_num,
            len(current_seeds),
            len(all_discovered),
        )

        # Fetch citations/references for current seeds
        round_candidates: list[CandidateRecord] = []
        round_api_calls = 0

        citation_limit = budget.limit_per_paper
        reference_limit = budget.limit_per_paper
        if budget.direction == "both" and budget.reference_boost > 1.0:
            reference_limit = int(budget.limit_per_paper * budget.reference_boost)

        for pid in current_seeds:
            if budget.direction in ("citations", "both"):
                cits = client.get_citations(pid, citation_limit)
                round_candidates.extend(cits)
                round_api_calls += 1

            if budget.direction in ("references", "both"):
                refs = client.get_references(pid, reference_limit)
                round_candidates.extend(refs)
                round_api_calls += 1

        api_calls += round_api_calls
        fetched_count = len(round_candidates)

        # Deduplicate: remove already-seen papers
        new_candidates: list[CandidateRecord] = []
        for c in round_candidates:
            if c.arxiv_id not in seen_ids:
                seen_ids.add(c.arxiv_id)
                new_candidates.append(c)

        # Further deduplicate within this round
        unique_in_round: dict[str, CandidateRecord] = {}
        for c in new_candidates:
            if c.arxiv_id not in unique_in_round:
                unique_in_round[c.arxiv_id] = c
        new_candidates = list(unique_in_round.values())
        new_count = len(new_candidates)

        # Compute relevance scores
        median_threshold = baseline_median
        if all_discovered and query_terms:
            median_threshold = max(
                baseline_median,
                _compute_median_score(all_discovered, query_terms),
            )

        relevant_count = 0
        scored_new: list[tuple[CandidateRecord, float]] = []
        for c in new_candidates:
            score = (
                _bm25_score_text(f"{c.title} {c.abstract}", query_terms)
                if query_terms
                else 0.0
            )
            scored_new.append((c, score))
            if score >= median_threshold:
                relevant_count += 1

        relevance_fraction = relevant_count / new_count if new_count > 0 else 0.0

        # Track category diversity
        round_categories = _extract_categories(new_candidates)
        new_categories = round_categories - all_categories
        all_categories.update(round_categories)

        # Record round stats
        round_stat = SnowballRound(
            round_number=round_num,
            seeds_count=len(current_seeds),
            fetched_count=fetched_count,
            new_count=new_count,
            relevant_count=relevant_count,
            relevance_fraction=relevance_fraction,
            unique_categories=len(all_categories),
            new_categories=len(new_categories),
        )
        rounds.append(round_stat)

        # Add all new candidates to the discovered pool
        all_discovered.extend(new_candidates)

        logger.info(
            "Round %d: fetched=%d, new=%d, relevant=%d (%.1f%%), "
            "categories=%d (+%d new)",
            round_num,
            fetched_count,
            new_count,
            relevant_count,
            relevance_fraction * 100,
            len(all_categories),
            len(new_categories),
        )

        # --- Stopping criteria ---

        # No new papers
        if new_count == 0:
            logger.info("Stopping: no new papers found in round %d", round_num)
            return all_discovered, SnowballResult(
                seed_ids=list(seed_ids),
                query_terms=query_terms,
                budget=budget,
                rounds=rounds,
                total_discovered=len(all_discovered),
                stop_reason=StopReason.NO_NEW_PAPERS,
                api_calls=api_calls,
            )

        # Max total papers
        if len(all_discovered) >= budget.max_total_papers:
            logger.info(
                "Stopping: reached max papers (%d >= %d)",
                len(all_discovered),
                budget.max_total_papers,
            )
            return all_discovered[: budget.max_total_papers], SnowballResult(
                seed_ids=list(seed_ids),
                query_terms=query_terms,
                budget=budget,
                rounds=rounds,
                total_discovered=min(len(all_discovered), budget.max_total_papers),
                stop_reason=StopReason.MAX_PAPERS,
                api_calls=api_calls,
            )

        # Marginal relevance decay
        if relevance_fraction < budget.relevance_decay_threshold:
            low_relevance_streak += 1
            if low_relevance_streak >= budget.decay_patience:
                logger.info(
                    "Stopping: relevance decay (%.1f%% < %.1f%% for %d rounds)",
                    relevance_fraction * 100,
                    budget.relevance_decay_threshold * 100,
                    low_relevance_streak,
                )
                return all_discovered, SnowballResult(
                    seed_ids=list(seed_ids),
                    query_terms=query_terms,
                    budget=budget,
                    rounds=rounds,
                    total_discovered=len(all_discovered),
                    stop_reason=StopReason.RELEVANCE_DECAY,
                    api_calls=api_calls,
                )
        else:
            low_relevance_streak = 0

        # Diversity saturation
        if len(new_categories) == 0:
            no_new_category_streak += 1
            if no_new_category_streak >= budget.diversity_window:
                logger.info(
                    "Stopping: diversity saturated (no new categories for %d rounds)",
                    no_new_category_streak,
                )
                return all_discovered, SnowballResult(
                    seed_ids=list(seed_ids),
                    query_terms=query_terms,
                    budget=budget,
                    rounds=rounds,
                    total_discovered=len(all_discovered),
                    stop_reason=StopReason.DIVERSITY_SATURATION,
                    api_calls=api_calls,
                )
        else:
            no_new_category_streak = 0

        # Select next round's seeds: top-scoring new papers
        scored_new.sort(key=lambda x: x[1], reverse=True)
        next_seed_count = min(
            max(3, len(current_seeds)),  # at least 3 seeds
            len(scored_new),
        )
        current_seeds = [c.arxiv_id for c, _ in scored_new[:next_seed_count]]

    # Max rounds reached
    logger.info("Stopping: max rounds (%d) reached", budget.max_rounds)
    return all_discovered, SnowballResult(
        seed_ids=list(seed_ids),
        query_terms=query_terms,
        budget=budget,
        rounds=rounds,
        total_discovered=len(all_discovered),
        stop_reason=StopReason.MAX_ROUNDS,
        api_calls=api_calls,
    )


def format_snowball_report(result: SnowballResult) -> str:
    """Format a snowball result as a human-readable Markdown report.

    Args:
        result: Snowball expansion result.

    Returns:
        Markdown string with round-by-round summary.
    """
    lines = [
        "# Snowball Expansion Report",
        "",
        f"**Seeds**: {len(result.seed_ids)} papers",
        f"**Query terms**: "
        f"{', '.join(result.query_terms) if result.query_terms else '(none)'}",
        f"**Direction**: {result.budget.direction}",
        f"**Stop reason**: {result.stop_reason.value}",
        f"**Total discovered**: {result.total_discovered}",
        f"**API calls**: {result.api_calls}",
        f"**Rounds**: {len(result.rounds)}",
        "",
        "## Round-by-Round Summary",
        "",
        "| Round | Seeds | Fetched | New | Relevant | Rel% | Categories | +New |",
        "|-------|-------|---------|-----|----------|------|------------|------|",
    ]

    for r in result.rounds:
        lines.append(
            f"| {r.round_number} | {r.seeds_count} | {r.fetched_count} | "
            f"{r.new_count} | {r.relevant_count} | "
            f"{r.relevance_fraction:.1%} | {r.unique_categories} | "
            f"+{r.new_categories} |"
        )

    lines.extend(
        [
            "",
            "## Budget Configuration",
            "",
            f"- Max rounds: {result.budget.max_rounds}",
            f"- Max papers: {result.budget.max_total_papers}",
            f"- Relevance threshold: {result.budget.relevance_decay_threshold:.0%}",
            f"- Decay patience: {result.budget.decay_patience} rounds",
            f"- Diversity window: {result.budget.diversity_window} rounds",
            f"- Limit per paper: {result.budget.limit_per_paper}",
            f"- Reference boost: {result.budget.reference_boost:.1f}x",
        ]
    )

    return "\n".join(lines)
