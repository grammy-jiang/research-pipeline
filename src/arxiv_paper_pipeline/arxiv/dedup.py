"""Version deduplication and near-duplicate detection for arXiv candidates."""

import logging

from arxiv_paper_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)


def dedup_by_version(candidates: list[CandidateRecord]) -> list[CandidateRecord]:
    """Deduplicate candidates by arXiv ID, keeping the latest version.

    Args:
        candidates: List of candidate records (may contain duplicates).

    Returns:
        Deduplicated list, one entry per base arXiv ID.
    """
    best: dict[str, CandidateRecord] = {}
    for c in candidates:
        existing = best.get(c.arxiv_id)
        if existing is None:
            best[c.arxiv_id] = c
        elif c.version > existing.version:
            logger.debug(
                "Replacing %s %s with %s",
                c.arxiv_id,
                existing.version,
                c.version,
            )
            best[c.arxiv_id] = c

    deduped = list(best.values())
    removed = len(candidates) - len(deduped)
    if removed > 0:
        logger.info(
            "Dedup: %d candidates → %d (removed %d version duplicates)",
            len(candidates),
            len(deduped),
            removed,
        )
    return deduped


def dedup_across_queries(
    candidate_lists: list[list[CandidateRecord]],
) -> list[CandidateRecord]:
    """Merge candidate lists from multiple queries, deduplicating by ID.

    Args:
        candidate_lists: List of candidate lists from different queries.

    Returns:
        Single merged and deduplicated list.
    """
    merged: list[CandidateRecord] = []
    for cl in candidate_lists:
        merged.extend(cl)

    result = dedup_by_version(merged)
    logger.info(
        "Merged %d lists (%d total) → %d unique candidates",
        len(candidate_lists),
        sum(len(cl) for cl in candidate_lists),
        len(result),
    )
    return result
