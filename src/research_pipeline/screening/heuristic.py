"""Heuristic relevance scoring using BM25, category matching, and recency."""

import logging
import math
from datetime import UTC, datetime

from rank_bm25 import BM25Okapi

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.screening import CheapScoreBreakdown

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def _recency_bonus(published: datetime, half_life_days: float = 90.0) -> float:
    """Compute an exponential decay recency bonus.

    Args:
        published: Paper publication date.
        half_life_days: Number of days for the bonus to halve.

    Returns:
        Recency bonus between 0 and 1.
    """
    now = datetime.now(UTC)
    age_days = max((now - published).total_seconds() / 86400, 0)
    return math.exp(-0.693 * age_days / half_life_days)


def _compute_bm25_scores(
    corpus_texts: list[str],
    query_terms: list[str],
) -> list[float]:
    """Compute BM25 scores for each document against a set of query terms.

    Args:
        corpus_texts: List of document texts.
        query_terms: Query terms to score against.

    Returns:
        List of BM25 scores (one per document).
    """
    if not corpus_texts or not query_terms:
        return [0.0] * len(corpus_texts)

    tokenized_corpus = [_tokenize(text) for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _tokenize(" ".join(query_terms))
    scores = bm25.get_scores(tokenized_query)
    return [float(s) for s in scores]


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def score_candidates(
    candidates: list[CandidateRecord],
    must_terms: list[str],
    nice_terms: list[str],
    negative_terms: list[str],
    target_categories: list[str],
    weights: dict[str, float] | None = None,
    semantic_scores: list[float] | None = None,
) -> list[CheapScoreBreakdown]:
    """Score all candidates using heuristic methods.

    Args:
        candidates: List of candidate papers.
        must_terms: Required terms (high weight).
        nice_terms: Optional terms (lower weight).
        negative_terms: Terms that reduce score.
        target_categories: Preferred arXiv categories.
        weights: Score component weights. Defaults provided.
        semantic_scores: Pre-computed semantic similarity scores (0-1),
            one per candidate. When provided, the "semantic_similarity"
            weight is used. Pass None to disable.

    Returns:
        List of score breakdowns, one per candidate.
    """
    use_semantic = semantic_scores is not None and len(semantic_scores) == len(
        candidates
    )

    if use_semantic:
        w = weights or {
            "bm25_title": 0.20,
            "bm25_abstract": 0.25,
            "semantic_similarity": 0.25,
            "cat_match": 0.12,
            "negative_penalty": 0.08,
            "recency_bonus": 0.10,
        }
    else:
        w = weights or {
            "bm25_title": 0.30,
            "bm25_abstract": 0.35,
            "cat_match": 0.15,
            "negative_penalty": 0.10,
            "recency_bonus": 0.10,
        }

    query_terms = must_terms + nice_terms
    titles = [c.title for c in candidates]
    abstracts = [c.abstract for c in candidates]

    raw_title_scores = _compute_bm25_scores(titles, query_terms)
    raw_abstract_scores = _compute_bm25_scores(abstracts, query_terms)

    title_scores = _normalize_scores(raw_title_scores)
    abstract_scores = _normalize_scores(raw_abstract_scores)

    target_set = set(target_categories)
    neg_lower = {t.lower() for t in negative_terms}

    breakdowns: list[CheapScoreBreakdown] = []
    for i, candidate in enumerate(candidates):
        bm25_title = title_scores[i]
        bm25_abstract = abstract_scores[i]

        # Category match: 1.0 if primary category matches, 0.5 if any category matches
        cat_match = 0.0
        if target_set:
            if candidate.primary_category in target_set:
                cat_match = 1.0
            elif target_set.intersection(candidate.categories):
                cat_match = 0.5

        # Negative penalty: count negative term hits in title + abstract
        neg_penalty = 0.0
        if neg_lower:
            combined_lower = (candidate.title + " " + candidate.abstract).lower()
            hits = sum(1 for t in neg_lower if t in combined_lower)
            neg_penalty = min(hits * 0.3, 1.0)

        recency = _recency_bonus(candidate.published)

        sem_score = semantic_scores[i] if use_semantic else None

        cheap_score = (
            w["bm25_title"] * bm25_title
            + w["bm25_abstract"] * bm25_abstract
            + w["cat_match"] * cat_match
            - w["negative_penalty"] * neg_penalty
            + w["recency_bonus"] * recency
        )
        if use_semantic and sem_score is not None:
            cheap_score += w.get("semantic_similarity", 0.0) * sem_score
        cheap_score = max(0.0, min(1.0, cheap_score))

        breakdowns.append(
            CheapScoreBreakdown(
                bm25_title=round(bm25_title, 4),
                bm25_abstract=round(bm25_abstract, 4),
                cat_match=round(cat_match, 4),
                negative_penalty=round(neg_penalty, 4),
                recency_bonus=round(recency, 4),
                semantic_score=round(sem_score, 4) if sem_score is not None else None,
                cheap_score=round(cheap_score, 4),
            )
        )

    logger.info(
        "Scored %d candidates: mean=%.3f, max=%.3f, min=%.3f",
        len(breakdowns),
        sum(b.cheap_score for b in breakdowns) / max(len(breakdowns), 1),
        max((b.cheap_score for b in breakdowns), default=0),
        min((b.cheap_score for b in breakdowns), default=0),
    )
    return breakdowns


def select_topk(
    candidates: list[CandidateRecord],
    scores: list[CheapScoreBreakdown],
    top_k: int = 50,
    min_score: float = 0.0,
) -> list[tuple[CandidateRecord, CheapScoreBreakdown]]:
    """Select top-K candidates by heuristic score for LLM second pass.

    Args:
        candidates: All candidates.
        scores: Corresponding score breakdowns.
        top_k: Maximum number to select.
        min_score: Minimum score threshold.

    Returns:
        List of (candidate, score) tuples, sorted by score descending.
    """
    paired = list(zip(candidates, scores, strict=True))
    filtered = [(c, s) for c, s in paired if s.cheap_score >= min_score]
    filtered.sort(key=lambda x: x[1].cheap_score, reverse=True)
    selected = filtered[:top_k]
    logger.info(
        "Selected top-%d from %d candidates (min_score=%.2f)",
        len(selected),
        len(filtered),
        min_score,
    )
    return selected
