"""Heuristic relevance scoring using BM25, category matching, and recency."""

import logging
import math
from datetime import UTC, datetime

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

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
            "bm25_must_title": 0.12,
            "bm25_nice_title": 0.08,
            "bm25_must_abstract": 0.15,
            "bm25_nice_abstract": 0.10,
            "semantic_similarity": 0.25,
            "cat_match": 0.12,
            "negative_penalty": 0.08,
            "recency_bonus": 0.10,
        }
    else:
        w = weights or {
            "bm25_must_title": 0.20,
            "bm25_nice_title": 0.10,
            "bm25_must_abstract": 0.25,
            "bm25_nice_abstract": 0.10,
            "cat_match": 0.15,
            "negative_penalty": 0.10,
            "recency_bonus": 0.10,
        }

    titles = [c.title for c in candidates]
    abstracts = [c.abstract for c in candidates]

    # Score must_terms and nice_terms separately so must_terms dominate
    raw_must_title = _compute_bm25_scores(titles, must_terms)
    raw_nice_title = (
        _compute_bm25_scores(titles, nice_terms)
        if nice_terms
        else [0.0] * len(candidates)
    )
    raw_must_abstract = _compute_bm25_scores(abstracts, must_terms)
    raw_nice_abstract = (
        _compute_bm25_scores(abstracts, nice_terms)
        if nice_terms
        else [0.0] * len(candidates)
    )

    must_title_scores = _normalize_scores(raw_must_title)
    nice_title_scores = _normalize_scores(raw_nice_title)
    must_abstract_scores = _normalize_scores(raw_must_abstract)
    nice_abstract_scores = _normalize_scores(raw_nice_abstract)

    target_set = set(target_categories)
    neg_lower = {t.lower() for t in negative_terms}

    breakdowns: list[CheapScoreBreakdown] = []
    for i, candidate in enumerate(candidates):
        bm25_title = (
            w.get("bm25_must_title", 0.0) * must_title_scores[i]
            + w.get("bm25_nice_title", 0.0) * nice_title_scores[i]
        )
        bm25_abstract = (
            w.get("bm25_must_abstract", 0.0) * must_abstract_scores[i]
            + w.get("bm25_nice_abstract", 0.0) * nice_abstract_scores[i]
        )

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

        sem_score = (
            semantic_scores[i] if semantic_scores is not None and use_semantic else None
        )

        cheap_score = (
            bm25_title
            + bm25_abstract
            + w["cat_match"] * cat_match
            - w["negative_penalty"] * neg_penalty
            + w["recency_bonus"] * recency
        )
        if use_semantic and sem_score is not None:
            cheap_score += w.get("semantic_similarity", 0.0) * sem_score
        cheap_score = max(0.0, min(1.0, cheap_score))

        # Store combined BM25 scores in the breakdown for compatibility
        combined_bm25_title = must_title_scores[i] * 0.67 + nice_title_scores[i] * 0.33
        combined_bm25_abstract = (
            must_abstract_scores[i] * 0.67 + nice_abstract_scores[i] * 0.33
        )

        breakdowns.append(
            CheapScoreBreakdown(
                bm25_title=round(combined_bm25_title, 4),
                bm25_abstract=round(combined_bm25_abstract, 4),
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
    diversity: bool = False,
    diversity_lambda: float = 0.3,
) -> list[tuple[CandidateRecord, CheapScoreBreakdown]]:
    """Select top-K candidates by heuristic score with optional diversity.

    When ``diversity=True``, uses a greedy MMR-style (Maximal Marginal Relevance)
    reranking that balances relevance score with coverage of different categories,
    time periods, and sources to avoid an echo-chamber shortlist.

    Args:
        candidates: All candidates.
        scores: Corresponding score breakdowns.
        top_k: Maximum number to select.
        min_score: Minimum score threshold.
        diversity: Enable diversity-aware reranking.
        diversity_lambda: Balance between relevance (0.0) and diversity (1.0).
            Default 0.3 gives moderate diversity.

    Returns:
        List of (candidate, score) tuples, sorted by score descending.
    """
    paired = list(zip(candidates, scores, strict=True))
    filtered = [(c, s) for c, s in paired if s.cheap_score >= min_score]

    if not diversity or diversity_lambda <= 0.0:
        filtered.sort(key=lambda x: x[1].cheap_score, reverse=True)
        selected = filtered[:top_k]
        logger.info(
            "Selected top-%d from %d candidates (min_score=%.2f)",
            len(selected),
            len(filtered),
            min_score,
        )
        return selected

    # Diversity-aware MMR-style selection
    selected = _diverse_select(filtered, top_k, diversity_lambda)
    logger.info(
        "Selected top-%d from %d candidates (diversity=%.2f, min_score=%.2f)",
        len(selected),
        len(filtered),
        diversity_lambda,
        min_score,
    )
    return selected


def _tokenize_document(candidate: CandidateRecord) -> set[str]:
    """Tokenize a candidate's title + abstract into a term set.

    Args:
        candidate: Paper to tokenize.

    Returns:
        Set of lowercase tokens.
    """
    text = f"{candidate.title} {candidate.abstract}"
    return set(_tokenize(text))


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets.

    Args:
        set_a: First token set.
        set_b: Second token set.

    Returns:
        Jaccard similarity in [0, 1].
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _max_content_similarity(
    candidate_tokens: set[str],
    selected_token_sets: list[set[str]],
) -> float:
    """Compute maximum Jaccard similarity to any already-selected document.

    This is the content-based redundancy measure for true MMR: higher
    values mean the candidate is more similar to an already-selected paper.

    Args:
        candidate_tokens: Token set of the candidate.
        selected_token_sets: Token sets of already-selected papers.

    Returns:
        Maximum similarity in [0, 1]. Returns 0.0 when no papers selected yet.
    """
    if not selected_token_sets:
        return 0.0
    return max(_jaccard_similarity(candidate_tokens, s) for s in selected_token_sets)


def _diverse_select(
    pool: list[tuple[CandidateRecord, CheapScoreBreakdown]],
    top_k: int,
    lam: float,
) -> list[tuple[CandidateRecord, CheapScoreBreakdown]]:
    """Greedy diversity-aware selection using true MMR.

    At each step, selects the candidate that maximizes:
        score = (1 - λ) * relevance - λ * max_sim_to_selected

    This implements Maximal Marginal Relevance (Carbonell & Goldstein, 1998)
    using Jaccard similarity on tokenized title+abstract as the document
    similarity measure, combined with metadata novelty (category, year,
    source) for a blended diversity signal.

    The MMR score for each candidate *c* given selected set *S* is:

        MMR(c) = (1 - λ) * rel(c) + λ * (α * metadata_novelty(c, S)
                                          + (1 - α) * (1 - max_sim(c, S)))

    where α = 0.4 balances metadata vs content diversity.

    Args:
        pool: Available (candidate, score) pairs.
        top_k: How many to select.
        lam: Diversity weight in [0, 1].

    Returns:
        Selected (candidate, score) pairs.
    """
    if not pool:
        return []

    # Normalize relevance scores to [0, 1]
    max_score = max(s.cheap_score for _, s in pool)
    min_score = min(s.cheap_score for _, s in pool)
    score_range = max_score - min_score if max_score > min_score else 1.0

    # Pre-tokenize all documents for content similarity
    pool_tokens = [_tokenize_document(c) for c, _ in pool]

    selected: list[tuple[CandidateRecord, CheapScoreBreakdown]] = []
    selected_token_sets: list[set[str]] = []
    remaining = list(range(len(pool)))

    # Track coverage of diversity dimensions
    covered_categories: dict[str, int] = {}
    covered_years: dict[int, int] = {}
    covered_sources: dict[str, int] = {}

    # Balance between metadata novelty and content novelty
    content_weight = 0.6  # content diversity matters more than metadata

    for _ in range(min(top_k, len(pool))):
        best_pool_idx = -1
        best_remaining_pos = -1
        best_mmr = -1.0

        for pos, pool_idx in enumerate(remaining):
            c, s = pool[pool_idx]
            relevance = (s.cheap_score - min_score) / score_range

            # Metadata novelty (category, year, source)
            metadata_novelty = _compute_novelty(
                c, covered_categories, covered_years, covered_sources
            )

            # Content novelty: 1 - max similarity to selected
            content_sim = _max_content_similarity(
                pool_tokens[pool_idx], selected_token_sets
            )
            content_novelty = 1.0 - content_sim

            # Blended diversity signal
            diversity = (
                1.0 - content_weight
            ) * metadata_novelty + content_weight * content_novelty

            mmr = (1.0 - lam) * relevance + lam * diversity
            if mmr > best_mmr:
                best_mmr = mmr
                best_pool_idx = pool_idx
                best_remaining_pos = pos

        if best_pool_idx < 0:
            break

        remaining.pop(best_remaining_pos)
        chosen_c, chosen_s = pool[best_pool_idx]
        selected.append((chosen_c, chosen_s))
        selected_token_sets.append(pool_tokens[best_pool_idx])

        # Update metadata coverage
        cat = chosen_c.primary_category
        covered_categories[cat] = covered_categories.get(cat, 0) + 1
        year = chosen_c.year or chosen_c.published.year
        covered_years[year] = covered_years.get(year, 0) + 1
        src = chosen_c.source
        covered_sources[src] = covered_sources.get(src, 0) + 1

    return selected


def _compute_novelty(
    candidate: CandidateRecord,
    covered_categories: dict[str, int],
    covered_years: dict[int, int],
    covered_sources: dict[str, int],
) -> float:
    """Compute metadata novelty score for a candidate given current coverage.

    Higher scores mean more novel (underrepresented dimensions).

    Args:
        candidate: Paper to evaluate.
        covered_categories: Category → count of already selected papers.
        covered_years: Year → count of already selected papers.
        covered_sources: Source → count of already selected papers.

    Returns:
        Novelty score in [0, 1].
    """
    if not covered_categories and not covered_years and not covered_sources:
        return 1.0  # First pick always maximally novel

    total_selected = sum(covered_categories.values()) or 1

    # Category novelty: fewer papers from this category = more novel
    cat = candidate.primary_category
    cat_count = covered_categories.get(cat, 0)
    cat_novelty = 1.0 - (cat_count / total_selected)

    # Year novelty: fewer papers from this year = more novel
    year = candidate.year or candidate.published.year
    year_count = covered_years.get(year, 0)
    year_novelty = 1.0 - (year_count / total_selected)

    # Source novelty: fewer papers from this source = more novel
    src = candidate.source
    src_count = covered_sources.get(src, 0)
    src_novelty = 1.0 - (src_count / total_selected)

    # Weighted combination: category matters most for methodology diversity
    return 0.50 * cat_novelty + 0.25 * year_novelty + 0.25 * src_novelty
