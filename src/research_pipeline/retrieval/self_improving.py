"""Self-improving retrieval: iterative query refinement with convergence detection.

Implements an iterative retrieval loop inspired by SiRe (Self-Improving
Retrieval) that:

1. Runs retrieval with the current query terms
2. Analyzes result quality (coverage, relevance distribution)
3. Refines the query via term feedback (add high-signal, drop low-signal)
4. Detects convergence (score plateau, term stability, max iterations)

Works entirely with heuristics — no LLM required. Degrades to single-pass
retrieval when query feedback produces no meaningful refinements.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.query_plan import QueryPlan

logger = logging.getLogger(__name__)

# Common stopwords for term extraction
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "we",
        "our",
        "they",
        "their",
        "which",
        "who",
        "whom",
        "what",
        "where",
        "when",
        "how",
        "not",
        "no",
        "nor",
        "as",
        "if",
        "then",
        "than",
        "so",
        "very",
        "also",
        "just",
        "more",
        "most",
        "such",
        "each",
        "other",
        "some",
        "all",
        "both",
        "any",
        "many",
        "much",
        "few",
        "only",
        "own",
        "paper",
        "propose",
        "proposed",
        "method",
        "approach",
        "model",
        "show",
        "shown",
        "results",
        "result",
        "using",
        "used",
        "use",
        "new",
        "based",
        "study",
        "data",
        "two",
        "first",
        "one",
    }
)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")


@dataclass
class RetrievalIteration:
    """Record of a single retrieval iteration."""

    iteration: int
    query_terms: list[str]
    result_count: int
    avg_score: float
    top_score: float
    coverage: float
    refined_terms: list[str]


@dataclass
class ConvergenceInfo:
    """Why the retrieval loop stopped."""

    reason: str  # "max_iterations", "score_plateau", "term_stable", "no_results"
    iterations_run: int
    final_avg_score: float
    score_improvement: float


@dataclass
class SelfImprovingResult:
    """Complete result from the self-improving retrieval loop."""

    iterations: list[RetrievalIteration] = field(default_factory=list)
    convergence: ConvergenceInfo | None = None
    final_query_terms: list[str] = field(default_factory=list)
    final_papers: list[CandidateRecord] = field(default_factory=list)

    @property
    def total_iterations(self) -> int:
        """Number of iterations run."""
        return len(self.iterations)

    @property
    def converged(self) -> bool:
        """Whether the loop converged naturally (not just max iterations)."""
        if self.convergence is None:
            return False
        return self.convergence.reason in ("score_plateau", "term_stable")


def _compute_coverage(query_terms: list[str], papers: list[CandidateRecord]) -> float:
    """Fraction of query terms found in at least one paper title/abstract."""
    if not query_terms or not papers:
        return 0.0
    combined_text = " ".join(f"{p.title} {p.abstract}".lower() for p in papers)
    found = sum(1 for t in query_terms if t.lower() in combined_text)
    return found / len(query_terms)


def _compute_avg_score(
    papers: list[CandidateRecord], scores: dict[str, float]
) -> float:
    """Average score across papers (0.0 if no scores)."""
    if not papers:
        return 0.0
    vals = [scores.get(p.arxiv_id, 0.0) for p in papers]
    return sum(vals) / len(vals) if vals else 0.0


def _compute_top_score(
    papers: list[CandidateRecord], scores: dict[str, float]
) -> float:
    """Top score across papers (0.0 if no scores)."""
    if not papers:
        return 0.0
    vals = [scores.get(p.arxiv_id, 0.0) for p in papers]
    return max(vals) if vals else 0.0


def _extract_top_terms(
    papers: list[CandidateRecord], existing_terms: set[str], top_n: int = 5
) -> list[str]:
    """Extract frequent terms from paper titles/abstracts not in existing terms."""
    counter: Counter[str] = Counter()
    for p in papers:
        text = f"{p.title} {p.abstract}".lower()
        tokens = _TOKEN_RE.findall(text)
        for t in tokens:
            if t not in _STOPWORDS and t not in existing_terms:
                counter[t] += 1
    return [term for term, _ in counter.most_common(top_n)]


def _refine_terms(
    current_terms: list[str],
    papers: list[CandidateRecord],
    coverage: float,
) -> list[str]:
    """Refine query terms based on retrieval results.

    Strategy:
    - Drop terms with zero coverage in results (noise removal)
    - Add high-frequency terms from top results (signal boost)
    - Keep the refined list bounded
    """
    if not papers:
        return current_terms

    existing_lower = {t.lower() for t in current_terms}
    combined_text = " ".join(f"{p.title} {p.abstract}".lower() for p in papers)

    # Keep terms that appear in at least one paper
    retained = [t for t in current_terms if t.lower() in combined_text]

    # Avoid query collapse
    if len(retained) < max(1, len(current_terms) // 2):
        retained = list(current_terms)

    # Add high-signal terms from results
    new_terms = _extract_top_terms(papers, existing_lower, top_n=3)
    refined = retained + [t for t in new_terms if t not in retained]

    # Cap total terms
    max_terms = max(len(current_terms) + 3, 15)
    return refined[:max_terms]


def _check_convergence(
    iterations: list[RetrievalIteration],
    max_iterations: int,
    score_threshold: float,
    term_overlap_threshold: float,
) -> ConvergenceInfo | None:
    """Check if the retrieval loop should stop."""
    if not iterations:
        return None

    current = iterations[-1]
    first = iterations[0]
    total_improvement = current.avg_score - first.avg_score

    if len(iterations) >= max_iterations:
        return ConvergenceInfo(
            reason="max_iterations",
            iterations_run=len(iterations),
            final_avg_score=current.avg_score,
            score_improvement=total_improvement,
        )

    if len(iterations) < 2:
        return None

    prev = iterations[-2]

    # Score plateau
    score_delta = abs(current.avg_score - prev.avg_score)
    if score_delta < score_threshold:
        return ConvergenceInfo(
            reason="score_plateau",
            iterations_run=len(iterations),
            final_avg_score=current.avg_score,
            score_improvement=total_improvement,
        )

    # Term stability
    current_set = {t.lower() for t in current.query_terms}
    prev_set = {t.lower() for t in prev.query_terms}
    if current_set and prev_set:
        overlap = len(current_set & prev_set) / max(len(current_set), len(prev_set))
        if overlap >= term_overlap_threshold:
            return ConvergenceInfo(
                reason="term_stable",
                iterations_run=len(iterations),
                final_avg_score=current.avg_score,
                score_improvement=total_improvement,
            )

    # No results
    if current.result_count == 0:
        return ConvergenceInfo(
            reason="no_results",
            iterations_run=len(iterations),
            final_avg_score=current.avg_score,
            score_improvement=total_improvement,
        )

    return None


def _score_papers(
    papers: list[CandidateRecord], terms: list[str]
) -> tuple[list[CandidateRecord], dict[str, float]]:
    """Score papers based on term overlap and return sorted list with scores.

    Args:
        papers: Candidate papers.
        terms: Query terms to match.

    Returns:
        Tuple of (sorted papers, score dict keyed by arxiv_id).
    """
    if not terms:
        return papers, {p.arxiv_id: 0.0 for p in papers}

    scores: dict[str, float] = {}
    for p in papers:
        text = f"{p.title} {p.abstract}".lower()
        matches = sum(1 for t in terms if t.lower() in text)
        scores[p.arxiv_id] = matches / len(terms)

    sorted_papers = sorted(
        papers,
        key=lambda x: scores.get(x.arxiv_id, 0.0),
        reverse=True,
    )
    return sorted_papers, scores


def run_self_improving_retrieval(
    query_plan: QueryPlan,
    papers: list[CandidateRecord],
    max_iterations: int = 3,
    score_threshold: float = 0.01,
    term_overlap_threshold: float = 0.9,
) -> SelfImprovingResult:
    """Run iterative self-improving retrieval loop.

    Starting from the query plan's terms, iteratively:
    1. Score papers against current terms
    2. Analyze coverage and quality
    3. Refine terms based on feedback
    4. Check convergence

    Args:
        query_plan: The query plan with initial terms.
        papers: All candidate papers to score against.
        max_iterations: Maximum refinement iterations.
        score_threshold: Minimum score improvement per iteration.
        term_overlap_threshold: Term overlap ratio for stability detection.

    Returns:
        SelfImprovingResult with iteration history and final state.
    """
    result = SelfImprovingResult()

    if not papers:
        result.convergence = ConvergenceInfo(
            reason="no_results",
            iterations_run=0,
            final_avg_score=0.0,
            score_improvement=0.0,
        )
        result.final_query_terms = list(query_plan.must_terms)
        return result

    current_terms = list(query_plan.must_terms) + list(query_plan.nice_terms)
    if not current_terms:
        current_terms = query_plan.topic_raw.lower().split()

    for i in range(max_iterations):
        scored_papers, scores = _score_papers(papers, current_terms)
        avg_score = _compute_avg_score(scored_papers, scores)
        top_score = _compute_top_score(scored_papers, scores)
        coverage = _compute_coverage(current_terms, scored_papers)

        refined = _refine_terms(current_terms, scored_papers, coverage)

        iteration = RetrievalIteration(
            iteration=i,
            query_terms=list(current_terms),
            result_count=len(scored_papers),
            avg_score=avg_score,
            top_score=top_score,
            coverage=coverage,
            refined_terms=refined,
        )
        result.iterations.append(iteration)

        logger.info(
            "SIR iteration %d: terms=%d, results=%d, avg=%.3f, coverage=%.2f",
            i,
            len(current_terms),
            len(scored_papers),
            avg_score,
            coverage,
        )

        convergence = _check_convergence(
            result.iterations,
            max_iterations,
            score_threshold,
            term_overlap_threshold,
        )
        if convergence is not None:
            result.convergence = convergence
            result.final_query_terms = refined
            result.final_papers = scored_papers
            logger.info(
                "SIR converged: %s after %d iterations",
                convergence.reason,
                convergence.iterations_run,
            )
            return result

        current_terms = refined

    # Safety fallback (max_iterations should catch above)
    result.final_query_terms = current_terms
    final_papers, _ = _score_papers(papers, current_terms)
    result.final_papers = final_papers
    result.convergence = ConvergenceInfo(
        reason="max_iterations",
        iterations_run=len(result.iterations),
        final_avg_score=(result.iterations[-1].avg_score if result.iterations else 0.0),
        score_improvement=0.0,
    )
    return result
