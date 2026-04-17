"""Query-adaptive retrieval stopping criteria.

Implements three stopping strategies based on HingeMem (WWW '26) and
ACQO (arXiv 2601.21208) research:

1. **Recall-oriented**: Knee detection — stop when marginal relevance
   gain per additional result drops below a threshold.
2. **Precision-oriented**: Top-k saturation — stop when ≥80% of the
   top-k results exceed a quality threshold.
3. **Judgment-oriented**: Top-1 early stop — stop when the best result
   is stable across consecutive batches.

Additionally provides:
- Score plateau detection (ACQO-inspired)
- Composite adaptive stopping that selects strategy by query type
- Budget-aware hard limits

Reference papers:
- HingeMem (2604.06845) — boundary-guided segmentation + query-adaptive
- ACQO (2601.21208) — adaptive query count via RL, rank-score fusion
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Query intent classification for stopping strategy selection."""

    RECALL = "recall"
    PRECISION = "precision"
    JUDGMENT = "judgment"
    AUTO = "auto"


class StopReason(str, Enum):
    """Reason the stopping criterion triggered."""

    KNEE_DETECTED = "knee_detected"
    SATURATION_REACHED = "saturation_reached"
    TOP1_STABLE = "top1_stable"
    SCORE_PLATEAU = "score_plateau"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MIN_RESULTS_NOT_MET = "min_results_not_met"
    NOT_STOPPED = "not_stopped"


@dataclass
class StoppingDecision:
    """Result of a stopping criterion evaluation."""

    should_stop: bool
    reason: StopReason
    details: str = ""
    batches_processed: int = 0
    total_results: int = 0
    current_score: float = 0.0


@dataclass
class BatchScores:
    """Scores from one retrieval batch for tracking convergence."""

    batch_index: int
    scores: list[float]
    best_score: float = 0.0
    mean_score: float = 0.0
    relevant_count: int = 0

    def __post_init__(self) -> None:
        if self.scores:
            self.best_score = max(self.scores)
            self.mean_score = sum(self.scores) / len(self.scores)


@dataclass
class StoppingState:
    """Accumulated state for adaptive stopping decisions."""

    query_type: QueryType = QueryType.AUTO
    batches: list[BatchScores] = field(default_factory=list)
    max_budget: int = 500
    min_results: int = 5
    relevance_threshold: float = 0.5

    @property
    def total_results(self) -> int:
        return sum(len(b.scores) for b in self.batches)

    @property
    def all_scores(self) -> list[float]:
        result: list[float] = []
        for b in self.batches:
            result.extend(b.scores)
        return result


def detect_knee(
    cumulative_scores: list[float],
    threshold: float = 0.05,
) -> tuple[bool, int]:
    """Detect knee point in cumulative relevance scores.

    The knee is where marginal gain drops below *threshold* of the
    initial gain rate. Uses the L-method approximation: fit two
    line segments and find the elbow.

    Args:
        cumulative_scores: Running sum of sorted (desc) relevance scores.
        threshold: Minimum marginal gain ratio vs initial slope.

    Returns:
        Tuple of (knee_found, knee_index).
    """
    if len(cumulative_scores) < 3:
        return False, 0

    # Compute marginal gains
    gains = [cumulative_scores[0]]
    for i in range(1, len(cumulative_scores)):
        gains.append(cumulative_scores[i] - cumulative_scores[i - 1])

    if gains[0] <= 0:
        return True, 0

    # Find where marginal gain drops below threshold * initial gain
    cutoff = threshold * gains[0]
    for i, gain in enumerate(gains):
        if gain < cutoff and i >= 2:
            return True, i

    return False, 0


def check_recall_stopping(
    state: StoppingState,
    knee_threshold: float = 0.05,
) -> StoppingDecision:
    """Recall-oriented stopping: knee detection on cumulative relevance.

    Stop when the marginal relevance gain per additional result drops
    below the threshold fraction of the initial gain rate.

    Args:
        state: Current retrieval state with batch scores.
        knee_threshold: Minimum gain ratio (default 5% of initial).

    Returns:
        StoppingDecision with knee detection result.
    """
    all_scores = sorted(state.all_scores, reverse=True)
    if len(all_scores) < state.min_results:
        return StoppingDecision(
            should_stop=False,
            reason=StopReason.MIN_RESULTS_NOT_MET,
            details=f"Only {len(all_scores)} results, need {state.min_results}",
            batches_processed=len(state.batches),
            total_results=len(all_scores),
        )

    # Build cumulative relevance
    cumulative = []
    running = 0.0
    for s in all_scores:
        running += max(s, 0.0)
        cumulative.append(running)

    knee_found, knee_idx = detect_knee(cumulative, knee_threshold)

    if knee_found and knee_idx >= state.min_results:
        return StoppingDecision(
            should_stop=True,
            reason=StopReason.KNEE_DETECTED,
            details=(
                f"Knee at index {knee_idx} of {len(all_scores)} results; "
                f"marginal gain below {knee_threshold:.1%} of initial"
            ),
            batches_processed=len(state.batches),
            total_results=len(all_scores),
            current_score=cumulative[knee_idx] if knee_idx < len(cumulative) else 0.0,
        )

    return StoppingDecision(
        should_stop=False,
        reason=StopReason.NOT_STOPPED,
        details=f"No knee detected in {len(all_scores)} results",
        batches_processed=len(state.batches),
        total_results=len(all_scores),
    )


def check_precision_stopping(
    state: StoppingState,
    saturation_ratio: float = 0.80,
    top_k: int = 20,
) -> StoppingDecision:
    """Precision-oriented stopping: top-k saturation check.

    Stop when at least *saturation_ratio* of the top-k results exceed
    the relevance threshold. This ensures high-quality results dominate
    before stopping.

    Args:
        state: Current retrieval state.
        saturation_ratio: Fraction of top-k that must be relevant (default 80%).
        top_k: Number of top results to evaluate.

    Returns:
        StoppingDecision with saturation status.
    """
    all_scores = sorted(state.all_scores, reverse=True)
    if len(all_scores) < state.min_results:
        return StoppingDecision(
            should_stop=False,
            reason=StopReason.MIN_RESULTS_NOT_MET,
            details=f"Only {len(all_scores)} results, need {state.min_results}",
            batches_processed=len(state.batches),
            total_results=len(all_scores),
        )

    eval_k = min(top_k, len(all_scores))
    top_scores = all_scores[:eval_k]
    relevant = sum(1 for s in top_scores if s >= state.relevance_threshold)
    ratio = relevant / eval_k if eval_k > 0 else 0.0

    if ratio >= saturation_ratio:
        return StoppingDecision(
            should_stop=True,
            reason=StopReason.SATURATION_REACHED,
            details=(
                f"Top-{eval_k} saturation: {relevant}/{eval_k} "
                f"({ratio:.1%}) >= {saturation_ratio:.1%} threshold"
            ),
            batches_processed=len(state.batches),
            total_results=len(all_scores),
            current_score=ratio,
        )

    return StoppingDecision(
        should_stop=False,
        reason=StopReason.NOT_STOPPED,
        details=(
            f"Top-{eval_k} saturation: {relevant}/{eval_k} "
            f"({ratio:.1%}) < {saturation_ratio:.1%}"
        ),
        batches_processed=len(state.batches),
        total_results=len(all_scores),
        current_score=ratio,
    )


def check_judgment_stopping(
    state: StoppingState,
    stability_window: int = 3,
    tolerance: float = 0.01,
) -> StoppingDecision:
    """Judgment-oriented stopping: top-1 stability check.

    Stop when the best score has been stable (within *tolerance*) for
    *stability_window* consecutive batches. If the best result isn't
    changing, further retrieval is unlikely to help.

    Args:
        state: Current retrieval state with batch history.
        stability_window: Number of consecutive stable batches needed.
        tolerance: Maximum allowed change in top-1 score.

    Returns:
        StoppingDecision with stability assessment.
    """
    if len(state.batches) < stability_window:
        return StoppingDecision(
            should_stop=False,
            reason=StopReason.NOT_STOPPED,
            details=(
                f"Only {len(state.batches)} batches, "
                f"need {stability_window} for stability check"
            ),
            batches_processed=len(state.batches),
            total_results=state.total_results,
        )

    # Check if top-1 score is stable over the window
    recent = state.batches[-stability_window:]

    # Track the running best across all batches up to each point
    running_best = 0.0
    best_per_batch: list[float] = []
    for b in state.batches:
        running_best = max(running_best, b.best_score)
        best_per_batch.append(running_best)

    # Check stability of the running best over the window
    window_bests = best_per_batch[-stability_window:]
    max_change = max(window_bests) - min(window_bests)

    if max_change <= tolerance and state.total_results >= state.min_results:
        return StoppingDecision(
            should_stop=True,
            reason=StopReason.TOP1_STABLE,
            details=(
                f"Top-1 score stable at {window_bests[-1]:.4f} "
                f"(change {max_change:.4f} <= {tolerance}) "
                f"over {stability_window} batches"
            ),
            batches_processed=len(state.batches),
            total_results=state.total_results,
            current_score=window_bests[-1],
        )

    return StoppingDecision(
        should_stop=False,
        reason=StopReason.NOT_STOPPED,
        details=(
            f"Top-1 change {max_change:.4f} > {tolerance} "
            f"over last {len(recent)} batches"
        ),
        batches_processed=len(state.batches),
        total_results=state.total_results,
        current_score=window_bests[-1] if window_bests else 0.0,
    )


def check_score_plateau(
    state: StoppingState,
    window: int = 3,
    improvement_threshold: float = 0.02,
) -> StoppingDecision:
    """ACQO-inspired score plateau detection.

    Stop when the mean score improvement across consecutive batches
    is below the threshold for *window* batches.

    Args:
        state: Current retrieval state.
        window: Number of batches to check for plateau.
        improvement_threshold: Minimum mean score improvement.

    Returns:
        StoppingDecision with plateau assessment.
    """
    if len(state.batches) < window + 1:
        return StoppingDecision(
            should_stop=False,
            reason=StopReason.NOT_STOPPED,
            details=f"Need {window + 1} batches, have {len(state.batches)}",
            batches_processed=len(state.batches),
            total_results=state.total_results,
        )

    recent = state.batches[-(window + 1) :]
    improvements = []
    for i in range(1, len(recent)):
        diff = recent[i].mean_score - recent[i - 1].mean_score
        improvements.append(diff)

    avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0

    if (
        avg_improvement < improvement_threshold
        and state.total_results >= state.min_results
    ):
        return StoppingDecision(
            should_stop=True,
            reason=StopReason.SCORE_PLATEAU,
            details=(
                f"Mean score plateau: avg improvement {avg_improvement:.4f} "
                f"< {improvement_threshold} over {window} batches"
            ),
            batches_processed=len(state.batches),
            total_results=state.total_results,
            current_score=recent[-1].mean_score,
        )

    return StoppingDecision(
        should_stop=False,
        reason=StopReason.NOT_STOPPED,
        details=f"Score improving at {avg_improvement:.4f}/batch",
        batches_processed=len(state.batches),
        total_results=state.total_results,
        current_score=recent[-1].mean_score,
    )


def classify_query_type(
    query: str,
    result_count_hint: int | None = None,
) -> QueryType:
    """Heuristic classification of query intent.

    Maps a query string to one of three retrieval intent types to select
    the appropriate stopping strategy:
    - **recall**: broad/survey queries → knee detection
    - **precision**: specific/narrow queries → top-k saturation
    - **judgment**: comparative/evaluative queries → top-1 stability

    Args:
        query: The search query string.
        result_count_hint: Optional expected result count hint.

    Returns:
        Classified QueryType.
    """
    q_lower = query.lower()

    # Judgment indicators: comparison, evaluation, best/worst
    judgment_markers = [
        "compare",
        "comparison",
        "versus",
        "vs",
        "better",
        "best",
        "evaluate",
        "which",
        "rank",
        "benchmark",
    ]
    if any(m in q_lower for m in judgment_markers):
        return QueryType.JUDGMENT

    # Precision indicators: specific, narrow focus
    precision_markers = [
        "specific",
        "exact",
        "particular",
        "how to",
        "implement",
        "algorithm for",
        "method for",
        "technique for",
    ]
    if any(m in q_lower for m in precision_markers):
        return QueryType.PRECISION

    # Recall indicators: broad, survey-like
    recall_markers = [
        "survey",
        "overview",
        "review",
        "comprehensive",
        "all",
        "taxonomy",
        "classification",
        "landscape",
        "state of the art",
        "sota",
    ]
    if any(m in q_lower for m in recall_markers):
        return QueryType.RECALL

    # Default based on query length heuristic
    word_count = len(q_lower.split())
    if word_count <= 3:
        return QueryType.RECALL
    if word_count >= 8:
        return QueryType.PRECISION

    return QueryType.RECALL


def evaluate_stopping(
    state: StoppingState,
    query: str | None = None,
    knee_threshold: float = 0.05,
    saturation_ratio: float = 0.80,
    top_k: int = 20,
    stability_window: int = 3,
    stability_tolerance: float = 0.01,
    plateau_window: int = 3,
    plateau_threshold: float = 0.02,
) -> StoppingDecision:
    """Composite adaptive stopping evaluation.

    Selects the appropriate stopping strategy based on query type and
    evaluates it. Also checks budget limits and score plateau as
    universal backstops.

    Args:
        state: Accumulated retrieval state.
        query: Optional query string for type classification.
        knee_threshold: Threshold for knee detection.
        saturation_ratio: Ratio for precision saturation.
        top_k: Top-k for precision evaluation.
        stability_window: Batches for judgment stability.
        stability_tolerance: Tolerance for top-1 changes.
        plateau_window: Batches for plateau detection.
        plateau_threshold: Improvement threshold for plateau.

    Returns:
        StoppingDecision from the selected or first-triggered strategy.
    """
    # Hard budget limit
    if state.total_results >= state.max_budget:
        return StoppingDecision(
            should_stop=True,
            reason=StopReason.BUDGET_EXHAUSTED,
            details=(
                f"Budget limit reached: " f"{state.total_results} >= {state.max_budget}"
            ),
            batches_processed=len(state.batches),
            total_results=state.total_results,
        )

    # Determine query type
    qtype = state.query_type
    if qtype == QueryType.AUTO and query:
        qtype = classify_query_type(query)
        logger.debug("Auto-classified query type: %s for '%s'", qtype.value, query)

    # Select primary strategy
    if qtype == QueryType.RECALL:
        primary = check_recall_stopping(state, knee_threshold)
    elif qtype == QueryType.PRECISION:
        primary = check_precision_stopping(state, saturation_ratio, top_k)
    elif qtype == QueryType.JUDGMENT:
        primary = check_judgment_stopping(state, stability_window, stability_tolerance)
    else:
        primary = check_recall_stopping(state, knee_threshold)

    if primary.should_stop:
        logger.info(
            "Adaptive stopping (%s): %s — %s",
            qtype.value,
            primary.reason.value,
            primary.details,
        )
        return primary

    # Universal backstop: score plateau
    plateau = check_score_plateau(state, plateau_window, plateau_threshold)
    if plateau.should_stop:
        logger.info("Adaptive stopping (plateau backstop): %s", plateau.details)
        return plateau

    return StoppingDecision(
        should_stop=False,
        reason=StopReason.NOT_STOPPED,
        details=f"Continuing retrieval ({qtype.value} strategy): {primary.details}",
        batches_processed=len(state.batches),
        total_results=state.total_results,
    )


def marginal_gain_ratio(cumulative: list[float], index: int) -> float:
    """Compute marginal gain ratio at a given index.

    Useful for diagnostics and visualization of the stopping curve.

    Args:
        cumulative: Cumulative relevance scores (sorted desc, running sum).
        index: Index to compute ratio at.

    Returns:
        Ratio of marginal gain at index vs initial gain, or 0.0.
    """
    if not cumulative or index < 1 or index >= len(cumulative):
        return 0.0

    initial_gain = cumulative[0]
    if initial_gain <= 0:
        return 0.0

    current_gain = cumulative[index] - cumulative[index - 1]
    return current_gain / initial_gain
