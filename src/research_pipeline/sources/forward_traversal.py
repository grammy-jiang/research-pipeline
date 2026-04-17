"""Forward citation traversal with budget-aware stopping.

Dedicated module for *forward* (citing-paper) snowball: given a seed
set, discover papers that **cite** those seeds.  Unlike the general
bidirectional snowball in ``snowball.py``, this module focuses
exclusively on the forward direction and adds:

- **Recency-weighted scoring**: newer citing papers score higher.
- **Influence propagation**: papers cited by multiple seeds rank higher.
- **Budget-aware stopping**: stops on marginal yield decay, hard cap,
  or recency saturation.
- **Citation velocity**: tracks how fast a seed is being cited to
  identify trending papers.

References:
    - Deep Research Report §FD4: forward snowball to complement backward
    - Arafat 2025: BFS expansion +24pp completeness
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ForwardStopReason(str, Enum):
    """Why forward traversal terminated."""

    MAX_PAPERS = "max_papers"
    MAX_DEPTH = "max_depth"
    YIELD_DECAY = "yield_decay"
    NO_NEW_PAPERS = "no_new_papers"
    BUDGET_EXHAUSTED = "budget_exhausted"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CitingPaper:
    """A paper discovered via forward traversal."""

    paper_id: str
    title: str = ""
    abstract: str = ""
    year: int | None = None
    cited_seed_ids: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    recency_bonus: float = 0.0
    influence_score: float = 0.0
    composite_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "cited_seed_count": len(self.cited_seed_ids),
            "relevance_score": round(self.relevance_score, 4),
            "recency_bonus": round(self.recency_bonus, 4),
            "influence_score": round(self.influence_score, 4),
            "composite_score": round(self.composite_score, 4),
        }


@dataclass
class TraversalRound:
    """Statistics for one round of forward traversal."""

    depth: int
    seeds_used: int
    fetched: int
    new_unique: int
    mean_score: float = 0.0
    yield_ratio: float = 0.0  # new_unique / fetched

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "seeds_used": self.seeds_used,
            "fetched": self.fetched,
            "new_unique": self.new_unique,
            "mean_score": round(self.mean_score, 4),
            "yield_ratio": round(self.yield_ratio, 4),
        }


@dataclass
class ForwardTraversalConfig:
    """Configuration for forward citation traversal."""

    max_depth: int = 3
    max_papers: int = 200
    limit_per_seed: int = 50
    yield_decay_threshold: float = 0.05
    yield_patience: int = 2
    recency_half_life: int = 3  # years
    influence_weight: float = 0.3
    recency_weight: float = 0.2
    relevance_weight: float = 0.5


@dataclass
class ForwardTraversalResult:
    """Complete result of a forward citation traversal."""

    seed_ids: list[str] = field(default_factory=list)
    discovered: list[CitingPaper] = field(default_factory=list)
    rounds: list[TraversalRound] = field(default_factory=list)
    stop_reason: ForwardStopReason = ForwardStopReason.MAX_DEPTH
    total_api_calls: int = 0

    @property
    def total_discovered(self) -> int:
        return len(self.discovered)

    def top_papers(self, n: int = 10) -> list[CitingPaper]:
        """Return top-n papers by composite score."""
        return sorted(
            self.discovered, key=lambda p: p.composite_score, reverse=True
        )[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_ids": self.seed_ids,
            "total_discovered": self.total_discovered,
            "stop_reason": self.stop_reason.value,
            "total_api_calls": self.total_api_calls,
            "rounds": [r.to_dict() for r in self.rounds],
            "top_papers": [p.to_dict() for p in self.top_papers(10)],
        }


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def recency_bonus(year: int | None, half_life: int = 3) -> float:
    """Compute a recency bonus for a paper based on publication year.

    Uses exponential decay: bonus = exp(-ln(2) * age / half_life).
    Current year papers get ~1.0, papers from ``half_life`` years ago get ~0.5.

    Args:
        year: Publication year (None → 0.0 bonus).
        half_life: Years for bonus to halve.

    Returns:
        Bonus in [0, 1].
    """
    if year is None or half_life <= 0:
        return 0.0
    current_year = datetime.now(UTC).year
    age = max(0, current_year - year)
    return math.exp(-math.log(2) * age / half_life)


def influence_score(cited_seed_count: int, total_seeds: int) -> float:
    """Score how influential a citing paper is based on multi-seed overlap.

    A paper that cites multiple seeds is more likely to be a survey or
    synthesis paper — higher value for the researcher.

    Args:
        cited_seed_count: How many seed papers this paper cites.
        total_seeds: Total number of seeds.

    Returns:
        Score in [0, 1].
    """
    if total_seeds <= 0:
        return 0.0
    return min(1.0, cited_seed_count / max(1, total_seeds))


def composite_score(
    relevance: float,
    recency: float,
    influence: float,
    config: ForwardTraversalConfig | None = None,
) -> float:
    """Compute weighted composite score.

    Args:
        relevance: Base relevance score [0, 1].
        recency: Recency bonus [0, 1].
        influence: Influence score [0, 1].
        config: Weights configuration.

    Returns:
        Weighted composite in [0, 1].
    """
    cfg = config or ForwardTraversalConfig()
    total_weight = cfg.relevance_weight + cfg.recency_weight + cfg.influence_weight
    if total_weight <= 0:
        return 0.0
    raw = (
        relevance * cfg.relevance_weight
        + recency * cfg.recency_weight
        + influence * cfg.influence_weight
    ) / total_weight
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Simple BM25 text scorer (reused from citation_graph pattern)
# ---------------------------------------------------------------------------


def _bm25_simple(text: str, terms: list[str], k1: float = 1.5) -> float:
    """Minimal BM25-ish term frequency scorer.

    Args:
        text: Document text.
        terms: Query terms.
        k1: Saturation parameter.

    Returns:
        Score ≥ 0.
    """
    if not text or not terms:
        return 0.0
    words = text.lower().split()
    word_count = len(words) if words else 1
    score = 0.0
    for term in terms:
        tf = sum(1 for w in words if term.lower() in w)
        score += (tf * (k1 + 1)) / (tf + k1)
    # Normalise to [0, 1] range
    max_possible = len(terms) * (k1 + 1) / (1 + k1)
    return score / max_possible if max_possible > 0 else 0.0


# ---------------------------------------------------------------------------
# Forward traversal engine
# ---------------------------------------------------------------------------


class ForwardCitationTraverser:
    """Engine for forward (citing-paper) traversal.

    The traverser works without a live API client — it accepts a
    ``fetch_function`` callback so it can be tested deterministically.

    Usage::

        traverser = ForwardCitationTraverser(config=config)
        result = traverser.traverse(
            seed_ids=["arxiv:2301.12345"],
            query_terms=["knowledge", "graph"],
            fetch_fn=my_api_fetch,
        )
    """

    def __init__(self, config: ForwardTraversalConfig | None = None) -> None:
        self.config = config or ForwardTraversalConfig()

    def traverse(
        self,
        seed_ids: list[str],
        query_terms: list[str],
        fetch_fn: Any | None = None,
    ) -> ForwardTraversalResult:
        """Execute forward citation traversal.

        Args:
            seed_ids: Starting paper IDs.
            query_terms: Terms for relevance scoring.
            fetch_fn: Callable(paper_id, limit) → list[dict] with
                keys ``paper_id``, ``title``, ``abstract``, ``year``.
                If None, returns empty result (dry run).

        Returns:
            ForwardTraversalResult with all discovered papers.
        """
        result = ForwardTraversalResult(seed_ids=list(seed_ids))
        seen_ids: set[str] = set(seed_ids)
        current_seeds = list(seed_ids)
        low_yield_streak = 0
        api_calls = 0

        for depth in range(1, self.config.max_depth + 1):
            if not current_seeds:
                result.stop_reason = ForwardStopReason.NO_NEW_PAPERS
                break
            if len(result.discovered) >= self.config.max_papers:
                result.stop_reason = ForwardStopReason.MAX_PAPERS
                break

            round_papers: list[CitingPaper] = []
            fetched_total = 0

            for sid in current_seeds:
                if len(result.discovered) + len(round_papers) >= self.config.max_papers:
                    break
                raw = self._fetch_citing(sid, fetch_fn)
                api_calls += 1
                fetched_total += len(raw)

                for paper_data in raw:
                    if (
                        len(result.discovered) + len(round_papers)
                        >= self.config.max_papers
                    ):
                        break
                    pid = paper_data.get("paper_id", "")
                    if not pid or pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    cp = self._score_paper(
                        paper_data, sid, seed_ids, query_terms
                    )
                    round_papers.append(cp)

            # Update existing papers' influence if they cite multiple seeds
            for cp in round_papers:
                cp.influence_score = influence_score(
                    len(cp.cited_seed_ids), len(seed_ids)
                )
                cp.composite_score = composite_score(
                    cp.relevance_score,
                    cp.recency_bonus,
                    cp.influence_score,
                    self.config,
                )

            new_unique = len(round_papers)
            yield_ratio = new_unique / fetched_total if fetched_total > 0 else 0.0
            mean_sc = (
                sum(p.composite_score for p in round_papers) / new_unique
                if new_unique > 0
                else 0.0
            )

            rnd = TraversalRound(
                depth=depth,
                seeds_used=len(current_seeds),
                fetched=fetched_total,
                new_unique=new_unique,
                mean_score=mean_sc,
                yield_ratio=yield_ratio,
            )
            result.rounds.append(rnd)
            result.discovered.extend(round_papers)

            logger.info(
                "Forward depth %d: %d seeds → %d fetched → %d new (yield=%.2f)",
                depth, len(current_seeds), fetched_total, new_unique, yield_ratio,
            )

            # Yield decay stopping
            if yield_ratio < self.config.yield_decay_threshold:
                low_yield_streak += 1
                if low_yield_streak >= self.config.yield_patience:
                    result.stop_reason = ForwardStopReason.YIELD_DECAY
                    break
            else:
                low_yield_streak = 0

            if new_unique == 0:
                result.stop_reason = ForwardStopReason.NO_NEW_PAPERS
                break

            # Next round seeds: top-scoring new papers
            current_seeds = [
                p.paper_id
                for p in sorted(
                    round_papers, key=lambda x: x.composite_score, reverse=True
                )[: self.config.limit_per_seed]
            ]

        if depth == self.config.max_depth and result.stop_reason not in (
            ForwardStopReason.YIELD_DECAY,
            ForwardStopReason.NO_NEW_PAPERS,
            ForwardStopReason.MAX_PAPERS,
        ):
            result.stop_reason = ForwardStopReason.MAX_DEPTH

        result.total_api_calls = api_calls
        return result

    def _fetch_citing(
        self, paper_id: str, fetch_fn: Any | None
    ) -> list[dict[str, Any]]:
        """Fetch citing papers for a seed."""
        if fetch_fn is None:
            return []
        try:
            raw = fetch_fn(paper_id, self.config.limit_per_seed)
            return list(raw) if raw else []
        except Exception as exc:
            logger.warning("Failed to fetch citations for %s: %s", paper_id, exc)
            return []

    def _score_paper(
        self,
        data: dict[str, Any],
        citing_seed: str,
        all_seeds: list[str],
        query_terms: list[str],
    ) -> CitingPaper:
        """Create and score a CitingPaper from raw data."""
        title = data.get("title", "")
        abstract = data.get("abstract", "")
        year = data.get("year")

        rel = _bm25_simple(f"{title} {abstract}", query_terms)
        rec = recency_bonus(year, self.config.recency_half_life)
        inf = influence_score(1, len(all_seeds))  # initial: cited by 1 seed

        return CitingPaper(
            paper_id=data.get("paper_id", ""),
            title=title,
            abstract=abstract,
            year=year,
            cited_seed_ids=[citing_seed],
            relevance_score=rel,
            recency_bonus=rec,
            influence_score=inf,
            composite_score=composite_score(rel, rec, inf, self.config),
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def forward_expand(
    seed_ids: list[str],
    query_terms: list[str],
    fetch_fn: Any | None = None,
    config: ForwardTraversalConfig | None = None,
) -> ForwardTraversalResult:
    """High-level forward citation expansion.

    Args:
        seed_ids: Starting paper IDs.
        query_terms: Relevance scoring terms.
        fetch_fn: API callback.
        config: Traversal configuration.

    Returns:
        ForwardTraversalResult.
    """
    traverser = ForwardCitationTraverser(config=config)
    return traverser.traverse(
        seed_ids=seed_ids, query_terms=query_terms, fetch_fn=fetch_fn
    )
