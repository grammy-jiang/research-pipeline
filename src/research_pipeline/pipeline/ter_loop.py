"""THINK→EXECUTE→REFLECT iterative gap-filling loop.

After initial pipeline run, identifies coverage gaps in the synthesis,
generates new queries to fill them, and re-runs search→screen to find
additional papers. Repeats up to max_iterations or until convergence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
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
        "for",
        "and",
        "nor",
        "but",
        "or",
        "yet",
        "so",
        "in",
        "on",
        "at",
        "to",
        "of",
        "with",
        "by",
        "from",
        "as",
        "into",
        "this",
        "that",
        "these",
        "those",
        "not",
        "no",
        "how",
        "what",
        "which",
    }
)

_GAP_MARKERS = (
    "?",
    "future work",
    "remains unclear",
    "not addressed",
    "limited",
    "gap",
    "missing",
    "unexplored",
    "further research",
)


@dataclass
class GapAnalysis:
    """Result of analyzing gaps in the current synthesis."""

    gaps: list[str]
    suggested_queries: list[str]
    iteration: int = 0
    gap_count: int = 0

    def __post_init__(self) -> None:
        self.gap_count = len(self.gaps)


@dataclass
class TERIteration:
    """Record of one THINK→EXECUTE→REFLECT cycle."""

    iteration: int
    gaps_found: list[str]
    queries_generated: list[str]
    new_papers_found: int = 0
    new_papers_relevant: int = 0
    converged: bool = False


@dataclass
class TERResult:
    """Overall result of the TER loop."""

    iterations: list[TERIteration] = field(default_factory=list)
    total_iterations: int = 0
    converged: bool = False
    convergence_reason: str = ""
    total_new_papers: int = 0


def _extract_key_terms(text: str, max_terms: int = 6) -> list[str]:
    """Extract informative terms from text, filtering stopwords."""
    words = text.split()
    return [w for w in words if w.lower() not in _STOPWORDS][:max_terms]


def _query_from_gap(gap: str) -> str:
    """Generate a search query from a gap description."""
    key_words = _extract_key_terms(gap)
    return " ".join(key_words)


def identify_gaps(
    synthesis_text: str,
    topic: str,
    existing_papers: list[str],
    llm_provider: object | None = None,
) -> GapAnalysis:
    """THINK phase: analyze synthesis to find coverage gaps.

    Uses heuristic analysis of the synthesis text to find:
    - Mentioned but unexplored topics
    - Open questions from the synthesis
    - Missing comparisons or perspectives

    Args:
        synthesis_text: Current synthesis markdown text.
        topic: Original research topic.
        existing_papers: List of paper titles already covered.
        llm_provider: Optional LLM for better gap detection.

    Returns:
        GapAnalysis with identified gaps and suggested queries.
    """
    gaps: list[str] = []
    queries: list[str] = []

    lines = synthesis_text.split("\n")

    # Heuristic gap detection from bullet lines with gap markers
    for line in lines:
        line_lower = line.lower().strip()
        if line.startswith("- ") and any(
            marker in line_lower for marker in _GAP_MARKERS
        ):
            gap = line.strip("- ").strip()
            if gap and len(gap) > 10 and gap not in gaps:
                gaps.append(gap)

    # Check for explicit "Open Questions" or "Research Gaps" sections
    in_open_questions = False
    for line in lines:
        if "open question" in line.lower() or "research gap" in line.lower():
            in_open_questions = True
            continue
        if in_open_questions:
            if line.startswith("#"):
                in_open_questions = False
                continue
            if line.startswith("- "):
                gap = line.strip("- ").strip()
                if gap and gap not in gaps and len(gap) > 10:
                    gaps.append(gap)

    # Generate search queries from gaps (limit to top 5)
    for gap in gaps[:5]:
        query = _query_from_gap(gap)
        if query and query not in queries:
            queries.append(query)

    # If LLM available, use it for better gap analysis
    if llm_provider is not None:
        try:
            prompt = (
                f"Given this research synthesis on '{topic}', identify the top 3 "
                f"coverage gaps not addressed by these papers: "
                f"{', '.join(existing_papers[:10])}.\n\n"
                f"Synthesis excerpt:\n{synthesis_text[:2000]}\n\n"
                f"Return each gap as a one-line description, one per line."
            )
            response = llm_provider.complete(prompt)  # type: ignore[union-attr]
            if response:
                llm_gaps = [
                    line.strip().lstrip("0123456789.-) ")
                    for line in response.strip().split("\n")
                    if line.strip() and len(line.strip()) > 10
                ]
                for g in llm_gaps[:3]:
                    if g not in gaps:
                        gaps.append(g)
                        query = _query_from_gap(g)
                        if query and query not in queries:
                            queries.append(query)
        except Exception as exc:
            logger.warning("LLM gap analysis failed, using heuristic only: %s", exc)

    return GapAnalysis(gaps=gaps, suggested_queries=queries)


def check_convergence(
    current: GapAnalysis,
    previous: GapAnalysis | None,
    iteration: int,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[bool, str]:
    """REFLECT phase: determine if the loop should stop.

    Convergence criteria:
    1. Max iterations reached
    2. No gaps found
    3. No new queries to try
    4. Gap count not decreasing (stalled)

    Args:
        current: Current iteration's gap analysis.
        previous: Previous iteration's gap analysis (None for first).
        iteration: Current iteration number (0-based).
        max_iterations: Maximum allowed iterations.

    Returns:
        Tuple of (converged: bool, reason: str).
    """
    if iteration >= max_iterations:
        return True, f"max iterations reached ({max_iterations})"

    if current.gap_count == 0:
        return True, "no gaps remaining"

    if not current.suggested_queries:
        return True, "no new queries to try"

    if previous is not None and current.gap_count >= previous.gap_count:
        return True, (
            f"gap count not decreasing "
            f"(was {previous.gap_count}, now {current.gap_count})"
        )

    return False, ""


def save_ter_state(
    run_root: Path,
    result: TERResult,
    iteration_data: dict[str, object] | None = None,
) -> Path:
    """Persist TER loop state to disk for resume support.

    Args:
        run_root: Run directory root.
        result: Current TER result.
        iteration_data: Optional data for the current iteration.

    Returns:
        Path to saved state file.
    """
    ter_dir = run_root / "ter_loop"
    ter_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, object] = {
        "total_iterations": result.total_iterations,
        "converged": result.converged,
        "convergence_reason": result.convergence_reason,
        "total_new_papers": result.total_new_papers,
        "iterations": [
            {
                "iteration": it.iteration,
                "gaps_found": it.gaps_found,
                "queries_generated": it.queries_generated,
                "new_papers_found": it.new_papers_found,
                "new_papers_relevant": it.new_papers_relevant,
                "converged": it.converged,
            }
            for it in result.iterations
        ],
    }

    state_path = ter_dir / "ter_state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    if iteration_data is not None:
        iter_path = ter_dir / f"iteration_{result.total_iterations}.json"
        iter_path.write_text(json.dumps(iteration_data, indent=2), encoding="utf-8")

    return state_path


def load_ter_state(run_root: Path) -> TERResult | None:
    """Load TER loop state from disk.

    Args:
        run_root: Run directory root.

    Returns:
        TERResult if state exists, None otherwise.
    """
    state_path = run_root / "ter_loop" / "ter_state.json"
    if not state_path.exists():
        return None

    data = json.loads(state_path.read_text(encoding="utf-8"))
    result = TERResult(
        total_iterations=data["total_iterations"],
        converged=data["converged"],
        convergence_reason=data.get("convergence_reason", ""),
        total_new_papers=data.get("total_new_papers", 0),
    )
    for it_data in data.get("iterations", []):
        result.iterations.append(
            TERIteration(
                iteration=it_data["iteration"],
                gaps_found=it_data["gaps_found"],
                queries_generated=it_data["queries_generated"],
                new_papers_found=it_data.get("new_papers_found", 0),
                new_papers_relevant=it_data.get("new_papers_relevant", 0),
                converged=it_data.get("converged", False),
            )
        )
    return result
