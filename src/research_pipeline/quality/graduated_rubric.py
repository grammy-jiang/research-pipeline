"""Graduated rubric scoring for research outputs.

Extends the RACE scoring framework with discrete grade levels
(Excellent / Good / Adequate / Poor) and per-dimension descriptors.
Replaces binary pass/fail gating with nuanced quality tiers.

Each dimension (Readability, Actionability, Comprehensiveness,
Evidence) is mapped to a 4-level grade with configurable thresholds.
Composite grades aggregate dimension grades for overall assessment.

References:
    Deep-research report Theme 7 (Output Quality Control) — graduated
    rubrics as a replacement for binary pass/fail scoring.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from research_pipeline.quality.race_scoring import RACEScore, compute_race_score

logger = logging.getLogger(__name__)


# ── Grade levels ─────────────────────────────────────────────────────


class Grade(IntEnum):
    """Discrete quality grade levels (higher = better)."""

    POOR = 1
    ADEQUATE = 2
    GOOD = 3
    EXCELLENT = 4


GRADE_LABELS: dict[Grade, str] = {
    Grade.POOR: "Poor",
    Grade.ADEQUATE: "Adequate",
    Grade.GOOD: "Good",
    Grade.EXCELLENT: "Excellent",
}


# ── Threshold configuration ──────────────────────────────────────────


@dataclass(frozen=True)
class GradeThresholds:
    """Score thresholds for mapping continuous [0,1] scores to grades.

    Scores >= excellent → EXCELLENT
    Scores >= good      → GOOD
    Scores >= adequate  → ADEQUATE
    Otherwise           → POOR
    """

    excellent: float = 0.80
    good: float = 0.60
    adequate: float = 0.40

    def classify(self, score: float) -> Grade:
        """Map a continuous score to a discrete grade."""
        if score >= self.excellent:
            return Grade.EXCELLENT
        if score >= self.good:
            return Grade.GOOD
        if score >= self.adequate:
            return Grade.ADEQUATE
        return Grade.POOR


DEFAULT_THRESHOLDS = GradeThresholds()


# ── Per-dimension grade descriptors ──────────────────────────────────

DIMENSION_DESCRIPTORS: dict[str, dict[Grade, str]] = {
    "readability": {
        Grade.EXCELLENT: (
            "Clear, well-structured prose with appropriate sentence "
            "length, good heading density, and effective use of lists."
        ),
        Grade.GOOD: (
            "Generally readable with minor structural issues; "
            "some sections could use better organization."
        ),
        Grade.ADEQUATE: (
            "Readable but with notable issues in sentence length, "
            "paragraph structure, or heading usage."
        ),
        Grade.POOR: (
            "Difficult to read; poor structure, inconsistent "
            "formatting, or excessive sentence length."
        ),
    },
    "actionability": {
        Grade.EXCELLENT: (
            "Strong practical recommendations with specific, "
            "measurable findings and clear next steps."
        ),
        Grade.GOOD: (
            "Contains useful recommendations with some specific "
            "findings; could be more actionable."
        ),
        Grade.ADEQUATE: (
            "Some recommendations present but lacking specificity; "
            "few concrete findings."
        ),
        Grade.POOR: (
            "No clear recommendations or actionable findings; purely descriptive."
        ),
    },
    "comprehensiveness": {
        Grade.EXCELLENT: (
            "Thorough coverage of all expected sections with "
            "diverse citations and appropriate depth."
        ),
        Grade.GOOD: (
            "Covers most expected sections with reasonable "
            "citation diversity; minor gaps."
        ),
        Grade.ADEQUATE: (
            "Covers core sections but missing important areas; "
            "limited citation diversity."
        ),
        Grade.POOR: (
            "Major sections missing; insufficient depth or citation coverage."
        ),
    },
    "evidence": {
        Grade.EXCELLENT: (
            "Rich evidence integration with high citation density, "
            "confidence annotations, and evidence mapping."
        ),
        Grade.GOOD: (
            "Good citation density with some confidence "
            "annotations; evidence generally well-integrated."
        ),
        Grade.ADEQUATE: (
            "Some citations present but sparse; limited confidence "
            "annotations or evidence mapping."
        ),
        Grade.POOR: (
            "Few or no citations; claims unsupported by evidence; "
            "no confidence annotations."
        ),
    },
}


# ── Rubric result ────────────────────────────────────────────────────


@dataclass
class DimensionGrade:
    """Grade for a single RACE dimension."""

    dimension: str
    score: float
    grade: Grade
    label: str
    descriptor: str


@dataclass
class RubricResult:
    """Full graduated rubric result for a research report.

    Attributes:
        dimensions: Per-dimension grades.
        overall_score: Numeric overall score [0, 1].
        overall_grade: Composite grade (minimum of dimension grades).
        race_score: Underlying RACE score object.
        pass_threshold: Minimum grade for passing.
        passed: Whether the report meets the pass threshold.
        summary: Human-readable summary string.
    """

    dimensions: list[DimensionGrade] = field(default_factory=list)
    overall_score: float = 0.0
    overall_grade: Grade = Grade.POOR
    race_score: RACEScore | None = None
    pass_threshold: Grade = Grade.ADEQUATE
    passed: bool = False
    summary: str = ""


# ── Core scoring function ────────────────────────────────────────────


def score_rubric(
    text: str,
    thresholds: GradeThresholds | None = None,
    pass_threshold: Grade = Grade.ADEQUATE,
    race_score: RACEScore | None = None,
) -> RubricResult:
    """Score a research report using the graduated rubric.

    Args:
        text: Markdown report content.
        thresholds: Grade thresholds.  Defaults to ``DEFAULT_THRESHOLDS``.
        pass_threshold: Minimum overall grade to pass.
        race_score: Pre-computed RACE score.  If ``None``, computed
            from *text*.

    Returns:
        A ``RubricResult`` with per-dimension grades and overall assessment.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if race_score is None:
        race_score = compute_race_score(text)

    dimension_scores = {
        "readability": race_score.readability,
        "actionability": race_score.actionability,
        "comprehensiveness": race_score.comprehensiveness,
        "evidence": race_score.evidence,
    }

    dimensions: list[DimensionGrade] = []
    for dim_name, dim_score in dimension_scores.items():
        grade = thresholds.classify(dim_score)
        descriptor = DIMENSION_DESCRIPTORS.get(dim_name, {}).get(grade, "")
        dimensions.append(
            DimensionGrade(
                dimension=dim_name,
                score=dim_score,
                grade=grade,
                label=GRADE_LABELS[grade],
                descriptor=descriptor,
            )
        )

    # Overall grade = minimum dimension grade (weakest link)
    overall_grade = Grade(min(d.grade for d in dimensions))
    passed = overall_grade >= pass_threshold

    summary = _build_summary(dimensions, overall_grade, passed, pass_threshold)

    logger.debug(
        "Rubric result: overall=%s (%s), passed=%s",
        GRADE_LABELS[overall_grade],
        race_score.overall,
        passed,
    )

    return RubricResult(
        dimensions=dimensions,
        overall_score=race_score.overall,
        overall_grade=overall_grade,
        race_score=race_score,
        pass_threshold=pass_threshold,
        passed=passed,
        summary=summary,
    )


def _build_summary(
    dimensions: list[DimensionGrade],
    overall: Grade,
    passed: bool,
    threshold: Grade,
) -> str:
    """Build a human-readable rubric summary."""
    parts = [f"Overall: {GRADE_LABELS[overall]}"]
    for d in dimensions:
        parts.append(f"  {d.dimension.capitalize()}: {d.label} ({d.score:.2f})")
    status = "PASS" if passed else "FAIL"
    parts.append(f"Status: {status} (threshold: {GRADE_LABELS[threshold]})")
    return "\n".join(parts)


# ── Configurable criteria (FrontierFinance-style rubric extension) ───


@dataclass(frozen=True)
class Criterion:
    """A single configurable rubric criterion.

    Criteria are grouped under a dimension (e.g. ``evidence``) and scored
    on ``[0, 1]``. Weights are relative within a dimension; absolute
    weights are normalised at aggregation time. A criterion is satisfied
    if its score ≥ ``pass_threshold``.
    """

    name: str
    dimension: str
    weight: float = 1.0
    pass_threshold: float = 0.5
    description: str = ""


@dataclass
class CriterionResult:
    """Scored single-criterion outcome."""

    criterion: Criterion
    score: float
    passed: bool


def score_criteria(
    text: str,
    criteria: list[Criterion],
    scorer: Callable[[str, Criterion], float] | None = None,
) -> list[CriterionResult]:
    """Score each criterion in *criteria* against *text*.

    Args:
        text: Report content to score.
        criteria: Configured criteria list (can be loaded from JSON).
        scorer: Optional function ``(text, criterion) -> score``. The
            default is a deterministic keyword-density scorer: criterion
            passes when one of the whitespace-separated keywords in
            ``criterion.description`` appears in ``text``. This gives
            tests a hook for stubbing without requiring an LLM.
    """
    results: list[CriterionResult] = []
    actual_scorer = scorer if scorer is not None else _default_criterion_scorer
    for c in criteria:
        score = float(max(0.0, min(1.0, actual_scorer(text, c))))
        results.append(
            CriterionResult(criterion=c, score=score, passed=score >= c.pass_threshold)
        )
    return results


def _default_criterion_scorer(text: str, criterion: Criterion) -> float:
    """Simple keyword-based scorer used when no custom scorer is supplied."""
    if not criterion.description:
        return 0.0
    needle_tokens = [t for t in criterion.description.lower().split() if len(t) > 3]
    if not needle_tokens:
        return 0.0
    haystack = text.lower()
    hits = sum(1 for t in needle_tokens if t in haystack)
    return hits / len(needle_tokens)


def aggregate_criteria_by_dimension(
    results: list[CriterionResult],
) -> dict[str, float]:
    """Aggregate criterion scores into a per-dimension weighted mean."""
    grouped: dict[str, list[CriterionResult]] = {}
    for r in results:
        grouped.setdefault(r.criterion.dimension, []).append(r)
    out: dict[str, float] = {}
    for dim, items in grouped.items():
        total_w = sum(max(0.0, it.criterion.weight) for it in items)
        if total_w <= 0:
            out[dim] = 0.0
            continue
        weighted = sum(it.score * max(0.0, it.criterion.weight) for it in items)
        out[dim] = round(weighted / total_w, 6)
    return out


def load_criteria_from_json(payload: list[dict[str, Any]]) -> list[Criterion]:
    """Build :class:`Criterion` list from a JSON-compatible payload.

    Each entry must have ``name`` and ``dimension``. ``weight``,
    ``pass_threshold``, and ``description`` are optional with sensible
    defaults.
    """
    criteria: list[Criterion] = []
    for entry in payload:
        try:
            criteria.append(
                Criterion(
                    name=str(entry["name"]),
                    dimension=str(entry["dimension"]),
                    weight=float(entry.get("weight", 1.0)),
                    pass_threshold=float(entry.get("pass_threshold", 0.5)),
                    description=str(entry.get("description", "")),
                )
            )
        except KeyError as exc:
            raise ValueError(f"criterion entry missing required field: {exc}") from exc
    return criteria


# ── Batch scoring ────────────────────────────────────────────────────


def score_rubric_batch(
    texts: list[str],
    thresholds: GradeThresholds | None = None,
    pass_threshold: Grade = Grade.ADEQUATE,
) -> list[RubricResult]:
    """Score multiple reports using the graduated rubric.

    Args:
        texts: List of markdown report contents.
        thresholds: Shared grade thresholds.
        pass_threshold: Shared pass threshold.

    Returns:
        List of ``RubricResult`` objects.
    """
    return [
        score_rubric(text, thresholds=thresholds, pass_threshold=pass_threshold)
        for text in texts
    ]


# ── Summary aggregation ──────────────────────────────────────────────


@dataclass
class BatchRubricStats:
    """Aggregate statistics for a batch of rubric results."""

    total: int
    passed: int
    failed: int
    pass_rate: float
    grade_distribution: dict[str, int]
    avg_overall_score: float
    weakest_dimension: str
    details: list[dict[str, Any]] = field(default_factory=list)


def compute_batch_rubric_stats(results: list[RubricResult]) -> BatchRubricStats:
    """Compute aggregate rubric statistics for a batch."""
    if not results:
        return BatchRubricStats(
            total=0,
            passed=0,
            failed=0,
            pass_rate=0.0,
            grade_distribution={},
            avg_overall_score=0.0,
            weakest_dimension="",
        )

    total = len(results)
    passed = sum(1 for r in results if r.passed)

    grade_dist: dict[str, int] = {
        "Excellent": 0,
        "Good": 0,
        "Adequate": 0,
        "Poor": 0,
    }
    for r in results:
        label = GRADE_LABELS[r.overall_grade]
        grade_dist[label] = grade_dist.get(label, 0) + 1

    avg_score = sum(r.overall_score for r in results) / total

    # Find weakest dimension across all results
    dim_avgs: dict[str, list[float]] = {}
    for r in results:
        for d in r.dimensions:
            dim_avgs.setdefault(d.dimension, []).append(d.score)

    weakest = ""
    if dim_avgs:
        weakest = min(
            dim_avgs,
            key=lambda k: sum(dim_avgs[k]) / len(dim_avgs[k]),
        )

    return BatchRubricStats(
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total,
        grade_distribution=grade_dist,
        avg_overall_score=round(avg_score, 4),
        weakest_dimension=weakest,
        details=[
            {
                "overall_grade": GRADE_LABELS[r.overall_grade],
                "overall_score": r.overall_score,
                "passed": r.passed,
            }
            for r in results
        ],
    )
