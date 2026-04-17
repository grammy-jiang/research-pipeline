"""Tests for quality.graduated_rubric — graduated rubric scoring."""

from __future__ import annotations

from research_pipeline.quality.graduated_rubric import (
    DIMENSION_DESCRIPTORS,
    GRADE_LABELS,
    BatchRubricStats,
    DimensionGrade,
    Grade,
    GradeThresholds,
    RubricResult,
    compute_batch_rubric_stats,
    score_rubric,
    score_rubric_batch,
)
from research_pipeline.quality.race_scoring import RACEScore

# ── Grade enum ───────────────────────────────────────────────────────


class TestGrade:
    def test_ordering(self) -> None:
        assert Grade.POOR < Grade.ADEQUATE < Grade.GOOD < Grade.EXCELLENT

    def test_int_values(self) -> None:
        assert int(Grade.POOR) == 1
        assert int(Grade.EXCELLENT) == 4


# ── GradeThresholds ──────────────────────────────────────────────────


class TestGradeThresholds:
    def test_classify_excellent(self) -> None:
        t = GradeThresholds()
        assert t.classify(0.85) == Grade.EXCELLENT

    def test_classify_good(self) -> None:
        t = GradeThresholds()
        assert t.classify(0.65) == Grade.GOOD

    def test_classify_adequate(self) -> None:
        t = GradeThresholds()
        assert t.classify(0.45) == Grade.ADEQUATE

    def test_classify_poor(self) -> None:
        t = GradeThresholds()
        assert t.classify(0.20) == Grade.POOR

    def test_classify_boundary_excellent(self) -> None:
        t = GradeThresholds()
        assert t.classify(0.80) == Grade.EXCELLENT

    def test_classify_boundary_good(self) -> None:
        t = GradeThresholds()
        assert t.classify(0.60) == Grade.GOOD

    def test_classify_boundary_adequate(self) -> None:
        t = GradeThresholds()
        assert t.classify(0.40) == Grade.ADEQUATE

    def test_custom_thresholds(self) -> None:
        t = GradeThresholds(excellent=0.90, good=0.70, adequate=0.50)
        assert t.classify(0.85) == Grade.GOOD  # below 0.90


# ── Dimension descriptors ────────────────────────────────────────────


class TestDimensionDescriptors:
    def test_all_dimensions_present(self) -> None:
        expected = {"readability", "actionability", "comprehensiveness", "evidence"}
        assert set(DIMENSION_DESCRIPTORS.keys()) == expected

    def test_all_grades_have_descriptors(self) -> None:
        for dim, grades in DIMENSION_DESCRIPTORS.items():
            for grade in Grade:
                assert grade in grades, f"{dim} missing descriptor for {grade}"


# ── score_rubric with pre-computed RACE score ────────────────────────


class TestScoreRubric:
    def _make_race(
        self, r: float = 0.5, a: float = 0.5, c: float = 0.5, e: float = 0.5
    ) -> RACEScore:
        return RACEScore(
            readability=r,
            actionability=a,
            comprehensiveness=c,
            evidence=e,
            overall=round((r + a + c + e) / 4, 4),
        )

    def test_all_excellent(self) -> None:
        race = self._make_race(0.9, 0.9, 0.9, 0.9)
        result = score_rubric("", race_score=race)
        assert result.overall_grade == Grade.EXCELLENT
        assert result.passed is True
        assert all(d.grade == Grade.EXCELLENT for d in result.dimensions)

    def test_all_poor(self) -> None:
        race = self._make_race(0.1, 0.1, 0.1, 0.1)
        result = score_rubric("", race_score=race)
        assert result.overall_grade == Grade.POOR
        assert result.passed is False

    def test_weakest_link_determines_overall(self) -> None:
        race = self._make_race(0.9, 0.9, 0.9, 0.3)
        result = score_rubric("", race_score=race)
        # Overall is minimum = POOR (evidence = 0.3)
        assert result.overall_grade == Grade.POOR

    def test_pass_threshold_adequate(self) -> None:
        race = self._make_race(0.5, 0.5, 0.5, 0.5)
        result = score_rubric("", race_score=race, pass_threshold=Grade.ADEQUATE)
        assert result.passed is True

    def test_pass_threshold_good_fails_adequate(self) -> None:
        race = self._make_race(0.5, 0.5, 0.5, 0.5)
        result = score_rubric("", race_score=race, pass_threshold=Grade.GOOD)
        assert result.passed is False

    def test_returns_rubric_result(self) -> None:
        race = self._make_race()
        result = score_rubric("", race_score=race)
        assert isinstance(result, RubricResult)
        assert len(result.dimensions) == 4

    def test_dimension_grades_have_descriptors(self) -> None:
        race = self._make_race(0.9, 0.5, 0.7, 0.3)
        result = score_rubric("", race_score=race)
        for d in result.dimensions:
            assert isinstance(d, DimensionGrade)
            assert d.descriptor != ""
            assert d.label in GRADE_LABELS.values()

    def test_summary_contains_status(self) -> None:
        race = self._make_race(0.9, 0.9, 0.9, 0.9)
        result = score_rubric("", race_score=race)
        assert "PASS" in result.summary
        assert "Overall:" in result.summary

    def test_custom_thresholds(self) -> None:
        race = self._make_race(0.75, 0.75, 0.75, 0.75)
        strict = GradeThresholds(excellent=0.95, good=0.80, adequate=0.60)
        result = score_rubric("", race_score=race, thresholds=strict)
        assert result.overall_grade == Grade.ADEQUATE

    def test_from_text(self) -> None:
        # Minimal text — should produce some score
        text = "# Summary\n\nThis is a test report. It has some content."
        result = score_rubric(text)
        assert result.race_score is not None
        assert 0.0 <= result.overall_score <= 1.0


# ── score_rubric_batch ───────────────────────────────────────────────


class TestScoreRubricBatch:
    def test_empty_batch(self) -> None:
        results = score_rubric_batch([])
        assert results == []

    def test_batch_returns_list(self) -> None:
        texts = ["# Report 1\nSome content.", "# Report 2\nMore content."]
        results = score_rubric_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, RubricResult) for r in results)


# ── compute_batch_rubric_stats ───────────────────────────────────────


class TestComputeBatchRubricStats:
    def _make_result(self, overall: Grade, passed: bool, score: float) -> RubricResult:
        return RubricResult(
            dimensions=[
                DimensionGrade(
                    dimension="readability",
                    score=score,
                    grade=overall,
                    label=GRADE_LABELS[overall],
                    descriptor="",
                ),
            ],
            overall_score=score,
            overall_grade=overall,
            passed=passed,
        )

    def test_empty_results(self) -> None:
        stats = compute_batch_rubric_stats([])
        assert stats.total == 0
        assert stats.pass_rate == 0.0

    def test_all_passing(self) -> None:
        results = [
            self._make_result(Grade.EXCELLENT, True, 0.9),
            self._make_result(Grade.GOOD, True, 0.7),
        ]
        stats = compute_batch_rubric_stats(results)
        assert stats.total == 2
        assert stats.passed == 2
        assert stats.failed == 0
        assert stats.pass_rate == 1.0

    def test_mixed_pass_fail(self) -> None:
        results = [
            self._make_result(Grade.EXCELLENT, True, 0.9),
            self._make_result(Grade.POOR, False, 0.2),
        ]
        stats = compute_batch_rubric_stats(results)
        assert stats.passed == 1
        assert stats.failed == 1
        assert stats.pass_rate == 0.5

    def test_grade_distribution(self) -> None:
        results = [
            self._make_result(Grade.EXCELLENT, True, 0.9),
            self._make_result(Grade.EXCELLENT, True, 0.85),
            self._make_result(Grade.GOOD, True, 0.7),
        ]
        stats = compute_batch_rubric_stats(results)
        assert stats.grade_distribution["Excellent"] == 2
        assert stats.grade_distribution["Good"] == 1

    def test_returns_batch_stats(self) -> None:
        results = [self._make_result(Grade.GOOD, True, 0.7)]
        stats = compute_batch_rubric_stats(results)
        assert isinstance(stats, BatchRubricStats)
        assert len(stats.details) == 1
