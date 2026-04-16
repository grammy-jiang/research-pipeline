"""Tests for Pass@k + Pass[k] dual evaluation metrics (B6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_pipeline.evaluation.dual_metrics import (
    AggregateMetrics,
    DualMetricResult,
    MetricsStore,
    SampleResult,
    aggregate_metrics,
    apply_safety_gate,
    compute_dual_metrics,
    compute_pass_at_k,
    compute_pass_bracket_k,
    evaluate_runs,
    evaluate_sample,
)

# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestSampleResult:
    """Tests for SampleResult dataclass."""

    def test_creation(self) -> None:
        s = SampleResult(sample_id="s1", run_id="run1", correct=True)
        assert s.sample_id == "s1"
        assert s.correct is True
        assert s.fabrication_detected is False

    def test_defaults(self) -> None:
        s = SampleResult(sample_id="s2", run_id="run2")
        assert s.quality_score == 0.0
        assert s.details == {}


class TestDualMetricResult:
    """Tests for DualMetricResult dataclass."""

    def test_to_dict_roundtrip(self) -> None:
        result = DualMetricResult(
            query="test query",
            k=5,
            n=10,
            c=7,
            pass_at_k=0.95,
            pass_bracket_k=0.6,
            safety_gate=1.0,
            gated_pass_at_k=0.95,
            gated_pass_bracket_k=0.6,
            fabrication_count=0,
            samples=[SampleResult("s1", "r1", correct=True)],
        )
        d = result.to_dict()
        assert d["query"] == "test query"
        assert d["k"] == 5
        assert d["pass_at_k"] == 0.95
        assert len(d["samples"]) == 1
        assert d["samples"][0]["correct"] is True


class TestAggregateMetrics:
    """Tests for AggregateMetrics dataclass."""

    def test_to_dict(self) -> None:
        agg = AggregateMetrics(
            total_queries=2,
            mean_pass_at_k=0.8,
            mean_pass_bracket_k=0.4,
            reliability_gap=0.4,
        )
        d = agg.to_dict()
        assert d["total_queries"] == 2
        assert d["reliability_gap"] == 0.4


# ---------------------------------------------------------------------------
# Core metric computation tests
# ---------------------------------------------------------------------------


class TestComputePassAtK:
    """Tests for compute_pass_at_k."""

    def test_all_correct(self) -> None:
        # All 10 samples correct, k=5 -> should be 1.0
        assert compute_pass_at_k(10, 10, 5) == 1.0

    def test_none_correct(self) -> None:
        assert compute_pass_at_k(10, 0, 5) == 0.0

    def test_some_correct(self) -> None:
        # n=10, c=3, k=5 -> P(at least 1 correct in 5)
        result = compute_pass_at_k(10, 3, 5)
        assert 0.0 < result < 1.0
        # With 3/10 correct, sampling 5, quite likely to get at least one
        assert result > 0.8

    def test_k_equals_n(self) -> None:
        # k=n=10, c=5 -> guaranteed to include at least one correct
        assert compute_pass_at_k(10, 5, 10) == 1.0

    def test_k_greater_than_n(self) -> None:
        # k > n: clamped to n
        result = compute_pass_at_k(5, 3, 10)
        assert result == 1.0  # k clamped to 5, c=3/5 guaranteed at least 1

    def test_n_zero(self) -> None:
        assert compute_pass_at_k(0, 0, 5) == 0.0

    def test_k_zero(self) -> None:
        assert compute_pass_at_k(10, 5, 0) == 0.0

    def test_one_correct_large_n(self) -> None:
        # n=100, c=1, k=5 -> small but nonzero
        result = compute_pass_at_k(100, 1, 5)
        assert 0.0 < result < 0.1

    def test_monotone_in_c(self) -> None:
        # More correct samples -> higher Pass@k
        r1 = compute_pass_at_k(10, 2, 5)
        r2 = compute_pass_at_k(10, 5, 5)
        r3 = compute_pass_at_k(10, 8, 5)
        assert r1 <= r2 <= r3


class TestComputePassBracketK:
    """Tests for compute_pass_bracket_k."""

    def test_all_correct(self) -> None:
        assert compute_pass_bracket_k(10, 10, 5) == 1.0

    def test_none_correct(self) -> None:
        assert compute_pass_bracket_k(10, 0, 5) == 0.0

    def test_some_correct(self) -> None:
        # n=10, c=3, k=5 -> P(all 5 correct) = C(3,5)/C(10,5) = 0
        assert compute_pass_bracket_k(10, 3, 5) == 0.0

    def test_c_equals_k(self) -> None:
        # n=10, c=5, k=5 -> C(5,5)/C(10,5) = 1/252
        result = compute_pass_bracket_k(10, 5, 5)
        assert result == pytest.approx(1.0 / 252, rel=1e-3)

    def test_k_equals_one(self) -> None:
        # k=1: Pass[1] = c/n
        result = compute_pass_bracket_k(10, 7, 1)
        assert result == pytest.approx(0.7, rel=1e-3)

    def test_n_zero(self) -> None:
        assert compute_pass_bracket_k(0, 0, 5) == 0.0

    def test_k_zero(self) -> None:
        assert compute_pass_bracket_k(10, 5, 0) == 0.0

    def test_pass_bracket_k_leq_pass_at_k(self) -> None:
        # Pass[k] should always be <= Pass@k
        for n, c, k in [(10, 5, 3), (20, 10, 5), (8, 4, 4)]:
            pak = compute_pass_at_k(n, c, k)
            pbk = compute_pass_bracket_k(n, c, k)
            assert pbk <= pak + 1e-10


class TestApplySafetyGate:
    """Tests for apply_safety_gate."""

    def test_no_fabrication(self) -> None:
        samples = [
            SampleResult("s1", "r1", correct=True),
            SampleResult("s2", "r2", correct=False),
        ]
        assert apply_safety_gate(samples) == 1.0

    def test_with_fabrication(self) -> None:
        samples = [
            SampleResult("s1", "r1", correct=True),
            SampleResult("s2", "r2", correct=True, fabrication_detected=True),
        ]
        assert apply_safety_gate(samples) == 0.0

    def test_empty_samples(self) -> None:
        assert apply_safety_gate([]) == 1.0


# ---------------------------------------------------------------------------
# Dual metric computation tests
# ---------------------------------------------------------------------------


class TestComputeDualMetrics:
    """Tests for compute_dual_metrics."""

    def test_basic_computation(self) -> None:
        samples = [
            SampleResult("s1", "r1", correct=True),
            SampleResult("s2", "r2", correct=True),
            SampleResult("s3", "r3", correct=False),
            SampleResult("s4", "r4", correct=True),
            SampleResult("s5", "r5", correct=False),
        ]
        result = compute_dual_metrics("test query", samples, k=3)

        assert result.query == "test query"
        assert result.n == 5
        assert result.c == 3
        assert result.k == 3
        assert result.pass_at_k > 0
        assert result.pass_bracket_k >= 0
        assert result.safety_gate == 1.0
        assert result.gated_pass_at_k == result.pass_at_k

    def test_safety_gate_zeros_scores(self) -> None:
        samples = [
            SampleResult("s1", "r1", correct=True, fabrication_detected=True),
            SampleResult("s2", "r2", correct=True),
        ]
        result = compute_dual_metrics("query", samples, k=2)

        assert result.safety_gate == 0.0
        assert result.gated_pass_at_k == 0.0
        assert result.gated_pass_bracket_k == 0.0
        assert result.fabrication_count == 1

    def test_empty_samples(self) -> None:
        result = compute_dual_metrics("query", [], k=5)
        assert result.n == 0
        assert result.pass_at_k == 0.0
        assert result.pass_bracket_k == 0.0

    def test_k_larger_than_samples(self) -> None:
        samples = [SampleResult("s1", "r1", correct=True)]
        result = compute_dual_metrics("query", samples, k=10)
        assert result.k == 1  # clamped


class TestAggregateMetrics2:
    """Tests for aggregate_metrics."""

    def test_aggregate_multiple(self) -> None:
        results = [
            DualMetricResult(
                query="q1",
                k=5,
                n=10,
                c=8,
                pass_at_k=0.95,
                pass_bracket_k=0.4,
                safety_gate=1.0,
                gated_pass_at_k=0.95,
                gated_pass_bracket_k=0.4,
            ),
            DualMetricResult(
                query="q2",
                k=5,
                n=10,
                c=5,
                pass_at_k=0.8,
                pass_bracket_k=0.2,
                safety_gate=0.0,
                gated_pass_at_k=0.0,
                gated_pass_bracket_k=0.0,
            ),
        ]
        agg = aggregate_metrics(results)

        assert agg.total_queries == 2
        assert agg.mean_pass_at_k == pytest.approx(0.875)
        assert agg.mean_pass_bracket_k == pytest.approx(0.3)
        assert agg.safety_violation_rate == pytest.approx(0.5)
        assert agg.reliability_gap == pytest.approx(0.575)

    def test_aggregate_empty(self) -> None:
        agg = aggregate_metrics([])
        assert agg.total_queries == 0
        assert agg.mean_pass_at_k == 0.0


# ---------------------------------------------------------------------------
# Sample evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateSample:
    """Tests for evaluate_sample."""

    def test_no_synthesis(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        result = evaluate_sample(run_dir, "run1")
        assert result.correct is False
        assert "No synthesis" in result.details["reason"]

    def test_valid_synthesis(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run2"
        summarize_dir = run_dir / "summarize"
        summarize_dir.mkdir(parents=True)

        synthesis = {
            "per_paper_summaries": [
                {
                    "title": "Paper A",
                    "findings": ["Finding 1", "Finding 2", "Finding 3"],
                    "evidence": [{"chunk_id": "c1"}],
                }
            ],
            "agreements": [
                {"claim": "Claim A", "supporting_papers": ["p1", "p2"], "evidence": []}
            ],
            "disagreements": [],
            "gaps": [{"description": "Gap 1"}],
        }
        (summarize_dir / "synthesis_report.json").write_text(json.dumps(synthesis))

        result = evaluate_sample(run_dir, "run2")
        assert result.correct is True
        assert result.quality_score > 0

    def test_fabrication_detection(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run3"
        summarize_dir = run_dir / "summarize"
        summarize_dir.mkdir(parents=True)

        synthesis = {
            "per_paper_summaries": [
                {
                    "title": "Paper B",
                    "findings": ["F1", "F2", "F3", "F4"],
                    "evidence": [],
                }
            ],
            "agreements": [],
            "disagreements": [],
            "gaps": [],
        }
        (summarize_dir / "synthesis_report.json").write_text(json.dumps(synthesis))

        result = evaluate_sample(run_dir, "run3")
        assert result.fabrication_detected is True

    def test_with_reference_findings(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run4"
        summarize_dir = run_dir / "summarize"
        summarize_dir.mkdir(parents=True)

        synthesis = {
            "per_paper_summaries": [
                {
                    "title": "Paper C",
                    "findings": [
                        "Transformers outperform LSTMs on translation tasks",
                        "Attention mechanism is the key innovation",
                    ],
                    "evidence": [{"chunk_id": "c1"}],
                }
            ],
            "agreements": [],
            "disagreements": [],
            "gaps": [],
        }
        (summarize_dir / "synthesis_report.json").write_text(json.dumps(synthesis))

        result = evaluate_sample(
            run_dir,
            "run4",
            reference_findings=[
                "Transformers outperform LSTMs on translation",
                "Self-attention mechanism enables parallel processing",
                "Model scales better than recurrent architectures",
            ],
            finding_match_threshold=0.3,
        )
        assert result.details["finding_count"] >= 2


# ---------------------------------------------------------------------------
# MetricsStore tests
# ---------------------------------------------------------------------------


class TestMetricsStore:
    """Tests for MetricsStore SQLite storage."""

    def test_store_and_retrieve(self, tmp_path: Path) -> None:
        db_path = tmp_path / "metrics.db"
        store = MetricsStore(db_path)
        try:
            result = DualMetricResult(
                query="test query",
                k=5,
                n=10,
                c=7,
                pass_at_k=0.95,
                pass_bracket_k=0.5,
                safety_gate=1.0,
                gated_pass_at_k=0.95,
                gated_pass_bracket_k=0.5,
                samples=[SampleResult("s1", "r1", correct=True)],
            )
            record_id = store.store_result(result)
            assert record_id >= 1

            history = store.get_history("test query")
            assert len(history) == 1
            assert history[0].query == "test query"
            assert history[0].pass_at_k == 0.95
        finally:
            store.close()

    def test_get_latest(self, tmp_path: Path) -> None:
        db_path = tmp_path / "metrics2.db"
        store = MetricsStore(db_path)
        try:
            for i in range(3):
                store.store_result(
                    DualMetricResult(
                        query="q1",
                        k=5,
                        n=10,
                        c=i + 1,
                        pass_at_k=0.1 * (i + 1),
                        pass_bracket_k=0.05 * (i + 1),
                    )
                )
            latest = store.get_latest("q1")
            assert latest is not None
            assert latest.c == 3
        finally:
            store.close()

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "metrics3.db"
        store = MetricsStore(db_path)
        try:
            assert store.get_latest("nonexistent") is None
        finally:
            store.close()

    def test_get_all(self, tmp_path: Path) -> None:
        db_path = tmp_path / "metrics4.db"
        store = MetricsStore(db_path)
        try:
            store.store_result(DualMetricResult(query="q1", k=5, n=10, c=5))
            store.store_result(DualMetricResult(query="q2", k=5, n=10, c=8))
            all_results = store.get_all()
            assert len(all_results) == 2
        finally:
            store.close()


# ---------------------------------------------------------------------------
# End-to-end evaluate_runs tests
# ---------------------------------------------------------------------------


class TestEvaluateRuns:
    """Tests for evaluate_runs high-level function."""

    def test_evaluate_multiple_runs(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        runs_dir = workspace / "runs"

        for i in range(3):
            run_dir = runs_dir / f"run-{i}"
            summarize_dir = run_dir / "summarize"
            summarize_dir.mkdir(parents=True)

            synthesis = {
                "per_paper_summaries": [
                    {
                        "title": f"Paper {i}",
                        "findings": [f"Finding {i}a", f"Finding {i}b"],
                        "evidence": [{"chunk_id": f"c{i}"}] if i < 2 else [],
                    }
                ],
                "agreements": [{"claim": f"Agree {i}", "supporting_papers": ["p1"]}],
                "disagreements": [],
                "gaps": [{"description": f"Gap {i}"}] if i == 0 else [],
            }
            (summarize_dir / "synthesis_report.json").write_text(json.dumps(synthesis))

        result = evaluate_runs(
            workspace,
            "test research",
            ["run-0", "run-1", "run-2"],
            k=3,
            store_results=True,
        )

        assert result.n == 3
        assert result.query == "test research"
        assert 0.0 <= result.pass_at_k <= 1.0
        assert 0.0 <= result.pass_bracket_k <= 1.0

        # Check DB was created
        assert (workspace / ".dual_metrics.db").exists()

    def test_evaluate_missing_runs(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        (workspace / "runs").mkdir(parents=True)

        result = evaluate_runs(
            workspace,
            "query",
            ["missing-1", "missing-2"],
            k=2,
            store_results=False,
        )

        assert result.n == 2
        assert result.c == 0
        assert result.pass_at_k == 0.0
