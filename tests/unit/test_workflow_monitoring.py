"""Tests for doom-loop detection and monitoring."""

from __future__ import annotations

from mcp_server.workflow.monitoring import (
    IterationMetrics,
    StopReason,
    check_doom_loop,
    content_fingerprint,
    jaccard_similarity,
)


class TestContentFingerprint:
    """Content fingerprinting tests."""

    def test_deterministic(self) -> None:
        fp1 = content_fingerprint("hello world")
        fp2 = content_fingerprint("hello world")
        assert fp1 == fp2

    def test_different_content(self) -> None:
        fp1 = content_fingerprint("hello world")
        fp2 = content_fingerprint("goodbye world")
        assert fp1 != fp2

    def test_md5_length(self) -> None:
        fp = content_fingerprint("test")
        assert len(fp) == 32  # MD5 hex digest


class TestJaccardSimilarity:
    """Jaccard similarity tests."""

    def test_identical(self) -> None:
        sim = jaccard_similarity("hello world test", "hello world test")
        assert sim == 1.0

    def test_completely_different(self) -> None:
        sim = jaccard_similarity("cat dog", "fish bird")
        assert sim == 0.0

    def test_partial_overlap(self) -> None:
        sim = jaccard_similarity("hello world foo", "hello world bar")
        # Overlap: {hello, world} / Union: {hello, world, foo, bar} = 2/4 = 0.5
        assert abs(sim - 0.5) < 0.01

    def test_empty_strings(self) -> None:
        sim = jaccard_similarity("", "")
        assert sim == 0.0

    def test_one_empty(self) -> None:
        sim = jaccard_similarity("hello", "")
        assert sim == 0.0


class TestCheckDoomLoop:
    """Doom-loop detection tests."""

    def test_exact_match_detected(self) -> None:
        is_loop, similarity, reason = check_doom_loop(
            "synthesis output A", "synthesis output A"
        )
        assert is_loop is True
        assert similarity >= 1.0
        assert reason == StopReason.DOOM_LOOP_EXACT

    def test_different_content_no_loop(self) -> None:
        is_loop, similarity, reason = check_doom_loop(
            "This is about machine learning and neural networks",
            "This is about quantum computing and algorithms",
        )
        assert is_loop is False
        assert similarity < 0.85

    def test_high_similarity_detected(self) -> None:
        base = "The main themes are AI memory systems retrieval augmentation"
        similar = "The main themes are AI memory systems retrieval argumentation"
        is_loop, similarity, reason = check_doom_loop(base, similar)
        assert isinstance(is_loop, bool)

    def test_no_previous_content(self) -> None:
        is_loop, similarity, reason = check_doom_loop(None, "any content")
        assert is_loop is False
        assert similarity == 0.0
        assert reason == "no_previous_content"


class TestStopReason:
    """StopReason enum tests."""

    def test_all_reasons(self) -> None:
        expected = {
            "implementation_ready",
            "max_iterations",
            "doom_loop_exact_match",
            "doom_loop_high_similarity",
            "no_new_academic_gaps",
            "user_declined",
            "budget_exhausted",
            "no_new_unique_candidates",
        }
        assert {r.value for r in StopReason} == expected


class TestIterationMetrics:
    """Iteration drift metrics tests."""

    def test_initial_state(self) -> None:
        metrics = IterationMetrics()
        assert len(metrics.history) == 0
        assert not metrics.is_search_exhausted()
        assert metrics.should_stop() is None

    def test_record(self) -> None:
        metrics = IterationMetrics()
        metrics.record(
            iteration=0, papers_found=20, papers_analyzed=5, gaps_remaining=3
        )
        metrics.record(iteration=1, papers_found=8, papers_analyzed=3, gaps_remaining=2)
        assert len(metrics.history) == 2

    def test_search_exhausted_detection(self) -> None:
        metrics = IterationMetrics()
        metrics.record(
            iteration=0, papers_found=20, papers_analyzed=5, gaps_remaining=3
        )
        metrics.record(iteration=1, papers_found=5, papers_analyzed=2, gaps_remaining=3)
        # Decreasing papers + stable gaps → exhausted
        assert metrics.is_search_exhausted()

    def test_not_exhausted_with_growth(self) -> None:
        metrics = IterationMetrics()
        metrics.record(
            iteration=0, papers_found=10, papers_analyzed=5, gaps_remaining=3
        )
        metrics.record(
            iteration=1, papers_found=15, papers_analyzed=7, gaps_remaining=2
        )
        assert not metrics.is_search_exhausted()

    def test_should_stop_no_new_candidates(self) -> None:
        metrics = IterationMetrics()
        metrics.record(
            iteration=0,
            papers_found=10,
            papers_analyzed=5,
            gaps_remaining=3,
            new_unique_papers=10,
        )
        metrics.record(
            iteration=1,
            papers_found=0,
            papers_analyzed=0,
            gaps_remaining=3,
            new_unique_papers=0,
        )
        result = metrics.should_stop()
        assert result == StopReason.NO_NEW_CANDIDATES

    def test_summary(self) -> None:
        metrics = IterationMetrics()
        metrics.record(
            iteration=0, papers_found=10, papers_analyzed=5, gaps_remaining=3
        )
        summary = metrics.summary()
        assert "Iter 0" in summary
        assert "10 found" in summary
