"""Tests for THINK→EXECUTE→REFLECT iterative gap-filling loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_pipeline.pipeline.ter_loop import (
    MAX_ITERATIONS,
    GapAnalysis,
    TERIteration,
    TERResult,
    check_convergence,
    identify_gaps,
    load_ter_state,
    save_ter_state,
)

# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


class TestGapAnalysis:
    """GapAnalysis dataclass tests."""

    def test_construction_basic(self) -> None:
        ga = GapAnalysis(gaps=["gap1", "gap2"], suggested_queries=["q1"])
        assert ga.gaps == ["gap1", "gap2"]
        assert ga.suggested_queries == ["q1"]
        assert ga.gap_count == 2

    def test_gap_count_auto_set(self) -> None:
        ga = GapAnalysis(gaps=["a", "b", "c"], suggested_queries=[])
        assert ga.gap_count == 3

    def test_empty_gaps(self) -> None:
        ga = GapAnalysis(gaps=[], suggested_queries=[])
        assert ga.gap_count == 0

    def test_iteration_field(self) -> None:
        ga = GapAnalysis(gaps=["g"], suggested_queries=[], iteration=2)
        assert ga.iteration == 2


class TestTERIteration:
    """TERIteration dataclass tests."""

    def test_construction(self) -> None:
        it = TERIteration(
            iteration=0,
            gaps_found=["g1"],
            queries_generated=["q1"],
            new_papers_found=5,
            new_papers_relevant=3,
            converged=False,
        )
        assert it.iteration == 0
        assert it.gaps_found == ["g1"]
        assert it.queries_generated == ["q1"]
        assert it.new_papers_found == 5
        assert it.new_papers_relevant == 3
        assert it.converged is False

    def test_defaults(self) -> None:
        it = TERIteration(iteration=1, gaps_found=[], queries_generated=[])
        assert it.new_papers_found == 0
        assert it.new_papers_relevant == 0
        assert it.converged is False


class TestTERResult:
    """TERResult dataclass tests."""

    def test_construction_defaults(self) -> None:
        r = TERResult()
        assert r.iterations == []
        assert r.total_iterations == 0
        assert r.converged is False
        assert r.convergence_reason == ""
        assert r.total_new_papers == 0

    def test_construction_with_values(self) -> None:
        r = TERResult(
            total_iterations=2,
            converged=True,
            convergence_reason="no gaps remaining",
            total_new_papers=10,
        )
        assert r.total_iterations == 2
        assert r.converged is True
        assert r.convergence_reason == "no gaps remaining"


# ---------------------------------------------------------------------------
# identify_gaps()
# ---------------------------------------------------------------------------


class TestIdentifyGaps:
    """Tests for identify_gaps heuristic gap detection."""

    def test_empty_synthesis(self) -> None:
        result = identify_gaps("", "AI agents", [])
        assert result.gaps == []
        assert result.suggested_queries == []
        assert result.gap_count == 0

    def test_synthesis_with_question_marks(self) -> None:
        text = (
            "# Synthesis\n"
            "- How does retrieval augmented generation scale?\n"
            "- What are the limits of context windows?\n"
        )
        result = identify_gaps(text, "LLM memory", [])
        assert result.gap_count >= 2
        assert any("retrieval" in g.lower() for g in result.gaps)

    def test_synthesis_with_open_questions_section(self) -> None:
        text = (
            "# Results\n"
            "Some findings here.\n"
            "## Open Questions\n"
            "- What impact does fine-tuning have on long-term memory?\n"
            "- How do different architectures compare for retrieval tasks?\n"
            "# Conclusion\n"
            "Done.\n"
        )
        result = identify_gaps(text, "memory", [])
        assert result.gap_count >= 2
        assert any("fine-tuning" in g.lower() for g in result.gaps)

    def test_synthesis_with_remains_unclear(self) -> None:
        text = (
            "- The relationship between context length and "
            "accuracy remains unclear for most models\n"
        )
        result = identify_gaps(text, "context", [])
        assert result.gap_count >= 1
        assert any("remains unclear" in g.lower() for g in result.gaps)

    def test_synthesis_with_gap_marker(self) -> None:
        text = (
            "- There is a significant gap in understanding multi-agent coordination\n"
        )
        result = identify_gaps(text, "agents", [])
        assert result.gap_count >= 1

    def test_generates_queries_from_gaps(self) -> None:
        text = (
            "- Future work should explore efficient "
            "transformer training methods\n"
            "- The impact of data quality on model performance "
            "remains unclear and needs further investigation\n"
        )
        result = identify_gaps(text, "transformers", [])
        assert len(result.suggested_queries) > 0
        # Queries should not contain stopwords
        for q in result.suggested_queries:
            words = q.lower().split()
            assert "the" not in words
            assert "of" not in words

    def test_max_five_gap_queries(self) -> None:
        lines = []
        for i in range(10):
            lines.append(
                f"- Research gap number {i}: relationship "
                f"between factor {i} and outcome "
                f"remains unclear\n"
            )
        text = "".join(lines)
        result = identify_gaps(text, "topic", [])
        assert len(result.suggested_queries) <= 5

    def test_llm_provider_none_no_error(self) -> None:
        result = identify_gaps("some text", "topic", [], llm_provider=None)
        assert isinstance(result, GapAnalysis)

    def test_short_lines_ignored(self) -> None:
        text = "- Too short?\n- Also tiny\n"
        result = identify_gaps(text, "topic", [])
        # Lines with <= 10 chars after stripping should be ignored
        assert result.gap_count == 0

    def test_research_gap_section(self) -> None:
        text = (
            "## Research Gaps\n"
            "- No studies address the intersection of "
            "reinforcement learning and symbolic reasoning\n"
            "- Scalability of graph-based memory structures "
            "is not well understood\n"
        )
        result = identify_gaps(text, "AI", [])
        assert result.gap_count >= 2

    def test_no_duplicate_gaps_across_sections(self) -> None:
        text = (
            "- The scalability problem remains unclear for distributed systems\n"
            "## Open Questions\n"
            "- The scalability problem remains unclear for distributed systems\n"
        )
        result = identify_gaps(text, "systems", [])
        # Should not have the same gap twice
        unique_gaps = set(result.gaps)
        assert len(unique_gaps) == len(result.gaps)

    def test_llm_provider_failure_graceful(self) -> None:
        class FailingLLM:
            def complete(self, prompt: str) -> str:
                raise RuntimeError("LLM unavailable")

        result = identify_gaps(
            "- Some gap remains unclear in the data\n",
            "topic",
            [],
            llm_provider=FailingLLM(),
        )
        # Should still return heuristic results
        assert isinstance(result, GapAnalysis)


# ---------------------------------------------------------------------------
# check_convergence()
# ---------------------------------------------------------------------------


class TestCheckConvergence:
    """Tests for convergence detection."""

    def test_max_iterations_reached(self) -> None:
        current = GapAnalysis(gaps=["g"], suggested_queries=["q"])
        converged, reason = check_convergence(
            current, None, iteration=3, max_iterations=3
        )
        assert converged is True
        assert "max iterations" in reason

    def test_no_gaps_remaining(self) -> None:
        current = GapAnalysis(gaps=[], suggested_queries=[])
        converged, reason = check_convergence(current, None, iteration=0)
        assert converged is True
        assert "no gaps remaining" in reason

    def test_no_new_queries(self) -> None:
        current = GapAnalysis(gaps=["g"], suggested_queries=[])
        converged, reason = check_convergence(current, None, iteration=0)
        assert converged is True
        assert "no new queries" in reason

    def test_gap_count_not_decreasing(self) -> None:
        previous = GapAnalysis(gaps=["g1", "g2"], suggested_queries=["q1"])
        current = GapAnalysis(gaps=["g1", "g2", "g3"], suggested_queries=["q1"])
        converged, reason = check_convergence(current, previous, iteration=1)
        assert converged is True
        assert "not decreasing" in reason

    def test_gap_count_equal_converges(self) -> None:
        previous = GapAnalysis(gaps=["g1", "g2"], suggested_queries=["q"])
        current = GapAnalysis(gaps=["g3", "g4"], suggested_queries=["q2"])
        converged, reason = check_convergence(current, previous, iteration=1)
        assert converged is True
        assert "not decreasing" in reason

    def test_gap_count_decreasing_not_converged(self) -> None:
        previous = GapAnalysis(gaps=["g1", "g2", "g3"], suggested_queries=["q"])
        current = GapAnalysis(gaps=["g4"], suggested_queries=["q2"])
        converged, reason = check_convergence(current, previous, iteration=1)
        assert converged is False
        assert reason == ""

    def test_first_iteration_with_gaps_not_converged(self) -> None:
        current = GapAnalysis(gaps=["g1"], suggested_queries=["q1"])
        converged, reason = check_convergence(current, None, iteration=0)
        assert converged is False
        assert reason == ""

    def test_custom_max_iterations(self) -> None:
        current = GapAnalysis(gaps=["g"], suggested_queries=["q"])
        converged, reason = check_convergence(
            current, None, iteration=5, max_iterations=5
        )
        assert converged is True
        assert "5" in reason

    def test_default_max_iterations(self) -> None:
        assert MAX_ITERATIONS == 3


# ---------------------------------------------------------------------------
# save_ter_state() / load_ter_state()
# ---------------------------------------------------------------------------


class TestTERStatePersistence:
    """Tests for state save/load roundtrip."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        result = TERResult(
            total_iterations=2,
            converged=True,
            convergence_reason="no gaps remaining",
            total_new_papers=5,
        )
        result.iterations.append(
            TERIteration(
                iteration=0,
                gaps_found=["gap A", "gap B"],
                queries_generated=["query 1"],
                new_papers_found=3,
                new_papers_relevant=2,
                converged=False,
            )
        )
        result.iterations.append(
            TERIteration(
                iteration=1,
                gaps_found=[],
                queries_generated=[],
                new_papers_found=0,
                new_papers_relevant=0,
                converged=True,
            )
        )

        save_ter_state(tmp_path, result)
        loaded = load_ter_state(tmp_path)

        assert loaded is not None
        assert loaded.total_iterations == 2
        assert loaded.converged is True
        assert loaded.convergence_reason == "no gaps remaining"
        assert loaded.total_new_papers == 5
        assert len(loaded.iterations) == 2
        assert loaded.iterations[0].gaps_found == ["gap A", "gap B"]
        assert loaded.iterations[0].queries_generated == ["query 1"]
        assert loaded.iterations[1].converged is True

    def test_creates_ter_loop_directory(self, tmp_path: Path) -> None:
        result = TERResult()
        save_ter_state(tmp_path, result)
        assert (tmp_path / "ter_loop").is_dir()
        assert (tmp_path / "ter_loop" / "ter_state.json").exists()

    def test_saves_iteration_files(self, tmp_path: Path) -> None:
        result = TERResult(total_iterations=1)
        iteration_data = {"gaps": ["g1"], "queries": ["q1"]}
        save_ter_state(tmp_path, result, iteration_data=iteration_data)

        iter_path = tmp_path / "ter_loop" / "iteration_1.json"
        assert iter_path.exists()
        data = json.loads(iter_path.read_text(encoding="utf-8"))
        assert data["gaps"] == ["g1"]

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        assert load_ter_state(tmp_path) is None

    def test_overwrite_state(self, tmp_path: Path) -> None:
        r1 = TERResult(total_iterations=1, converged=False)
        save_ter_state(tmp_path, r1)

        r2 = TERResult(total_iterations=2, converged=True, convergence_reason="done")
        save_ter_state(tmp_path, r2)

        loaded = load_ter_state(tmp_path)
        assert loaded is not None
        assert loaded.total_iterations == 2
        assert loaded.converged is True


# ---------------------------------------------------------------------------
# Integration: loop pattern
# ---------------------------------------------------------------------------


class TestTERLoopIntegration:
    """Test identify_gaps + check_convergence in loop pattern."""

    def test_loop_converges_on_empty_synthesis(self) -> None:
        """Empty synthesis → no gaps → converges immediately."""
        previous = None
        for i in range(MAX_ITERATIONS):
            ga = identify_gaps("", "topic", [])
            converged, reason = check_convergence(ga, previous, i)
            if converged:
                assert reason == "no gaps remaining"
                return
            previous = ga
        pytest.fail("Loop should have converged on empty synthesis")

    def test_loop_converges_when_gaps_stable(self) -> None:
        """Gaps that don't decrease → convergence on stall."""
        synthesis = (
            "- The relationship between X and Y remains unclear for all experiments\n"
            "- Future work should explore alternative methods for Z optimization\n"
        )
        previous = None
        converged_at = None
        for i in range(MAX_ITERATIONS):
            ga = identify_gaps(synthesis, "topic", [])
            converged, reason = check_convergence(ga, previous, i)
            if converged:
                converged_at = i
                break
            previous = ga
        # Should converge on second iteration (gaps same)
        assert converged_at is not None
        assert converged_at <= 2

    def test_loop_respects_max_iterations(self) -> None:
        """Loop stops at max_iterations even if gaps decrease."""
        iterations_run = 0
        previous = None
        for i in range(10):
            # Simulate decreasing gaps
            gap_count = max(1, 5 - i)
            gaps = [f"gap {j}" for j in range(gap_count)]
            queries = [f"query {j}" for j in range(gap_count)]
            ga = GapAnalysis(gaps=gaps, suggested_queries=queries)
            converged, reason = check_convergence(ga, previous, i, max_iterations=3)
            iterations_run = i + 1
            if converged:
                break
            previous = ga
        assert iterations_run <= 4  # At most max_iterations + 1 checks
