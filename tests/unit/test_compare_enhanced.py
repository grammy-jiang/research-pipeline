"""Tests for enhanced compare_runs metrics.

Covers Jaccard similarity, quality score summary, query plan diff,
source distribution diff, and end-to-end compare_runs integration.
"""

import json
from pathlib import Path

import pytest

from research_pipeline.cli.cmd_compare import (
    _compute_jaccard,
    _diff_query_plans,
    _diff_source_distributions,
    _load_query_plan,
    _quality_score_summary,
    compare_runs,
)

# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------


class TestComputeJaccard:
    """Tests for _compute_jaccard."""

    def test_full_overlap(self) -> None:
        """Identical sets yield Jaccard = 1.0."""
        ids = {"a", "b", "c"}
        assert _compute_jaccard(ids, ids) == 1.0

    def test_no_overlap(self) -> None:
        """Disjoint sets yield Jaccard = 0.0."""
        assert _compute_jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap returns correct ratio."""
        # intersection = {b}, union = {a, b, c} => 1/3
        result = _compute_jaccard({"a", "b"}, {"b", "c"})
        assert result == pytest.approx(1.0 / 3.0)

    def test_both_empty(self) -> None:
        """Two empty sets return 0.0 (convention)."""
        assert _compute_jaccard(set(), set()) == 0.0

    def test_one_empty(self) -> None:
        """One empty set always yields 0.0."""
        assert _compute_jaccard(set(), {"a"}) == 0.0
        assert _compute_jaccard({"a"}, set()) == 0.0


# ---------------------------------------------------------------------------
# Quality score summary
# ---------------------------------------------------------------------------


class TestQualityScoreSummary:
    """Tests for _quality_score_summary."""

    def test_all_improved(self) -> None:
        """All common papers improved their score."""
        scores_a = {"p1": 0.5, "p2": 0.3}
        scores_b = {"p1": 0.8, "p2": 0.6}
        result = _quality_score_summary(scores_a, scores_b, {"p1", "p2"})
        assert result["improved_count"] == 2.0
        assert result["declined_count"] == 0.0
        assert result["unchanged_count"] == 0.0
        assert result["mean_delta"] == pytest.approx(0.3)

    def test_mixed_changes(self) -> None:
        """Mix of improved, declined, and unchanged scores."""
        scores_a = {"p1": 0.5, "p2": 0.8, "p3": 0.4}
        scores_b = {"p1": 0.7, "p2": 0.6, "p3": 0.4}
        result = _quality_score_summary(scores_a, scores_b, {"p1", "p2", "p3"})
        assert result["improved_count"] == 1.0
        assert result["declined_count"] == 1.0
        assert result["unchanged_count"] == 1.0

    def test_no_common_ids(self) -> None:
        """No common papers yields zero defaults."""
        result = _quality_score_summary({"p1": 0.5}, {"p2": 0.8}, set())
        assert result["mean_delta"] == 0.0
        assert result["median_delta"] == 0.0
        assert result["std_delta"] == 0.0

    def test_missing_scores_skipped(self) -> None:
        """Papers in common_ids but missing from a score dict are skipped."""
        scores_a = {"p1": 0.5}
        scores_b: dict[str, float] = {}
        result = _quality_score_summary(scores_a, scores_b, {"p1"})
        assert result["mean_delta"] == 0.0
        assert result["improved_count"] == 0.0

    def test_single_paper_std_zero(self) -> None:
        """Single paper yields std_delta = 0.0 (not enough data)."""
        result = _quality_score_summary({"p1": 0.3}, {"p1": 0.7}, {"p1"})
        assert result["std_delta"] == 0.0
        assert result["mean_delta"] == pytest.approx(0.4)
        assert result["median_delta"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Query plan diff / topic drift
# ---------------------------------------------------------------------------


class TestDiffQueryPlans:
    """Tests for _diff_query_plans."""

    def test_added_and_removed_terms(self) -> None:
        """Detects added and removed must/nice terms."""
        plan_a = {
            "topic_raw": "LLM memory",
            "must_terms": ["memory", "retrieval"],
            "nice_terms": ["agent", "context"],
        }
        plan_b = {
            "topic_raw": "LLM long-term memory",
            "must_terms": ["memory", "long-term"],
            "nice_terms": ["context", "episodic"],
        }
        result = _diff_query_plans(plan_a, plan_b)
        assert result["topic_a"] == "LLM memory"
        assert result["topic_b"] == "LLM long-term memory"
        assert result["added_must_terms"] == ["long-term"]
        assert result["removed_must_terms"] == ["retrieval"]
        assert result["added_nice_terms"] == ["episodic"]
        assert result["removed_nice_terms"] == ["agent"]

    def test_identical_plans(self) -> None:
        """Identical plans yield no drift."""
        plan = {
            "topic_raw": "transformers",
            "must_terms": ["attention"],
            "nice_terms": ["efficiency"],
        }
        result = _diff_query_plans(plan, plan)
        assert result["added_must_terms"] == []
        assert result["removed_must_terms"] == []
        assert result["added_nice_terms"] == []
        assert result["removed_nice_terms"] == []

    def test_empty_plans(self) -> None:
        """Empty plans produce empty diffs."""
        result = _diff_query_plans({}, {})
        assert result["topic_a"] == ""
        assert result["topic_b"] == ""
        assert result["added_must_terms"] == []
        assert result["removed_must_terms"] == []

    def test_falls_back_to_normalized_topic(self) -> None:
        """Uses topic_normalized when topic_raw is absent."""
        plan_a = {"topic_normalized": "normalized A"}
        plan_b = {"topic_normalized": "normalized B"}
        result = _diff_query_plans(plan_a, plan_b)
        assert result["topic_a"] == "normalized A"
        assert result["topic_b"] == "normalized B"

    def test_none_terms_treated_as_empty(self) -> None:
        """None values for term lists are treated as empty."""
        plan_a: dict[str, object] = {"must_terms": None, "nice_terms": ["x"]}
        plan_b: dict[str, object] = {"must_terms": ["y"], "nice_terms": None}
        result = _diff_query_plans(plan_a, plan_b)
        assert result["added_must_terms"] == ["y"]
        assert result["removed_nice_terms"] == ["x"]


# ---------------------------------------------------------------------------
# Load query plan from disk
# ---------------------------------------------------------------------------


class TestLoadQueryPlan:
    """Tests for _load_query_plan."""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """Missing query_plan.json returns empty dict."""
        run_root = tmp_path / "run"
        run_root.mkdir()
        result = _load_query_plan(run_root)
        assert result == {}

    def test_loads_valid_plan(self, tmp_path: Path) -> None:
        """Valid JSON is loaded correctly."""
        plan_dir = tmp_path / "run" / "plan"
        plan_dir.mkdir(parents=True)
        plan_data = {"topic_raw": "test", "must_terms": ["a"]}
        (plan_dir / "query_plan.json").write_text(json.dumps(plan_data))
        result = _load_query_plan(tmp_path / "run")
        assert result["topic_raw"] == "test"

    def test_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        """Malformed JSON returns empty dict without raising."""
        plan_dir = tmp_path / "run" / "plan"
        plan_dir.mkdir(parents=True)
        (plan_dir / "query_plan.json").write_text("{bad json")
        result = _load_query_plan(tmp_path / "run")
        assert result == {}


# ---------------------------------------------------------------------------
# Source distribution diff
# ---------------------------------------------------------------------------


class TestDiffSourceDistributions:
    """Tests for _diff_source_distributions."""

    def test_added_and_removed_sources(self) -> None:
        """Detects sources added to run B and removed from run A."""
        dist_a = {"arxiv": 10, "scholar": 5}
        dist_b = {"arxiv": 8, "semantic_scholar": 3}
        result = _diff_source_distributions(dist_a, dist_b)
        assert result["added_sources"] == ["semantic_scholar"]
        assert result["removed_sources"] == ["scholar"]
        assert result["per_source"]["arxiv"] == {"run_a": 10, "run_b": 8}
        assert result["per_source"]["scholar"] == {"run_a": 5, "run_b": 0}

    def test_identical_distributions(self) -> None:
        """Same distributions yield no added/removed."""
        dist = {"arxiv": 10}
        result = _diff_source_distributions(dist, dict(dist))
        assert result["added_sources"] == []
        assert result["removed_sources"] == []

    def test_both_empty(self) -> None:
        """Two empty distributions yield empty result."""
        result = _diff_source_distributions({}, {})
        assert result["per_source"] == {}
        assert result["added_sources"] == []
        assert result["removed_sources"] == []


# ---------------------------------------------------------------------------
# End-to-end compare_runs integration
# ---------------------------------------------------------------------------


def _setup_run(
    root: Path,
    paper_ids: list[str],
    *,
    sources: list[str] | None = None,
    quality: dict[str, float] | None = None,
    query_plan: dict[str, object] | None = None,
) -> Path:
    """Create a minimal run directory for testing."""
    run_root = root
    run_root.mkdir(parents=True, exist_ok=True)

    # Screen shortlist
    screen_dir = run_root / "screen"
    screen_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, pid in enumerate(paper_ids):
        rec: dict[str, object] = {"arxiv_id": pid, "title": f"Paper {pid}"}
        if sources and i < len(sources):
            rec["source"] = sources[i]
        records.append(rec)
    (screen_dir / "shortlist.json").write_text(json.dumps(records))

    # Quality scores
    if quality:
        q_dir = run_root / "quality"
        q_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps({"paper_id": pid, "composite_score": sc})
            for pid, sc in quality.items()
        ]
        (q_dir / "quality_scores.jsonl").write_text("\n".join(lines) + "\n")

    # Query plan
    if query_plan:
        p_dir = run_root / "plan"
        p_dir.mkdir(parents=True, exist_ok=True)
        (p_dir / "query_plan.json").write_text(json.dumps(query_plan))

    return run_root


class TestCompareRunsIntegration:
    """Integration tests for the enhanced compare_runs."""

    def test_new_keys_present(self, tmp_path: Path) -> None:
        """compare_runs result contains all new metric keys."""
        run_a = _setup_run(tmp_path / "a", ["p1", "p2"])
        run_b = _setup_run(tmp_path / "b", ["p2", "p3"])
        result = compare_runs(run_a, run_b, "a", "b")

        assert "jaccard_similarity" in result
        assert "quality_summary" in result
        assert "topic_drift" in result
        assert "source_distribution" in result

    def test_jaccard_in_result(self, tmp_path: Path) -> None:
        """Jaccard similarity is correctly computed end-to-end."""
        run_a = _setup_run(tmp_path / "a", ["p1", "p2", "p3"])
        run_b = _setup_run(tmp_path / "b", ["p2", "p3", "p4"])
        result = compare_runs(run_a, run_b, "a", "b")
        # intersection = {p2, p3}, union = {p1, p2, p3, p4} => 2/4
        assert result["jaccard_similarity"] == pytest.approx(0.5)

    def test_quality_summary_in_result(self, tmp_path: Path) -> None:
        """Quality summary reflects score changes for common papers."""
        run_a = _setup_run(tmp_path / "a", ["p1", "p2"], quality={"p1": 0.3, "p2": 0.6})
        run_b = _setup_run(tmp_path / "b", ["p1", "p2"], quality={"p1": 0.5, "p2": 0.4})
        result = compare_runs(run_a, run_b, "a", "b")
        qs = result["quality_summary"]
        assert qs["improved_count"] == 1.0
        assert qs["declined_count"] == 1.0

    def test_topic_drift_in_result(self, tmp_path: Path) -> None:
        """Topic drift is populated from query plans."""
        run_a = _setup_run(
            tmp_path / "a",
            ["p1"],
            query_plan={"topic_raw": "AI safety", "must_terms": ["alignment"]},
        )
        run_b = _setup_run(
            tmp_path / "b",
            ["p1"],
            query_plan={
                "topic_raw": "AI safety and alignment",
                "must_terms": ["alignment", "RLHF"],
            },
        )
        result = compare_runs(run_a, run_b, "a", "b")
        td = result["topic_drift"]
        assert td["topic_a"] == "AI safety"
        assert td["added_must_terms"] == ["RLHF"]

    def test_source_distribution_in_result(self, tmp_path: Path) -> None:
        """Source distribution diff reflects different source mixes."""
        run_a = _setup_run(tmp_path / "a", ["p1", "p2"], sources=["arxiv", "scholar"])
        run_b = _setup_run(
            tmp_path / "b", ["p3", "p4"], sources=["arxiv", "semantic_scholar"]
        )
        result = compare_runs(run_a, run_b, "a", "b")
        sd = result["source_distribution"]
        assert "semantic_scholar" in sd["added_sources"]
        assert "scholar" in sd["removed_sources"]
