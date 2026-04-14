"""Tests for the compare command (cmd_compare)."""

import json
from pathlib import Path

from research_pipeline.cli.cmd_compare import (
    _diff_confidence,
    _diff_gaps,
    _diff_paper_sets,
    _diff_readiness,
    compare_runs,
)


def _setup_run(
    tmp_path: Path,
    name: str,
    paper_ids: list[str],
    synthesis: dict | None = None,
) -> Path:
    """Create a mock run directory with screen and synthesis data."""
    run_root = tmp_path / name
    screen_dir = run_root / "screen"
    screen_dir.mkdir(parents=True)

    # Write shortlist as JSON
    shortlist_path = screen_dir / "shortlist.json"
    records = []
    for pid in paper_ids:
        records.append({"arxiv_id": pid, "title": f"Paper {pid}"})
    shortlist_path.write_text(json.dumps(records))

    if synthesis:
        synth_dir = run_root / "synthesis"
        synth_dir.mkdir(parents=True)
        (synth_dir / "synthesis_results.json").write_text(json.dumps(synthesis))

    return run_root


class TestDiffPaperSets:
    def test_identical_sets(self) -> None:
        result = _diff_paper_sets({"a", "b"}, {"a", "b"})
        assert result["only_in_run_a"] == []
        assert result["only_in_run_b"] == []
        assert sorted(result["in_both"]) == ["a", "b"]

    def test_disjoint_sets(self) -> None:
        result = _diff_paper_sets({"a", "b"}, {"c", "d"})
        assert sorted(result["only_in_run_a"]) == ["a", "b"]
        assert sorted(result["only_in_run_b"]) == ["c", "d"]
        assert result["in_both"] == []

    def test_overlapping_sets(self) -> None:
        result = _diff_paper_sets({"a", "b", "c"}, {"b", "c", "d"})
        assert result["only_in_run_a"] == ["a"]
        assert result["only_in_run_b"] == ["d"]
        assert sorted(result["in_both"]) == ["b", "c"]


class TestDiffGaps:
    def test_resolved_gaps(self) -> None:
        synth_a = {"gaps": [{"description": "Gap A", "type": "ACADEMIC"}]}
        synth_b = {"gaps": []}
        result = _diff_gaps(synth_a, synth_b)
        assert len(result["resolved_gaps"]) == 1
        assert result["resolved_gaps"][0]["description"] == "Gap A"

    def test_new_gaps(self) -> None:
        synth_a = {"gaps": []}
        synth_b = {"gaps": [{"description": "Gap B", "type": "ENGINEERING"}]}
        result = _diff_gaps(synth_a, synth_b)
        assert len(result["new_gaps"]) == 1

    def test_persistent_gaps(self) -> None:
        synth_a = {"gaps": [{"description": "Gap X", "type": "ACADEMIC"}]}
        synth_b = {"gaps": [{"description": "Gap X", "type": "ACADEMIC"}]}
        result = _diff_gaps(synth_a, synth_b)
        assert len(result["persistent_gaps"]) == 1

    def test_none_synthesis(self) -> None:
        result = _diff_gaps(None, None)
        assert result["resolved_gaps"] == []
        assert result["new_gaps"] == []
        assert result["persistent_gaps"] == []


class TestDiffConfidence:
    def test_confidence_upgrade(self) -> None:
        synth_a = {
            "confidence_graded_findings": {
                "medium": [{"finding": "F1"}],
                "high": [],
                "low": [],
            }
        }
        synth_b = {
            "confidence_graded_findings": {
                "high": [{"finding": "F1"}],
                "medium": [],
                "low": [],
            }
        }
        changes = _diff_confidence(synth_a, synth_b)
        assert len(changes) == 1
        assert changes[0]["direction"] == "upgraded"

    def test_no_changes(self) -> None:
        synth = {
            "confidence_graded_findings": {
                "high": [{"finding": "F1"}],
                "medium": [],
                "low": [],
            }
        }
        changes = _diff_confidence(synth, synth)
        assert changes == []


class TestDiffReadiness:
    def test_readiness_change(self) -> None:
        synth_a = {"readiness": {"verdict": "HAS_GAPS"}}
        synth_b = {"readiness": {"verdict": "IMPLEMENTATION_READY"}}
        result = _diff_readiness(synth_a, synth_b)
        assert result["verdict_run_a"] == "HAS_GAPS"
        assert result["verdict_run_b"] == "IMPLEMENTATION_READY"

    def test_none_synthesis(self) -> None:
        result = _diff_readiness(None, None)
        assert result["verdict_run_a"] == "UNKNOWN"
        assert result["verdict_run_b"] == "UNKNOWN"


class TestCompareRuns:
    def test_basic_comparison(self, tmp_path: Path) -> None:
        synth_a = {
            "gaps": [{"description": "Gap 1", "type": "ACADEMIC"}],
            "readiness": {"verdict": "HAS_GAPS"},
        }
        synth_b = {
            "gaps": [],
            "readiness": {"verdict": "IMPLEMENTATION_READY"},
        }
        root_a = _setup_run(tmp_path, "run_a", ["p1", "p2", "p3"], synth_a)
        root_b = _setup_run(tmp_path, "run_b", ["p2", "p3", "p4"], synth_b)
        result = compare_runs(root_a, root_b, "run_a", "run_b")

        assert result["run_a"] == "run_a"
        assert result["run_b"] == "run_b"
        assert result["paper_diff"]["count_a"] == 3
        assert result["paper_diff"]["count_b"] == 3
        assert result["paper_diff"]["overlap"] == 2
        assert result["paper_diff"]["new_in_b"] == 1
        assert result["paper_diff"]["dropped_from_a"] == 1
        assert result["gap_analysis"]["resolved_count"] == 1
        assert result["readiness"]["verdict_run_a"] == "HAS_GAPS"
        assert result["readiness"]["verdict_run_b"] == "IMPLEMENTATION_READY"
        assert result["has_synthesis_a"] is True
        assert result["has_synthesis_b"] is True

    def test_comparison_without_synthesis(self, tmp_path: Path) -> None:
        root_a = _setup_run(tmp_path, "run_a", ["p1"])
        root_b = _setup_run(tmp_path, "run_b", ["p2"])
        result = compare_runs(root_a, root_b, "run_a", "run_b")
        assert result["has_synthesis_a"] is False
        assert result["has_synthesis_b"] is False
