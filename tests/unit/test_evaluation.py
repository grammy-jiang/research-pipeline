"""Tests for schema-grounded evaluation module."""

from __future__ import annotations

import json
from pathlib import Path

from research_pipeline.evaluation.schema_eval import (
    STAGE_EVALUATORS,
    EvalCheck,
    EvalReport,
    check_field_populated,
    check_list_count_consistency,
    check_score_range,
    evaluate_plan,
    evaluate_run,
    evaluate_screen,
    evaluate_search,
    evaluate_stage,
    evaluate_summarize,
)


# ---------------------------------------------------------------------------
# EvalCheck tests
# ---------------------------------------------------------------------------
class TestEvalCheck:
    def test_basic_creation(self) -> None:
        c = EvalCheck(name="test", description="desc", passed=True)
        assert c.passed is True
        assert c.severity == "error"
        assert c.details == ""

    def test_with_details(self) -> None:
        c = EvalCheck(
            name="f", description="d", passed=False, details="bad", severity="warning"
        )
        assert c.passed is False
        assert c.severity == "warning"
        assert c.details == "bad"


# ---------------------------------------------------------------------------
# EvalReport tests
# ---------------------------------------------------------------------------
class TestEvalReport:
    def test_empty_report_passes(self) -> None:
        r = EvalReport(stage="test")
        assert r.passed is True
        assert r.error_count == 0
        assert r.warning_count == 0

    def test_all_pass(self) -> None:
        r = EvalReport(
            stage="x",
            checks=[
                EvalCheck("a", "a", True),
                EvalCheck("b", "b", True, severity="warning"),
            ],
        )
        assert r.passed is True
        assert r.error_count == 0
        assert r.warning_count == 0

    def test_error_fails_report(self) -> None:
        r = EvalReport(
            stage="x",
            checks=[
                EvalCheck("a", "a", False),
                EvalCheck("b", "b", True),
            ],
        )
        assert r.passed is False
        assert r.error_count == 1

    def test_warning_does_not_fail(self) -> None:
        r = EvalReport(
            stage="x",
            checks=[
                EvalCheck("a", "a", True),
                EvalCheck("b", "b", False, severity="warning"),
            ],
        )
        assert r.passed is True
        assert r.warning_count == 1

    def test_summary_pass(self) -> None:
        r = EvalReport(stage="plan", checks=[EvalCheck("a", "a", True)])
        s = r.summary()
        assert "PASS" in s
        assert "plan" in s

    def test_summary_fail(self) -> None:
        r = EvalReport(stage="search", checks=[EvalCheck("a", "a", False)])
        s = r.summary()
        assert "FAIL" in s


# ---------------------------------------------------------------------------
# check_field_populated tests
# ---------------------------------------------------------------------------
class TestCheckFieldPopulated:
    def test_none_value(self) -> None:
        c = check_field_populated({"x": None}, "x", "test")
        assert c.passed is False

    def test_missing_field(self) -> None:
        c = check_field_populated({}, "x", "test")
        assert c.passed is False

    def test_empty_string(self) -> None:
        c = check_field_populated({"x": "  "}, "x", "test")
        assert c.passed is False

    def test_empty_list_is_warning(self) -> None:
        c = check_field_populated({"x": []}, "x", "test")
        assert c.passed is False
        assert c.severity == "warning"

    def test_valid_string(self) -> None:
        c = check_field_populated({"x": "hello"}, "x", "test")
        assert c.passed is True

    def test_valid_list(self) -> None:
        c = check_field_populated({"x": [1, 2]}, "x", "test")
        assert c.passed is True

    def test_valid_int(self) -> None:
        c = check_field_populated({"x": 42}, "x", "test")
        assert c.passed is True


# ---------------------------------------------------------------------------
# check_score_range tests
# ---------------------------------------------------------------------------
class TestCheckScoreRange:
    def test_absent_field_passes_as_info(self) -> None:
        c = check_score_range({}, "score")
        assert c.passed is True
        assert c.severity == "info"

    def test_in_range(self) -> None:
        c = check_score_range({"score": 0.5}, "score")
        assert c.passed is True

    def test_at_boundaries(self) -> None:
        assert check_score_range({"score": 0.0}, "score").passed is True
        assert check_score_range({"score": 1.0}, "score").passed is True

    def test_below_range(self) -> None:
        c = check_score_range({"score": -0.1}, "score")
        assert c.passed is False

    def test_above_range(self) -> None:
        c = check_score_range({"score": 1.1}, "score")
        assert c.passed is False

    def test_custom_range(self) -> None:
        c = check_score_range({"score": 50}, "score", min_val=0, max_val=100)
        assert c.passed is True

    def test_not_numeric(self) -> None:
        c = check_score_range({"score": "abc"}, "score")
        assert c.passed is False

    def test_string_numeric(self) -> None:
        c = check_score_range({"score": "0.7"}, "score")
        assert c.passed is True


# ---------------------------------------------------------------------------
# check_list_count_consistency tests
# ---------------------------------------------------------------------------
class TestCheckListCountConsistency:
    def test_no_count_field(self) -> None:
        c = check_list_count_consistency({"items": [1, 2]}, "items", "count")
        assert c.passed is True
        assert c.severity == "info"

    def test_matching(self) -> None:
        c = check_list_count_consistency(
            {"items": [1, 2, 3], "count": 3}, "items", "count"
        )
        assert c.passed is True

    def test_mismatch(self) -> None:
        c = check_list_count_consistency(
            {"items": [1, 2], "count": 5}, "items", "count"
        )
        assert c.passed is False

    def test_not_a_list(self) -> None:
        c = check_list_count_consistency(
            {"items": "not_list", "count": 1}, "items", "count"
        )
        assert c.passed is False


# ---------------------------------------------------------------------------
# evaluate_plan tests
# ---------------------------------------------------------------------------
class TestEvaluatePlan:
    def test_missing_plan(self, tmp_path: Path) -> None:
        r = evaluate_plan(tmp_path)
        assert r.passed is False

    def test_invalid_json(self, tmp_path: Path) -> None:
        plan_dir = tmp_path / "plan"
        plan_dir.mkdir()
        (plan_dir / "query_plan.json").write_text("not json", encoding="utf-8")
        r = evaluate_plan(tmp_path)
        assert r.passed is False

    def test_valid_plan(self, tmp_path: Path) -> None:
        plan_dir = tmp_path / "plan"
        plan_dir.mkdir()
        data = {
            "topic_raw": "AI memory",
            "must_terms": ["memory", "agent"],
            "query_variants": ["AI agent memory", "LLM memory systems"],
        }
        (plan_dir / "query_plan.json").write_text(json.dumps(data), encoding="utf-8")
        r = evaluate_plan(tmp_path)
        assert r.passed is True

    def test_empty_variants(self, tmp_path: Path) -> None:
        plan_dir = tmp_path / "plan"
        plan_dir.mkdir()
        data = {
            "topic_raw": "AI",
            "must_terms": ["ai"],
            "query_variants": [],
        }
        (plan_dir / "query_plan.json").write_text(json.dumps(data), encoding="utf-8")
        r = evaluate_plan(tmp_path)
        # Empty query_variants → warning (list) + fail (count < 1)
        assert any(not c.passed for c in r.checks)

    def test_missing_must_terms(self, tmp_path: Path) -> None:
        plan_dir = tmp_path / "plan"
        plan_dir.mkdir()
        data = {"topic_raw": "AI", "query_variants": ["AI"]}
        (plan_dir / "query_plan.json").write_text(json.dumps(data), encoding="utf-8")
        r = evaluate_plan(tmp_path)
        assert any(c.name == "must_terms_present" and not c.passed for c in r.checks)


# ---------------------------------------------------------------------------
# evaluate_search tests
# ---------------------------------------------------------------------------
class TestEvaluateSearch:
    def test_missing_candidates(self, tmp_path: Path) -> None:
        r = evaluate_search(tmp_path)
        assert r.passed is False

    def test_valid_candidates(self, tmp_path: Path) -> None:
        search_dir = tmp_path / "search"
        search_dir.mkdir()
        entries = [
            {"arxiv_id": "2301.00001", "title": "Paper 1"},
            {"arxiv_id": "2301.00002", "title": "Paper 2"},
        ]
        (search_dir / "candidates.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries), encoding="utf-8"
        )
        r = evaluate_search(tmp_path)
        assert r.passed is True

    def test_empty_candidates(self, tmp_path: Path) -> None:
        search_dir = tmp_path / "search"
        search_dir.mkdir()
        (search_dir / "candidates.jsonl").write_text("", encoding="utf-8")
        r = evaluate_search(tmp_path)
        assert any(not c.passed for c in r.checks)

    def test_invalid_json_candidate(self, tmp_path: Path) -> None:
        search_dir = tmp_path / "search"
        search_dir.mkdir()
        (search_dir / "candidates.jsonl").write_text("not json\n", encoding="utf-8")
        r = evaluate_search(tmp_path)
        assert any(not c.passed for c in r.checks)


# ---------------------------------------------------------------------------
# evaluate_screen tests
# ---------------------------------------------------------------------------
class TestEvaluateScreen:
    def test_missing_shortlist(self, tmp_path: Path) -> None:
        r = evaluate_screen(tmp_path)
        assert r.passed is False

    def test_valid_shortlist(self, tmp_path: Path) -> None:
        screen_dir = tmp_path / "screen"
        screen_dir.mkdir()
        entries = [
            {"arxiv_id": "2301.00001", "title": "P1", "blended_score": 0.8},
        ]
        (screen_dir / "shortlist.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries), encoding="utf-8"
        )
        r = evaluate_screen(tmp_path)
        assert r.passed is True

    def test_invalid_score(self, tmp_path: Path) -> None:
        screen_dir = tmp_path / "screen"
        screen_dir.mkdir()
        entries = [
            {"arxiv_id": "x", "title": "P1", "blended_score": 1.5},
        ]
        (screen_dir / "shortlist.jsonl").write_text(
            json.dumps(entries[0]), encoding="utf-8"
        )
        r = evaluate_screen(tmp_path)
        assert any(not c.passed for c in r.checks)


# ---------------------------------------------------------------------------
# evaluate_summarize tests
# ---------------------------------------------------------------------------
class TestEvaluateSummarize:
    def test_missing_synthesis(self, tmp_path: Path) -> None:
        r = evaluate_summarize(tmp_path)
        assert any(not c.passed for c in r.checks)

    def test_valid_synthesis(self, tmp_path: Path) -> None:
        sum_dir = tmp_path / "summarize"
        sum_dir.mkdir()
        data = {
            "topic": "AI",
            "paper_summaries": [{"id": "1"}],
            "paper_count": 1,
        }
        (sum_dir / "synthesis.json").write_text(json.dumps(data), encoding="utf-8")
        (sum_dir / "synthesis.md").write_text("# Report", encoding="utf-8")
        r = evaluate_summarize(tmp_path)
        assert r.passed is True

    def test_count_mismatch(self, tmp_path: Path) -> None:
        sum_dir = tmp_path / "summarize"
        sum_dir.mkdir()
        data = {
            "topic": "AI",
            "paper_summaries": [{"id": "1"}],
            "paper_count": 5,
        }
        (sum_dir / "synthesis.json").write_text(json.dumps(data), encoding="utf-8")
        (sum_dir / "synthesis.md").write_text("# Report", encoding="utf-8")
        r = evaluate_summarize(tmp_path)
        assert any(not c.passed for c in r.checks)

    def test_invalid_json(self, tmp_path: Path) -> None:
        sum_dir = tmp_path / "summarize"
        sum_dir.mkdir()
        (sum_dir / "synthesis.json").write_text("not json", encoding="utf-8")
        (sum_dir / "synthesis.md").write_text("# Report", encoding="utf-8")
        r = evaluate_summarize(tmp_path)
        assert any(not c.passed for c in r.checks)


# ---------------------------------------------------------------------------
# evaluate_stage / evaluate_run tests
# ---------------------------------------------------------------------------
class TestEvaluateStage:
    def test_unknown_stage(self, tmp_path: Path) -> None:
        r = evaluate_stage(tmp_path, "nonexistent")
        # Should return a warning, not fail
        assert all(c.severity == "warning" for c in r.checks if not c.passed)

    def test_known_stages(self) -> None:
        assert "plan" in STAGE_EVALUATORS
        assert "search" in STAGE_EVALUATORS
        assert "screen" in STAGE_EVALUATORS
        assert "summarize" in STAGE_EVALUATORS


class TestEvaluateRun:
    def test_empty_run(self, tmp_path: Path) -> None:
        reports = evaluate_run(tmp_path)
        assert len(reports) == 4  # default stages
        assert all(isinstance(r, EvalReport) for r in reports)

    def test_custom_stages(self, tmp_path: Path) -> None:
        reports = evaluate_run(tmp_path, stages=["plan"])
        assert len(reports) == 1
        assert reports[0].stage == "plan"

    def test_full_valid_run(self, tmp_path: Path) -> None:
        # Plan
        plan_dir = tmp_path / "plan"
        plan_dir.mkdir()
        (plan_dir / "query_plan.json").write_text(
            json.dumps(
                {
                    "topic_raw": "AI",
                    "must_terms": ["ai"],
                    "query_variants": ["AI agents"],
                }
            ),
            encoding="utf-8",
        )
        # Search
        search_dir = tmp_path / "search"
        search_dir.mkdir()
        (search_dir / "candidates.jsonl").write_text(
            json.dumps({"arxiv_id": "2301.00001", "title": "P1"}),
            encoding="utf-8",
        )
        # Screen
        screen_dir = tmp_path / "screen"
        screen_dir.mkdir()
        (screen_dir / "shortlist.jsonl").write_text(
            json.dumps({"arxiv_id": "2301.00001", "title": "P1", "score": 0.9}),
            encoding="utf-8",
        )
        # Summarize
        sum_dir = tmp_path / "summarize"
        sum_dir.mkdir()
        (sum_dir / "synthesis.json").write_text(
            json.dumps(
                {"topic": "AI", "paper_summaries": [{"id": "1"}], "paper_count": 1}
            ),
            encoding="utf-8",
        )
        (sum_dir / "synthesis.md").write_text("# Report", encoding="utf-8")

        reports = evaluate_run(tmp_path)
        assert all(r.passed for r in reports)
