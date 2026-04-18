"""Tests for pipeline stage registry and validation."""

from __future__ import annotations

from research_pipeline.pipeline.stages import STAGE_ORDER, validate_stage_name


class TestStageOrder:
    """Tests for the STAGE_ORDER constant."""

    def test_has_seven_stages(self) -> None:
        assert len(STAGE_ORDER) == 7

    def test_order_is_correct(self) -> None:
        expected = [
            "plan",
            "search",
            "screen",
            "download",
            "convert",
            "extract",
            "summarize",
        ]
        assert expected == STAGE_ORDER

    def test_plan_is_first(self) -> None:
        assert STAGE_ORDER[0] == "plan"

    def test_summarize_is_last(self) -> None:
        assert STAGE_ORDER[-1] == "summarize"


class TestValidateStageName:
    """Tests for validate_stage_name()."""

    def test_valid_stages_return_true(self) -> None:
        for stage in STAGE_ORDER:
            assert validate_stage_name(stage) is True

    def test_invalid_stage_returns_false(self) -> None:
        assert validate_stage_name("nonexistent") is False

    def test_empty_string_returns_false(self) -> None:
        assert validate_stage_name("") is False

    def test_case_sensitive(self) -> None:
        assert validate_stage_name("Plan") is False
        assert validate_stage_name("SEARCH") is False

    def test_whitespace_returns_false(self) -> None:
        assert validate_stage_name(" plan") is False
        assert validate_stage_name("plan ") is False
