"""Tests for human-in-the-loop approval gates."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from research_pipeline.pipeline.gates import (
    DEFAULT_GATE_STAGES,
    AutoApproveGate,
    CallbackGate,
    CliGate,
    GateCallback,
    GateContext,
    GateDecision,
    GateRejectedError,
    GateResult,
    build_stage_summary,
    check_gate,
)


# ---------------------------------------------------------------------------
# GateDecision enum
# ---------------------------------------------------------------------------
class TestGateDecision:
    """Test GateDecision enum values."""

    def test_values(self) -> None:
        assert GateDecision.APPROVE == "approve"
        assert GateDecision.REJECT == "reject"
        assert GateDecision.SKIP == "skip"

    def test_from_string(self) -> None:
        assert GateDecision("approve") is GateDecision.APPROVE
        assert GateDecision("reject") is GateDecision.REJECT
        assert GateDecision("skip") is GateDecision.SKIP


# ---------------------------------------------------------------------------
# GateContext dataclass
# ---------------------------------------------------------------------------
class TestGateContext:
    """Test GateContext construction and fields."""

    def test_minimal(self, tmp_path: Path) -> None:
        ctx = GateContext(
            completed_stage="screen",
            next_stage="download",
            run_id="test-run",
            run_root=tmp_path,
        )
        assert ctx.completed_stage == "screen"
        assert ctx.next_stage == "download"
        assert ctx.run_id == "test-run"
        assert ctx.stage_summary == ""
        assert ctx.artifact_counts == {}

    def test_full(self, tmp_path: Path) -> None:
        ctx = GateContext(
            completed_stage="screen",
            next_stage="download",
            run_id="test-run",
            run_root=tmp_path,
            stage_summary="Shortlisted 10 papers",
            artifact_counts={"shortlisted": 10},
        )
        assert ctx.stage_summary == "Shortlisted 10 papers"
        assert ctx.artifact_counts == {"shortlisted": 10}

    def test_frozen(self, tmp_path: Path) -> None:
        ctx = GateContext(
            completed_stage="screen",
            next_stage="download",
            run_id="run1",
            run_root=tmp_path,
        )
        with pytest.raises(AttributeError):
            ctx.completed_stage = "plan"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GateResult dataclass
# ---------------------------------------------------------------------------
class TestGateResult:
    """Test GateResult construction."""

    def test_defaults(self) -> None:
        result = GateResult(decision=GateDecision.APPROVE)
        assert result.decision is GateDecision.APPROVE
        assert result.reason == ""
        assert result.reviewer == "unknown"

    def test_full(self) -> None:
        result = GateResult(
            decision=GateDecision.REJECT,
            reason="bad results",
            reviewer="alice",
        )
        assert result.decision is GateDecision.REJECT
        assert result.reason == "bad results"
        assert result.reviewer == "alice"

    def test_frozen(self) -> None:
        result = GateResult(decision=GateDecision.APPROVE)
        with pytest.raises(AttributeError):
            result.decision = GateDecision.REJECT  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GateRejectedError
# ---------------------------------------------------------------------------
class TestGateRejectedError:
    """Test GateRejectedError exception."""

    def test_without_reason(self) -> None:
        exc = GateRejectedError("screen")
        assert exc.stage == "screen"
        assert exc.reason == ""
        assert "screen" in str(exc)

    def test_with_reason(self) -> None:
        exc = GateRejectedError("screen", "too many irrelevant papers")
        assert "too many irrelevant papers" in str(exc)

    def test_is_exception(self) -> None:
        with pytest.raises(GateRejectedError):
            raise GateRejectedError("download", "test")


# ---------------------------------------------------------------------------
# AutoApproveGate
# ---------------------------------------------------------------------------
class TestAutoApproveGate:
    """Test AutoApproveGate always approves."""

    def test_approves(self, tmp_path: Path) -> None:
        gate = AutoApproveGate()
        ctx = GateContext(
            completed_stage="screen",
            next_stage="download",
            run_id="run1",
            run_root=tmp_path,
        )
        result = gate.check(ctx)
        assert result.decision is GateDecision.APPROVE
        assert result.reviewer == "auto"

    def test_approves_all_stages(self, tmp_path: Path) -> None:
        gate = AutoApproveGate()
        for stage in ["plan", "search", "screen", "download", "convert"]:
            ctx = GateContext(
                completed_stage=stage,
                next_stage="next",
                run_id="run1",
                run_root=tmp_path,
            )
            result = gate.check(ctx)
            assert result.decision is GateDecision.APPROVE

    def test_implements_protocol(self) -> None:
        assert isinstance(AutoApproveGate(), GateCallback)


# ---------------------------------------------------------------------------
# CliGate
# ---------------------------------------------------------------------------
class TestCliGate:
    """Test interactive CLI gate."""

    def _make_gate(self, user_input: str) -> CliGate:
        """Create a CliGate with simulated input."""
        return CliGate(input_stream=io.StringIO(user_input + "\n"))

    def _make_context(self, tmp_path: Path) -> GateContext:
        return GateContext(
            completed_stage="screen",
            next_stage="download",
            run_id="run1",
            run_root=tmp_path,
            stage_summary="Shortlisted 10 papers",
            artifact_counts={"shortlisted": 10},
        )

    def test_approve_a(self, tmp_path: Path) -> None:
        gate = self._make_gate("a")
        result = gate.check(self._make_context(tmp_path))
        assert result.decision is GateDecision.APPROVE

    def test_approve_yes(self, tmp_path: Path) -> None:
        gate = self._make_gate("yes")
        result = gate.check(self._make_context(tmp_path))
        assert result.decision is GateDecision.APPROVE

    def test_approve_empty(self, tmp_path: Path) -> None:
        gate = self._make_gate("")
        result = gate.check(self._make_context(tmp_path))
        assert result.decision is GateDecision.APPROVE

    def test_reject_r(self, tmp_path: Path) -> None:
        gate = self._make_gate("r")
        result = gate.check(self._make_context(tmp_path))
        assert result.decision is GateDecision.REJECT

    def test_reject_no(self, tmp_path: Path) -> None:
        gate = self._make_gate("no")
        result = gate.check(self._make_context(tmp_path))
        assert result.decision is GateDecision.REJECT

    def test_skip_s(self, tmp_path: Path) -> None:
        gate = self._make_gate("s")
        result = gate.check(self._make_context(tmp_path))
        assert result.decision is GateDecision.SKIP

    def test_skip_word(self, tmp_path: Path) -> None:
        gate = self._make_gate("skip")
        result = gate.check(self._make_context(tmp_path))
        assert result.decision is GateDecision.SKIP

    def test_invalid_then_approve(self, tmp_path: Path) -> None:
        gate = CliGate(input_stream=io.StringIO("invalid\na\n"))
        result = gate.check(self._make_context(tmp_path))
        assert result.decision is GateDecision.APPROVE

    def test_eof_rejects(self, tmp_path: Path) -> None:
        gate = CliGate(input_stream=io.StringIO(""))
        result = gate.check(self._make_context(tmp_path))
        assert result.decision is GateDecision.REJECT

    def test_format_summary_includes_stage(self, tmp_path: Path) -> None:
        gate = CliGate()
        ctx = self._make_context(tmp_path)
        text = gate._format_summary(ctx)
        assert "screen" in text
        assert "download" in text
        assert "Shortlisted 10 papers" in text

    def test_format_summary_final_stage(self, tmp_path: Path) -> None:
        gate = CliGate()
        ctx = GateContext(
            completed_stage="summarize",
            next_stage=None,
            run_id="run1",
            run_root=tmp_path,
        )
        text = gate._format_summary(ctx)
        assert "final stage" in text

    def test_implements_protocol(self) -> None:
        assert isinstance(CliGate(), GateCallback)


# ---------------------------------------------------------------------------
# CallbackGate
# ---------------------------------------------------------------------------
class TestCallbackGate:
    """Test callback-based gate."""

    def test_delegates(self, tmp_path: Path) -> None:
        mock_fn = MagicMock(
            return_value=GateResult(
                decision=GateDecision.APPROVE,
                reviewer="mcp",
            )
        )
        gate = CallbackGate(mock_fn)
        ctx = GateContext(
            completed_stage="screen",
            next_stage="download",
            run_id="run1",
            run_root=tmp_path,
        )
        result = gate.check(ctx)
        assert result.decision is GateDecision.APPROVE
        assert result.reviewer == "mcp"
        mock_fn.assert_called_once_with(ctx)

    def test_reject_from_callback(self, tmp_path: Path) -> None:
        mock_fn = MagicMock(
            return_value=GateResult(
                decision=GateDecision.REJECT,
                reason="not good enough",
            )
        )
        gate = CallbackGate(mock_fn)
        ctx = GateContext(
            completed_stage="screen",
            next_stage="download",
            run_id="run1",
            run_root=tmp_path,
        )
        result = gate.check(ctx)
        assert result.decision is GateDecision.REJECT


# ---------------------------------------------------------------------------
# build_stage_summary
# ---------------------------------------------------------------------------
class TestBuildStageSummary:
    """Test stage summary builder."""

    def test_plan_summary(self, tmp_path: Path) -> None:
        plan_dir = tmp_path / "plan"
        plan_dir.mkdir()
        plan_data = {"query_variants": ["q1", "q2", "q3"]}
        (plan_dir / "query_plan.json").write_text(
            json.dumps(plan_data), encoding="utf-8"
        )
        summary, counts = build_stage_summary("plan", tmp_path)
        assert "3 query variants" in summary
        assert counts["query_variants"] == 3

    def test_search_summary(self, tmp_path: Path) -> None:
        search_dir = tmp_path / "search"
        search_dir.mkdir()
        (search_dir / "candidates.jsonl").write_text(
            "line1\nline2\nline3\n", encoding="utf-8"
        )
        summary, counts = build_stage_summary("search", tmp_path)
        assert "3" in summary
        assert counts["candidates"] == 3

    def test_screen_summary(self, tmp_path: Path) -> None:
        screen_dir = tmp_path / "screen"
        screen_dir.mkdir()
        (screen_dir / "shortlist.jsonl").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        summary, counts = build_stage_summary("screen", tmp_path)
        assert "5" in summary
        assert counts["shortlisted"] == 5

    def test_download_summary(self, tmp_path: Path) -> None:
        pdf_dir = tmp_path / "download" / "pdf"
        pdf_dir.mkdir(parents=True)
        for i in range(3):
            (pdf_dir / f"paper{i}.pdf").write_bytes(b"dummy")
        summary, counts = build_stage_summary("download", tmp_path)
        assert "3" in summary
        assert counts["pdfs"] == 3

    def test_convert_summary(self, tmp_path: Path) -> None:
        md_dir = tmp_path / "convert" / "markdown"
        md_dir.mkdir(parents=True)
        (md_dir / "paper.md").write_text("# Paper", encoding="utf-8")
        summary, counts = build_stage_summary("convert", tmp_path)
        assert "1" in summary

    def test_extract_summary(self, tmp_path: Path) -> None:
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        (extract_dir / "chunks.json").write_text("{}", encoding="utf-8")
        summary, counts = build_stage_summary("extract", tmp_path)
        assert "1" in summary

    def test_summarize_summary(self, tmp_path: Path) -> None:
        summ_dir = tmp_path / "summarize"
        summ_dir.mkdir()
        (summ_dir / "synthesis.json").write_text("{}", encoding="utf-8")
        summary, counts = build_stage_summary("summarize", tmp_path)
        assert "Synthesis" in summary

    def test_unknown_stage(self, tmp_path: Path) -> None:
        summary, counts = build_stage_summary("unknown", tmp_path)
        assert "unknown" in summary
        assert counts == {}

    def test_missing_plan_file(self, tmp_path: Path) -> None:
        summary, counts = build_stage_summary("plan", tmp_path)
        assert "plan" in summary.lower()

    def test_corrupt_plan_json(self, tmp_path: Path) -> None:
        plan_dir = tmp_path / "plan"
        plan_dir.mkdir()
        (plan_dir / "query_plan.json").write_text("not json", encoding="utf-8")
        summary, counts = build_stage_summary("plan", tmp_path)
        assert "Plan generated" in summary


# ---------------------------------------------------------------------------
# check_gate
# ---------------------------------------------------------------------------
class TestCheckGate:
    """Test the check_gate orchestration function."""

    def test_skips_non_gate_stages(self, tmp_path: Path) -> None:
        gate = AutoApproveGate()
        result = check_gate(
            gate=gate,
            completed_stage="plan",
            next_stage="search",
            run_id="run1",
            run_root=tmp_path,
            gate_stages=["screen"],
        )
        assert result is GateDecision.APPROVE

    def test_checks_gate_stage(self, tmp_path: Path) -> None:
        gate = AutoApproveGate()
        result = check_gate(
            gate=gate,
            completed_stage="screen",
            next_stage="download",
            run_id="run1",
            run_root=tmp_path,
            gate_stages=["screen"],
        )
        assert result is GateDecision.APPROVE

    def test_raises_on_reject(self, tmp_path: Path) -> None:
        mock_gate = MagicMock()
        mock_gate.check.return_value = GateResult(
            decision=GateDecision.REJECT,
            reason="bad quality",
        )
        with pytest.raises(GateRejectedError, match="screen"):
            check_gate(
                gate=mock_gate,
                completed_stage="screen",
                next_stage="download",
                run_id="run1",
                run_root=tmp_path,
                gate_stages=["screen"],
            )

    def test_uses_defaults_when_none(self, tmp_path: Path) -> None:
        gate = AutoApproveGate()
        # "screen" is in default gates
        result = check_gate(
            gate=gate,
            completed_stage="screen",
            next_stage="download",
            run_id="run1",
            run_root=tmp_path,
            gate_stages=None,
        )
        assert result is GateDecision.APPROVE

    def test_skip_decision(self, tmp_path: Path) -> None:
        mock_gate = MagicMock()
        mock_gate.check.return_value = GateResult(
            decision=GateDecision.SKIP,
            reason="skip remaining",
        )
        result = check_gate(
            gate=mock_gate,
            completed_stage="screen",
            next_stage="download",
            run_id="run1",
            run_root=tmp_path,
            gate_stages=["screen"],
        )
        assert result is GateDecision.SKIP

    def test_builds_context_correctly(self, tmp_path: Path) -> None:
        captured = {}

        def capture_gate(ctx: GateContext) -> GateResult:
            captured["ctx"] = ctx
            return GateResult(decision=GateDecision.APPROVE)

        gate = CallbackGate(capture_gate)
        screen_dir = tmp_path / "screen"
        screen_dir.mkdir()
        (screen_dir / "shortlist.jsonl").write_text("a\nb\n", encoding="utf-8")

        check_gate(
            gate=gate,
            completed_stage="screen",
            next_stage="download",
            run_id="test-run",
            run_root=tmp_path,
            gate_stages=["screen"],
        )
        ctx = captured["ctx"]
        assert ctx.completed_stage == "screen"
        assert ctx.next_stage == "download"
        assert ctx.run_id == "test-run"
        assert ctx.artifact_counts.get("shortlisted") == 2


# ---------------------------------------------------------------------------
# DEFAULT_GATE_STAGES
# ---------------------------------------------------------------------------
class TestDefaultGateStages:
    """Test default gate configuration."""

    def test_default_stages(self) -> None:
        assert "screen" in DEFAULT_GATE_STAGES
        assert "download" in DEFAULT_GATE_STAGES
        assert "summarize" in DEFAULT_GATE_STAGES

    def test_not_every_stage(self) -> None:
        # Gates shouldn't fire on every stage by default
        assert "plan" not in DEFAULT_GATE_STAGES
        assert "extract" not in DEFAULT_GATE_STAGES
