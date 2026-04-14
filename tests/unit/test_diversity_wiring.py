"""Tests for diversity-aware shortlisting wiring into config, CLI, and orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from research_pipeline.config.defaults import DEFAULTS
from research_pipeline.config.models import PipelineConfig, ScreenConfig


class TestScreenConfigDiversityDefaults:
    """ScreenConfig should expose diversity fields with correct defaults."""

    def test_diversity_default_false(self) -> None:
        cfg = ScreenConfig()
        assert cfg.diversity is False

    def test_diversity_lambda_default(self) -> None:
        cfg = ScreenConfig()
        assert cfg.diversity_lambda == pytest.approx(0.3)

    def test_diversity_true_propagates(self) -> None:
        cfg = ScreenConfig(diversity=True)
        assert cfg.diversity is True

    def test_diversity_lambda_custom(self) -> None:
        cfg = ScreenConfig(diversity_lambda=0.7)
        assert cfg.diversity_lambda == pytest.approx(0.7)

    def test_pipeline_config_screen_diversity(self) -> None:
        """Diversity fields accessible from top-level PipelineConfig."""
        cfg = PipelineConfig(screen=ScreenConfig(diversity=True, diversity_lambda=0.5))
        assert cfg.screen.diversity is True
        assert cfg.screen.diversity_lambda == pytest.approx(0.5)

    def test_screen_config_roundtrip(self) -> None:
        """ScreenConfig with diversity serializes and deserializes correctly."""
        original = ScreenConfig(diversity=True, diversity_lambda=0.42)
        dumped = original.model_dump(mode="json")
        restored = ScreenConfig.model_validate(dumped)
        assert restored.diversity is True
        assert restored.diversity_lambda == pytest.approx(0.42)


class TestDefaultsDiversityFields:
    """DEFAULTS dict must include diversity fields."""

    def test_defaults_has_diversity(self) -> None:
        assert "diversity" in DEFAULTS["screen"]
        assert DEFAULTS["screen"]["diversity"] is False

    def test_defaults_has_diversity_lambda(self) -> None:
        assert "diversity_lambda" in DEFAULTS["screen"]
        assert DEFAULTS["screen"]["diversity_lambda"] == pytest.approx(0.3)


class TestCmdScreenDiversityWiring:
    """run_screen should forward diversity params to select_topk."""

    @patch("research_pipeline.cli.cmd_screen.select_topk")
    @patch("research_pipeline.cli.cmd_screen.score_candidates")
    @patch("research_pipeline.cli.cmd_screen.load_config")
    @patch("research_pipeline.cli.cmd_screen.init_run")
    @patch("research_pipeline.cli.cmd_screen.read_jsonl")
    def test_diversity_forwarded_from_config(
        self,
        mock_read_jsonl: MagicMock,
        mock_init_run: MagicMock,
        mock_load_config: MagicMock,
        mock_score: MagicMock,
        mock_topk: MagicMock,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """When diversity CLI args are None, config values are used."""
        run_root = tmp_path / "runs" / "test-run"  # type: ignore[operator]
        for stage in ("plan", "search", "screen"):
            (run_root / stage).mkdir(parents=True, exist_ok=True)

        # Write plan file
        plan_path = run_root / "plan" / "query_plan.json"
        plan_path.write_text(
            '{"topic_raw":"test","topic_normalized":"test",'
            '"research_question":"test","must_terms":[],"nice_terms":[],'
            '"negative_terms":[],"candidate_categories":[],"query_variants":[]}'
        )

        # Write candidates file
        (run_root / "search" / "candidates.jsonl").write_text("")

        mock_init_run.return_value = ("test-run", run_root)
        mock_read_jsonl.return_value = []

        config = PipelineConfig(
            screen=ScreenConfig(diversity=True, diversity_lambda=0.6)
        )
        mock_load_config.return_value = config
        mock_score.return_value = []
        mock_topk.return_value = []

        from research_pipeline.cli.cmd_screen import run_screen

        run_screen(run_id="test-run", workspace=tmp_path / "runs")  # type: ignore[operator]

        mock_topk.assert_called_once()
        call_kwargs = mock_topk.call_args
        assert (
            call_kwargs.kwargs.get("diversity") is True
            or call_kwargs[1].get("diversity") is True
        )
        diversity_lambda = call_kwargs.kwargs.get("diversity_lambda") or call_kwargs[
            1
        ].get("diversity_lambda")
        assert diversity_lambda == pytest.approx(0.6)

    @patch("research_pipeline.cli.cmd_screen.select_topk")
    @patch("research_pipeline.cli.cmd_screen.score_candidates")
    @patch("research_pipeline.cli.cmd_screen.load_config")
    @patch("research_pipeline.cli.cmd_screen.init_run")
    @patch("research_pipeline.cli.cmd_screen.read_jsonl")
    def test_cli_override_takes_precedence(
        self,
        mock_read_jsonl: MagicMock,
        mock_init_run: MagicMock,
        mock_load_config: MagicMock,
        mock_score: MagicMock,
        mock_topk: MagicMock,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """CLI --diversity flag overrides config value."""
        run_root = tmp_path / "runs" / "test-run"  # type: ignore[operator]
        for stage in ("plan", "search", "screen"):
            (run_root / stage).mkdir(parents=True, exist_ok=True)

        plan_path = run_root / "plan" / "query_plan.json"
        plan_path.write_text(
            '{"topic_raw":"test","topic_normalized":"test",'
            '"research_question":"test","must_terms":[],"nice_terms":[],'
            '"negative_terms":[],"candidate_categories":[],"query_variants":[]}'
        )
        (run_root / "search" / "candidates.jsonl").write_text("")

        mock_init_run.return_value = ("test-run", run_root)
        mock_read_jsonl.return_value = []

        config = PipelineConfig(
            screen=ScreenConfig(diversity=False, diversity_lambda=0.3)
        )
        mock_load_config.return_value = config
        mock_score.return_value = []
        mock_topk.return_value = []

        from research_pipeline.cli.cmd_screen import run_screen

        run_screen(
            run_id="test-run",
            workspace=tmp_path / "runs",  # type: ignore[operator]
            diversity=True,
            diversity_lambda=0.8,
        )

        mock_topk.assert_called_once()
        call_kwargs = mock_topk.call_args
        assert (
            call_kwargs.kwargs.get("diversity") is True
            or call_kwargs[1].get("diversity") is True
        )
        diversity_lambda = call_kwargs.kwargs.get("diversity_lambda") or call_kwargs[
            1
        ].get("diversity_lambda")
        assert diversity_lambda == pytest.approx(0.8)
