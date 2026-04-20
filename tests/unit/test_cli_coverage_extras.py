"""Extra coverage tests for CLI modules: cmd_validate, cmd_compare, app.py.

Targets uncovered lines:
  - cmd_validate.py  379-412, 435, 439-440 (run_validate handler paths)
  - cmd_compare.py   26, 42, 229-230, 348/350, 465-510 (run_compare handler)
  - app.py           689-707, 723-738, 1232 (setup / install-skill wrappers)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# cmd_validate — run_validate handler
# ===================================================================

class TestRunValidateHandler:
    """Cover run_validate when run_id + workspace are provided."""

    @patch("research_pipeline.cli.cmd_validate.validate_report")
    @patch("research_pipeline.storage.workspace.init_run")
    @patch("research_pipeline.storage.workspace.get_stage_dir")
    @patch("research_pipeline.config.loader.load_config")
    def test_run_id_finds_synthesis_report(
        self,
        mock_load_config: MagicMock,
        mock_stage_dir: MagicMock,
        mock_init_run: MagicMock,
        mock_validate: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Lines 378-410: run_id+workspace → finds synthesis_report.md."""
        from research_pipeline.cli.cmd_validate import run_validate

        run_root = tmp_path / "runs" / "test-run"
        run_root.mkdir(parents=True)
        mock_init_run.return_value = ("test-run", run_root)
        mock_load_config.return_value = MagicMock(workspace=str(tmp_path))

        # Set up stage directories
        synth_dir = tmp_path / "summarize"
        synth_dir.mkdir(parents=True)
        report_file = synth_dir / "synthesis_report.md"
        report_file.write_text("# Report\n## Executive Summary\nOK")

        screen_dir = tmp_path / "screen"
        screen_dir.mkdir(parents=True)
        shortlist = [
            {"paper": {"arxiv_id": "2401.00001", "title": "Paper A"}},
            {"paper": {"arxiv_id": "2401.00002", "title": "Paper B"}},
        ]
        (screen_dir / "shortlist.json").write_text(json.dumps(shortlist))

        def stage_dir_side_effect(root: Path, stage: str) -> Path:
            return tmp_path / stage

        mock_stage_dir.side_effect = stage_dir_side_effect
        mock_validate.return_value = {
            "verdict": "PASS",
            "overall_score": 0.85,
            "issues": [],
        }

        run_validate(
            report=None,
            workspace=tmp_path,
            run_id="test-run",
            output=tmp_path / "out.json",
        )

        mock_validate.assert_called_once()
        call_kwargs = mock_validate.call_args
        # Verify paper_ids were loaded from shortlist
        assert call_kwargs[1]["paper_ids"] == ["2401.00001", "2401.00002"]
        assert call_kwargs[1]["paper_titles"] == ["Paper A", "Paper B"]
        assert (tmp_path / "out.json").exists()

    @patch("research_pipeline.cli.cmd_validate.validate_report")
    @patch("research_pipeline.storage.workspace.init_run")
    @patch("research_pipeline.storage.workspace.get_stage_dir")
    @patch("research_pipeline.config.loader.load_config")
    def test_run_id_no_synthesis_report(
        self,
        mock_load_config: MagicMock,
        mock_stage_dir: MagicMock,
        mock_init_run: MagicMock,
        mock_validate: MagicMock,
        tmp_path: Path,
    ) -> None:
        """run_id+workspace but no synthesis_report.md → logs error, returns."""
        from research_pipeline.cli.cmd_validate import run_validate

        run_root = tmp_path / "runs" / "test-run"
        run_root.mkdir(parents=True)
        mock_init_run.return_value = ("test-run", run_root)
        mock_load_config.return_value = MagicMock(workspace=str(tmp_path))

        # Dirs exist but no report file
        synth_dir = tmp_path / "summarize"
        synth_dir.mkdir(parents=True)
        screen_dir = tmp_path / "screen"
        screen_dir.mkdir(parents=True)

        def stage_dir_side_effect(root: Path, stage: str) -> Path:
            return tmp_path / stage

        mock_stage_dir.side_effect = stage_dir_side_effect

        run_validate(
            report=None,
            workspace=tmp_path,
            run_id="test-run",
        )

        # validate_report should NOT be called — no report found
        mock_validate.assert_not_called()

    @patch("research_pipeline.cli.cmd_validate.validate_report")
    def test_no_issues_log_path(
        self,
        mock_validate: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Line 435: validation passes with zero issues."""
        from research_pipeline.cli.cmd_validate import run_validate

        report_file = tmp_path / "report.md"
        report_file.write_text("# Report\n## Executive Summary\nOK")

        mock_validate.return_value = {
            "verdict": "PASS",
            "overall_score": 0.9,
            "issues": [],
        }

        run_validate(report=report_file)
        mock_validate.assert_called_once()
        # Verify output written next to report
        assert (tmp_path / "validation_result.json").exists()

    @patch("research_pipeline.cli.cmd_validate.validate_report")
    def test_fact_score_logging(
        self,
        mock_validate: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Lines 439-440: FACT score is logged when present in result."""
        from research_pipeline.cli.cmd_validate import run_validate

        report_file = tmp_path / "report.md"
        report_file.write_text("# Report\n## Executive Summary\nOK")

        mock_validate.return_value = {
            "verdict": "PASS",
            "overall_score": 0.85,
            "issues": [],
            "fact_score": {
                "citation_accuracy": 0.92,
                "effective_citation_ratio": 0.78,
                "verified_citations": 8,
                "total_citations": 10,
            },
        }

        run_validate(report=report_file)
        mock_validate.assert_called_once()

    @patch("research_pipeline.cli.cmd_validate.validate_report")
    def test_issues_are_logged(
        self,
        mock_validate: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Cover the issues iteration branch (line 432-433)."""
        from research_pipeline.cli.cmd_validate import run_validate

        report_file = tmp_path / "report.md"
        report_file.write_text("# Report\n")

        mock_validate.return_value = {
            "verdict": "FAIL",
            "overall_score": 0.3,
            "issues": ["Missing sections", "No citations"],
        }

        run_validate(report=report_file)
        mock_validate.assert_called_once()

    @patch("research_pipeline.cli.cmd_validate.validate_report")
    @patch("research_pipeline.storage.workspace.init_run")
    @patch("research_pipeline.storage.workspace.get_stage_dir")
    @patch("research_pipeline.config.loader.load_config")
    def test_shortlist_json_decode_error(
        self,
        mock_load_config: MagicMock,
        mock_stage_dir: MagicMock,
        mock_init_run: MagicMock,
        mock_validate: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Line 411-412: JSONDecodeError when reading shortlist."""
        from research_pipeline.cli.cmd_validate import run_validate

        run_root = tmp_path / "runs" / "test-run"
        run_root.mkdir(parents=True)
        mock_init_run.return_value = ("test-run", run_root)
        mock_load_config.return_value = MagicMock(workspace=str(tmp_path))

        synth_dir = tmp_path / "summarize"
        synth_dir.mkdir(parents=True)
        report_file = synth_dir / "synthesis_report.md"
        report_file.write_text("# Report\n## Executive Summary\nOK")

        screen_dir = tmp_path / "screen"
        screen_dir.mkdir(parents=True)
        (screen_dir / "shortlist.json").write_text("INVALID JSON{{{")

        def stage_dir_side_effect(root: Path, stage: str) -> Path:
            return tmp_path / stage

        mock_stage_dir.side_effect = stage_dir_side_effect

        mock_validate.return_value = {
            "verdict": "PASS",
            "overall_score": 0.8,
            "issues": [],
        }

        # Should not raise — catches JSONDecodeError
        run_validate(
            report=None,
            workspace=tmp_path,
            run_id="test-run",
        )
        # paper_ids/paper_titles will be empty due to JSON error
        call_kwargs = mock_validate.call_args
        assert call_kwargs[1]["paper_ids"] is None
        assert call_kwargs[1]["paper_titles"] is None

    @patch("research_pipeline.cli.cmd_validate.validate_report")
    def test_output_written_to_explicit_path(
        self,
        mock_validate: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Cover the output write path (line 449-451)."""
        from research_pipeline.cli.cmd_validate import run_validate

        report_file = tmp_path / "report.md"
        report_file.write_text("# Report\nContent")
        out_path = tmp_path / "custom_output.json"

        mock_validate.return_value = {
            "verdict": "PASS",
            "overall_score": 0.9,
            "issues": [],
        }

        run_validate(report=report_file, output=out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["verdict"] == "PASS"


# ===================================================================
# cmd_compare — helper functions and run_compare handler
# ===================================================================


class TestReadRecordsJson:
    """Line 26: _read_records JSON array branch."""

    def test_reads_json_file(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_compare import _read_records

        p = tmp_path / "data.json"
        p.write_text(json.dumps([{"id": "a"}, {"id": "b"}]))
        result = _read_records(p)
        assert len(result) == 2
        assert result[0]["id"] == "a"


class TestLoadPaperIdsEmptyReturn:
    """Line 42: _load_paper_ids returns empty set when no files exist."""

    def test_returns_empty_set(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_compare import _load_paper_ids

        # run_root with no screen or search dirs
        run_root = tmp_path / "empty-run"
        run_root.mkdir()
        result = _load_paper_ids(run_root)
        assert result == set()


class TestLoadSynthesisJsonDecodeError:
    """Lines 229-230: JSONDecodeError in _load_synthesis_json."""

    def test_json_decode_error_returns_none(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_compare import _load_synthesis_json

        run_root = tmp_path / "run"
        synth_dir = run_root / "synthesis"
        synth_dir.mkdir(parents=True)
        (synth_dir / "synthesis_results.json").write_text("NOT JSON{{")

        result = _load_synthesis_json(run_root)
        assert result is None


class TestDiffReadinessNonDict:
    """Lines 348/350: readiness value is not a dict."""

    def test_non_dict_readiness_a(self) -> None:
        from research_pipeline.cli.cmd_compare import _diff_readiness

        result = _diff_readiness(
            {"readiness": "READY"},  # non-dict
            {"readiness": {"verdict": "NOT_READY"}},
        )
        assert result["verdict_run_a"] == "UNKNOWN"
        assert result["verdict_run_b"] == "NOT_READY"

    def test_non_dict_readiness_b(self) -> None:
        from research_pipeline.cli.cmd_compare import _diff_readiness

        result = _diff_readiness(
            {"readiness": {"verdict": "READY"}},
            {"readiness": "stringvalue"},  # non-dict
        )
        assert result["verdict_run_a"] == "READY"
        assert result["verdict_run_b"] == "UNKNOWN"

    def test_both_non_dict(self) -> None:
        from research_pipeline.cli.cmd_compare import _diff_readiness

        result = _diff_readiness(
            {"readiness": 42},
            {"readiness": True},
        )
        assert result["verdict_run_a"] == "UNKNOWN"
        assert result["verdict_run_b"] == "UNKNOWN"


class TestRunCompareHandler:
    """Lines 465-510: run_compare full handler."""

    @patch("research_pipeline.cli.cmd_compare.compare_runs")
    @patch("research_pipeline.cli.cmd_compare.init_run")
    @patch("research_pipeline.cli.cmd_compare.load_config")
    def test_full_compare_flow(
        self,
        mock_config: MagicMock,
        mock_init_run: MagicMock,
        mock_compare: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Both run IDs provided → loads config, inits runs, calls compare."""
        from research_pipeline.cli.cmd_compare import run_compare

        root_a = tmp_path / "run-a"
        root_b = tmp_path / "run-b"
        root_a.mkdir()
        root_b.mkdir()

        mock_config.return_value = MagicMock(workspace=str(tmp_path))
        mock_init_run.side_effect = [
            ("run-a", root_a),
            ("run-b", root_b),
        ]

        mock_compare.return_value = {
            "run_a": "run-a",
            "run_b": "run-b",
            "paper_diff": {
                "count_a": 5,
                "count_b": 7,
                "overlap": 3,
                "new_in_b": 4,
                "dropped_from_a": 2,
            },
            "gap_analysis": {
                "resolved_count": 1,
                "new_count": 2,
                "persistent_count": 1,
                "resolved_gaps": [],
                "new_gaps": [],
                "persistent_gaps": [],
            },
            "confidence_changes": [{"finding": "X", "old": "low", "new": "high"}],
            "readiness": {
                "verdict_run_a": "NOT_READY",
                "verdict_run_b": "IMPLEMENTATION_READY",
            },
        }

        out_path = tmp_path / "comparison.json"
        run_compare(
            run_id_a="run-a",
            run_id_b="run-b",
            config_path=None,
            workspace=tmp_path,
            output=out_path,
        )

        mock_compare.assert_called_once_with(root_a, root_b, "run-a", "run-b")
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["run_a"] == "run-a"

    @patch("research_pipeline.cli.cmd_compare.load_config")
    def test_missing_run_ids(
        self,
        mock_config: MagicMock,
    ) -> None:
        """Lines 465-467: missing run IDs → logs error, returns early."""
        from research_pipeline.cli.cmd_compare import run_compare

        # Neither ID provided
        run_compare(run_id_a=None, run_id_b=None)
        mock_config.assert_not_called()

        # Only one ID
        run_compare(run_id_a="run-a", run_id_b=None)
        mock_config.assert_not_called()

        run_compare(run_id_a=None, run_id_b="run-b")
        mock_config.assert_not_called()

    @patch("research_pipeline.cli.cmd_compare.compare_runs")
    @patch("research_pipeline.cli.cmd_compare.init_run")
    @patch("research_pipeline.cli.cmd_compare.load_config")
    def test_default_output_path(
        self,
        mock_config: MagicMock,
        mock_init_run: MagicMock,
        mock_compare: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 508: default output path when no output specified."""
        from research_pipeline.cli.cmd_compare import run_compare

        monkeypatch.chdir(tmp_path)
        mock_config.return_value = MagicMock(workspace=str(tmp_path))
        mock_init_run.side_effect = [
            ("run-a", tmp_path / "a"),
            ("run-b", tmp_path / "b"),
        ]
        mock_compare.return_value = {
            "run_a": "run-a",
            "run_b": "run-b",
            "paper_diff": {
                "count_a": 0, "count_b": 0, "overlap": 0,
                "new_in_b": 0, "dropped_from_a": 0,
            },
            "gap_analysis": {
                "resolved_count": 0, "new_count": 0, "persistent_count": 0,
                "resolved_gaps": [], "new_gaps": [], "persistent_gaps": [],
            },
            "confidence_changes": [],
            "readiness": {"verdict_run_a": "UNKNOWN", "verdict_run_b": "UNKNOWN"},
        }

        run_compare(
            run_id_a="run-a",
            run_id_b="run-b",
            output=None,
        )
        default_out = Path("comparison_run-a_vs_run-b.json")
        assert default_out.exists()
        default_out.unlink()  # cleanup


# ===================================================================
# app.py — setup and install-skill wrappers
# ===================================================================


class TestSetupCommand:
    """Lines 689-707: setup command wrapper in app.py."""

    @patch("research_pipeline.cli.cmd_setup.run_setup")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_setup_defaults(
        self,
        mock_logging: MagicMock,
        mock_run_setup: MagicMock,
    ) -> None:
        from research_pipeline.cli.app import setup

        setup(
            skill_target="",
            agents_target="",
            symlink=False,
            force=False,
            skip_skill=False,
            skip_agents=False,
            verbose=False,
        )

        mock_logging.assert_called_once()
        mock_run_setup.assert_called_once()
        kw = mock_run_setup.call_args[1]
        assert kw["skill_target"] is None
        assert kw["symlink"] is False
        assert kw["force"] is False
        assert kw["skip_skill"] is False
        assert kw["skip_agents"] is False

    @patch("research_pipeline.cli.cmd_setup.run_setup")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_setup_custom_paths(
        self,
        mock_logging: MagicMock,
        mock_run_setup: MagicMock,
    ) -> None:
        from research_pipeline.cli.app import setup

        setup(
            skill_target="/custom/skill",
            agents_target="/custom/agents",
            symlink=True,
            force=True,
            skip_skill=True,
            skip_agents=True,
            verbose=True,
        )

        kw = mock_run_setup.call_args[1]
        assert kw["skill_target"] == Path("/custom/skill")
        assert kw["agents_target"] == Path("/custom/agents")
        assert kw["symlink"] is True
        assert kw["force"] is True
        assert kw["skip_skill"] is True
        assert kw["skip_agents"] is True

    @patch("research_pipeline.cli.cmd_setup.run_setup")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_setup_verbose_sets_debug(
        self,
        mock_logging: MagicMock,
        mock_run_setup: MagicMock,
    ) -> None:
        import logging

        from research_pipeline.cli.app import setup

        setup(
            skill_target="",
            agents_target="",
            symlink=False,
            force=False,
            skip_skill=False,
            skip_agents=False,
            verbose=True,
        )
        mock_logging.assert_called_once_with(level=logging.DEBUG)


class TestInstallSkillDeprecated:
    """Lines 723-738: install_skill wrapper emits DeprecationWarning."""

    @patch("research_pipeline.cli.cmd_setup.run_setup")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_install_skill_warns(
        self,
        mock_logging: MagicMock,
        mock_run_setup: MagicMock,
    ) -> None:
        from research_pipeline.cli.app import install_skill

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            install_skill(
                target="",
                symlink=False,
                force=False,
                verbose=False,
            )

        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert any("deprecated" in str(w.message).lower() for w in caught)
        mock_run_setup.assert_called_once()
        kw = mock_run_setup.call_args[1]
        assert kw["skip_agents"] is True

    @patch("research_pipeline.cli.cmd_setup.run_setup")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_install_skill_custom_target(
        self,
        mock_logging: MagicMock,
        mock_run_setup: MagicMock,
    ) -> None:
        from research_pipeline.cli.app import install_skill

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            install_skill(
                target="/my/dir",
                symlink=True,
                force=True,
                verbose=True,
            )

        kw = mock_run_setup.call_args[1]
        assert kw["skill_target"] == Path("/my/dir")
        assert kw["symlink"] is True
        assert kw["force"] is True

    @patch("research_pipeline.cli.cmd_setup.run_setup")
    @patch("research_pipeline.infra.logging.setup_logging")
    def test_install_skill_verbose_debug(
        self,
        mock_logging: MagicMock,
        mock_run_setup: MagicMock,
    ) -> None:
        import logging

        from research_pipeline.cli.app import install_skill

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            install_skill(target="", symlink=False, force=False, verbose=True)

        mock_logging.assert_called_once_with(level=logging.DEBUG)
