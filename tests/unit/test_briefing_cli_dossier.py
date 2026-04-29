"""Phase E E03 — manual CLI command brief dossier --cluster."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from research_pipeline.briefing.io import write_jsonl
from research_pipeline.briefing.layout import resolve_briefing_paths
from research_pipeline.cli.app import app
from tests.unit._dossier_fixtures import make_cluster

DATE = "2026-04-29"


def _setup_workspace(tmp_path: Path) -> Path:
    paths = resolve_briefing_paths(tmp_path, DATE)
    paths.create()
    cluster = make_cluster(cluster_id="cluster_a")
    write_jsonl(paths.ranked_clusters_path, [cluster])
    return tmp_path


def test_dossier_cli_manual_happy_path(tmp_path: Path) -> None:
    _setup_workspace(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "brief",
            "dossier",
            "--cluster",
            "cluster_a",
            "--workspace",
            str(tmp_path),
            "--date",
            DATE,
        ],
    )
    assert result.exit_code == 0, result.output
    paths = resolve_briefing_paths(tmp_path, DATE)
    files = list((paths.reports_dir / "dossiers").glob("*.md"))
    assert len(files) == 1
    md = files[0].read_text(encoding="utf-8")
    assert "## Evidence Timeline" in md
    assert "factuality_label=supported_fact" in md


def test_dossier_cli_missing_cluster_and_auto_errors(tmp_path: Path) -> None:
    _setup_workspace(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "brief",
            "dossier",
            "--workspace",
            str(tmp_path),
            "--date",
            DATE,
        ],
    )
    assert result.exit_code != 0
    assert "cluster" in result.output.lower() or "auto" in result.output.lower()


def test_dossier_cli_unknown_cluster_id_errors(tmp_path: Path) -> None:
    _setup_workspace(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "brief",
            "dossier",
            "--cluster",
            "nonexistent",
            "--workspace",
            str(tmp_path),
            "--date",
            DATE,
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_dossier_cli_rejects_cluster_without_primary_artifact(tmp_path: Path) -> None:
    paths = resolve_briefing_paths(tmp_path, DATE)
    paths.create()
    bad = make_cluster(cluster_id="cluster_no_pa", primary_artifact_present=False)
    write_jsonl(paths.ranked_clusters_path, [bad])
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "brief",
            "dossier",
            "--cluster",
            "cluster_no_pa",
            "--workspace",
            str(tmp_path),
            "--date",
            DATE,
        ],
    )
    assert result.exit_code != 0


def test_dossier_cli_writes_under_workspace(tmp_path: Path) -> None:
    _setup_workspace(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "brief",
            "dossier",
            "--cluster",
            "cluster_a",
            "--workspace",
            str(tmp_path),
            "--date",
            DATE,
        ],
    )
    assert result.exit_code == 0
    paths = resolve_briefing_paths(tmp_path, DATE)
    output_dir = paths.reports_dir / "dossiers"
    assert output_dir.is_dir()
    written = list(output_dir.glob("*.md"))
    assert all(p.is_relative_to(tmp_path) for p in written)
