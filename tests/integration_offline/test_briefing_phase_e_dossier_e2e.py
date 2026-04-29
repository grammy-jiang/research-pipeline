"""Phase E E08 — offline dossier e2e tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from research_pipeline.briefing.layout import resolve_briefing_paths
from research_pipeline.briefing.validate_dossier import validate_dossier_markdown
from research_pipeline.cli.app import app

FIXTURES = Path(__file__).parent.parent / "fixtures" / "briefing" / "e2e"
DATE = "2026-04-29"


def _bootstrap(tmp_path: Path, fixture_dir: Path) -> Path:
    paths = resolve_briefing_paths(tmp_path, DATE)
    paths.create()
    src = fixture_dir / "ranked_clusters.jsonl"
    target = paths.ranked_clusters_path
    target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def test_dossier_manual_happy_path(tmp_path: Path) -> None:
    _bootstrap(tmp_path, FIXTURES / "dossier_manual")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "brief",
            "dossier",
            "--cluster",
            "cluster_release_a",
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
    assert "https://example.com/release-a" in md
    validation = validate_dossier_markdown(md)
    assert validation.passed, validation.errors


def test_dossier_no_primary_artifact_rejected(tmp_path: Path) -> None:
    _bootstrap(tmp_path, FIXTURES / "dossier_no_primary_artifact")
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
    paths = resolve_briefing_paths(tmp_path, DATE)
    out_dir = paths.reports_dir / "dossiers"
    if out_dir.exists():
        assert not list(out_dir.glob("*.md"))


def test_dossier_long_rejection_via_validator(tmp_path: Path) -> None:
    _bootstrap(tmp_path, FIXTURES / "dossier_long_rejection")
    fake_md = (
        "---\ntype: topic-dossier\ndate: 2026-04-29\n---\n"
        "# overlong\n"
        "## Agent Read Map\n## One-paragraph Summary\n## What Changed\n"
        "## Why It Matters Technically\n## Evidence Timeline\n"
        "## Artifacts To Open\n## Open Questions\n## Agent Notes\n"
        "factuality_label=supported_fact https://example.com/x\n"
        + " ".join(["pad"] * 5000)
    )
    result = validate_dossier_markdown(fake_md)
    assert not result.passed
    assert any("word count" in e and "exceeds" in e for e in result.errors)


def test_dossier_offline_no_network(monkeypatch, tmp_path: Path) -> None:
    """Sanity check that the e2e flow uses no network."""
    import socket

    def _no_network(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("network access disabled in offline test")

    monkeypatch.setattr(socket, "create_connection", _no_network)
    _bootstrap(tmp_path, FIXTURES / "dossier_manual")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "brief",
            "dossier",
            "--cluster",
            "cluster_release_a",
            "--workspace",
            str(tmp_path),
            "--date",
            DATE,
        ],
    )
    assert result.exit_code == 0, result.output
