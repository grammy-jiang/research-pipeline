"""Phase E E07 — link manual dossier output to daily brief and archive."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.briefing.dossier import (
    build_dossier,
    render_dossier,
    write_dossier,
    write_dossier_with_archive,
)
from research_pipeline.briefing.report import render_daily_brief
from tests.unit._dossier_fixtures import make_cluster


def test_daily_brief_with_no_dossiers_has_no_section() -> None:
    cluster = make_cluster()
    md = render_daily_brief([cluster], run_date="2026-04-29")
    assert "## Linked Dossiers" not in md


def test_daily_brief_renders_linked_dossiers_section() -> None:
    cluster = make_cluster()
    md = render_daily_brief(
        [cluster],
        run_date="2026-04-29",
        dossier_links=[("Acme v1.0 dossier", "reports/dossiers/acme.md")],
    )
    assert "## Linked Dossiers" in md
    assert "[Acme v1.0 dossier](reports/dossiers/acme.md)" in md


def test_daily_brief_dossier_links_preserve_order() -> None:
    cluster = make_cluster()
    links = [
        ("Topic A", "reports/dossiers/a.md"),
        ("Topic B", "reports/dossiers/b.md"),
    ]
    md = render_daily_brief([cluster], run_date="2026-04-29", dossier_links=links)
    pos_a = md.find("Topic A")
    pos_b = md.find("Topic B")
    assert 0 < pos_a < pos_b


def test_dossier_archive_link_writes_both_paths(tmp_path: Path) -> None:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md = render_dossier(dossier, run_date="2026-04-29")
    primary = tmp_path / "reports" / "dossiers" / "d.md"
    archive = tmp_path / "vault" / "Dossiers" / "d.md"
    written, archived = write_dossier_with_archive(primary, md, archive_path=archive)
    assert written.read_text(encoding="utf-8") == md
    assert archived is not None
    assert archived.read_text(encoding="utf-8") == md


def test_dossier_archive_optional(tmp_path: Path) -> None:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md = render_dossier(dossier, run_date="2026-04-29")
    primary = tmp_path / "d.md"
    written, archived = write_dossier_with_archive(primary, md)
    assert written.exists()
    assert archived is None


def test_dossier_archive_idempotent(tmp_path: Path) -> None:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md = render_dossier(dossier, run_date="2026-04-29")
    primary = tmp_path / "d.md"
    archive = tmp_path / "archive.md"
    write_dossier_with_archive(primary, md, archive_path=archive)
    write_dossier_with_archive(primary, md, archive_path=archive)
    assert primary.read_text(encoding="utf-8") == md
    assert archive.read_text(encoding="utf-8") == md


def test_write_dossier_creates_parent_dirs(tmp_path: Path) -> None:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md = render_dossier(dossier, run_date="2026-04-29")
    nested = tmp_path / "a" / "b" / "c" / "out.md"
    write_dossier(nested, md)
    assert nested.exists()
