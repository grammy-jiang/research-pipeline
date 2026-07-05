"""Tests for unified convert-manifest rebuild from tier manifests (#30)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from research_pipeline.conversion.manifest_merge import rebuild_unified_manifest
from research_pipeline.models.conversion import ConvertManifestEntry
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir


def _entry(arxiv_id: str, tier: str, md: str = "m.md") -> ConvertManifestEntry:
    return ConvertManifestEntry(
        arxiv_id=arxiv_id,
        version="v1",
        pdf_path="p.pdf",
        pdf_sha256="x",
        markdown_path=md,
        converter_name="c",
        converter_version="1",
        converter_config_hash="h",
        converted_at=datetime(2024, 1, 1, tzinfo=UTC),
        status="converted",
        tier=tier,  # type: ignore[arg-type]
    )


def _write(path: Path, entries: list[ConvertManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, [e.model_dump(mode="json") for e in entries])


def test_rebuild_merges_tiers_fine_wins(tmp_path: Path) -> None:
    run_root = tmp_path / "run1"
    rough = get_stage_dir(run_root, "convert_rough") / "convert_rough_manifest.jsonl"
    fine = get_stage_dir(run_root, "convert_fine") / "convert_fine_manifest.jsonl"
    _write(rough, [_entry("a", "rough"), _entry("b", "rough")])
    _write(fine, [_entry("a", "fine", md="fine.md")])

    merged = rebuild_unified_manifest(run_root)

    by_id = {e.arxiv_id: e for e in merged}
    assert set(by_id) == {"a", "b"}
    assert by_id["a"].tier == "fine"  # fine overrides rough
    std = get_stage_dir(run_root, "convert_root") / "convert_manifest.jsonl"
    assert len(read_jsonl(std)) == 2


def test_rebuild_refreshes_stale_unified(tmp_path: Path) -> None:
    # A prior run left a 1-entry unified manifest; a re-run has 3 rough entries.
    run_root = tmp_path / "run2"
    std = get_stage_dir(run_root, "convert_root") / "convert_manifest.jsonl"
    _write(std, [_entry("old", "rough")])
    rough = get_stage_dir(run_root, "convert_rough") / "convert_rough_manifest.jsonl"
    _write(rough, [_entry(x, "rough") for x in ("a", "b", "c")])

    merged = rebuild_unified_manifest(run_root)

    assert {e.arxiv_id for e in merged} == {"a", "b", "c"}
    assert len(read_jsonl(std)) == 3
