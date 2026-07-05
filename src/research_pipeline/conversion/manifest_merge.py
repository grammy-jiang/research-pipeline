"""Rebuild the unified convert manifest from the two-tier manifests (#30).

``convert-rough`` and ``convert-fine`` each write their own tier manifest, but
the unified ``convert/convert_manifest.jsonl`` that ``extract``/``summarize``
consume was only materialized the first time. A second tier run therefore left
the unified manifest stale, and downstream stages silently processed a subset.
This helper rebuilds the unified manifest from the union of the tier manifests
so it always reflects the latest runs.
"""

from __future__ import annotations

from pathlib import Path

from research_pipeline.models.conversion import ConvertManifestEntry
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir

_KEEP_STATUSES = ("converted", "skipped_exists")


def rebuild_unified_manifest(run_root: Path) -> list[ConvertManifestEntry]:
    """Merge the rough + fine tier manifests into the unified manifest.

    Deduplicated by ``arxiv_id`` with the fine tier overriding rough
    (last-wins). Writes ``convert/convert_manifest.jsonl`` and returns the
    merged entries. A no-op-safe call: missing tier manifests are skipped.
    """
    std_path = get_stage_dir(run_root, "convert_root") / "convert_manifest.jsonl"
    rough_path = (
        get_stage_dir(run_root, "convert_rough") / "convert_rough_manifest.jsonl"
    )
    fine_path = get_stage_dir(run_root, "convert_fine") / "convert_fine_manifest.jsonl"

    entries_by_id: dict[str, ConvertManifestEntry] = {}
    for path in (rough_path, fine_path):  # rough first so fine wins on conflict
        if not path.exists():
            continue
        for record in read_jsonl(path):
            entry = ConvertManifestEntry.model_validate(record)
            if entry.status in _KEEP_STATUSES:
                entries_by_id[entry.arxiv_id] = entry

    merged = list(entries_by_id.values())
    write_jsonl(std_path, [entry.model_dump(mode="json") for entry in merged])
    return merged
