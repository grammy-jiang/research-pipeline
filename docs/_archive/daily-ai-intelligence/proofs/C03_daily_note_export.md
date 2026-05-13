# Proof Pack: C03 Daily note export

## Ticket
`C03_daily_note_export`

## Implemented Files
- `src/research_pipeline/briefing/obsidian_daily.py`
- `tests/unit/test_briefing_obsidian_daily.py`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_obsidian_daily.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 9/9 daily-note unit tests pass; ruff clean; mypy clean (34 source files).

## Obsidian Safety Evidence
- `daily_note_path` always builds the path via `config.namespace_root /
  "Daily" / f"{run_date}.md"`, then `export_daily_note` calls
  `validate_vault_path` (C01) for resolved-path safety.
- Existing files at the target are guarded by C01
  `is_owned_generated_note`; missing or mismatched `generated_id` raises
  and the existing file is preserved (regression test
  `test_export_daily_note_refuses_to_overwrite_human_note`).
- `config.dry_run=True` returns `None` and writes nothing
  (`test_export_daily_note_dry_run_writes_nothing`).
- Wiki-links in the brief body (`[[Topic Alpha]]`) round-trip verbatim;
  the legacy `---` frontmatter from `render_daily_brief` is stripped so
  the Obsidian-shaped frontmatter is the only YAML block.
- Re-running the export with the same inputs produces an identical file
  (idempotent rewrite, deterministic frontmatter ordering from C02).

## Next Ticket
`C04_topic_and_source_note_export`
