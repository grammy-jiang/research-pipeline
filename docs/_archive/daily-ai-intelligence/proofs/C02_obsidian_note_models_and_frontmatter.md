# Proof Pack: C02 Obsidian note models and frontmatter

## Ticket
`C02_obsidian_note_models_and_frontmatter`

## Implemented Files
- `src/research_pipeline/briefing/obsidian_notes.py`
- `tests/unit/test_briefing_obsidian_notes.py`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_obsidian_notes.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 13/13 unit tests pass; ruff clean; mypy clean (33 source files).

## Obsidian Safety Evidence
- `ObsidianNote` enforces a slug-like `generated_id` so ownership checks in
  C01 (`is_owned_generated_note`) can match exactly.
- `render_frontmatter` is deterministic (sorted keys), required for
  idempotent re-writes in C03–C05.
- `ObsidianNote` validators reject reserved keys (`type`, `generated_id`)
  in `extra`, missing `## Agent Read Map` heading, and stray `---`
  separators that would corrupt frontmatter parsing.
- `parse_frontmatter` rejects missing/malformed YAML blocks so unowned
  human notes fail the ownership check rather than being silently
  overwritten.
- No filesystem I/O in this module — writing is delegated to later
  tickets that compose with C01 path validators.

## Next Ticket
`C03_daily_note_export`
