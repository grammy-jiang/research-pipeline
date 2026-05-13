# Proof Pack: C04 Topic and source note export

## Ticket
`C04_topic_and_source_note_export`

## Implemented Files
- `src/research_pipeline/briefing/obsidian_topics.py`
- `src/research_pipeline/briefing/obsidian_sources.py`
- `tests/unit/test_briefing_obsidian_topics.py`
- `tests/unit/test_briefing_obsidian_sources.py`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_obsidian_topics.py tests/unit/test_briefing_obsidian_sources.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 16/16 unit tests pass; ruff clean; mypy clean (36 source files).

## Obsidian Safety Evidence
- `topic_note_path` and `source_note_path` reject any non-slug
  identifier before path construction, blocking traversal payloads such
  as `../escape`.
- Each export validates the resolved path with C01
  `validate_vault_path` and refuses to overwrite a non-owned file
  (regression tests confirm a hand-written file at the target path is
  preserved verbatim and an error is raised).
- `config.dry_run=True` returns `None` and writes nothing.
- Wiki-links inside the supplied bodies (`[[Source X]]`) are passed
  through unchanged — link generation is deferred to ticket C05.
- Re-running the export produces an identical file (idempotent).

## Next Ticket
`C05_wikilink_backlink_and_idempotent_update`
