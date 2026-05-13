# Proof Pack: C07 CLI export to Obsidian

## Ticket
`C07_cli_export_obsidian`

## Implemented Files
- `src/research_pipeline/cli/cmd_brief.py` (rewired `export-obsidian`)
- `tests/unit/test_briefing_cli_export_obsidian.py`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_cli_export_obsidian.py -xvs
uv run ruff format src/research_pipeline/cli/cmd_brief.py tests/unit/test_briefing_cli_export_obsidian.py
uv run ruff check src/research_pipeline/briefing src/research_pipeline/cli/cmd_brief.py tests/unit/test_briefing_cli_export_obsidian.py
uv run mypy src/research_pipeline/briefing src/research_pipeline/cli/cmd_brief.py
```

## Result
PASS — 5/5 CLI unit tests pass; ruff clean; mypy clean (39 source files).

## Obsidian Safety Evidence
- `export-obsidian` now constructs `ObsidianConfig(vault_root=vault, dry_run=...)`
  and routes Daily / Topic / Source notes through the C03 / C04 builders +
  exporters, so vault-path and `generated_id` ownership rules are enforced
  by the trusted lower layers (no string-template paths in the CLI).
- New `--dry-run` flag short-circuits before any file is written and only
  logs planned target paths (Daily + per-topic + per-source).
- A `ValueError` raised by any exporter (vault escape, refusing to
  overwrite a non-owned note) is caught and converted to `typer.Exit(1)`
  with `logger.error`; no partial vault state is committed past the point
  of refusal.
- After a successful real (non-dry-run) write, the CLI invokes
  `validate_obsidian_export(...)` (C06) on the just-written notes and
  returns exit code 1 with logged errors if any structural check fails.
- `[[wiki-link]]` syntax in note bodies is preserved end-to-end; the test
  for human-note overwrite confirms the CLI refuses to clobber a hand-
  written file (no `generated_id`).

## Test Coverage
- Daily / Topic / Source notes written to `<vault>/AI-Intelligence/...`
- `--dry-run` produces no markdown files
- Re-running `export-obsidian` is byte-idempotent for the daily note
- Refuses to overwrite a hand-written note without `generated_id`
- Re-runs over a corrupted but owned note restore a valid note layout

## Next Ticket
`C08_obsidian_offline_e2e`
