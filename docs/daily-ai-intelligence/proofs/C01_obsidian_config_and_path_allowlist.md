# Proof Pack: Add Obsidian config and path allowlist

## Ticket
`C01_obsidian_config_and_path_allowlist`

## Implemented Files
- `src/research_pipeline/briefing/obsidian.py` (added `ObsidianConfig`, `validate_vault_path`, `is_owned_generated_note`, `GENERATED_ID_KEY`, `DEFAULT_VAULT_SUBDIR`, `DEFAULT_ALLOWED_SUBDIRS`; existing `export_daily_note`/`export_topic_notes`/`export_source_notes` preserved)
- `tests/unit/test_briefing_obsidian_paths.py` (15 unit tests)

## DryRUN Summary
- Files added/modified: `src/research_pipeline/briefing/obsidian.py`,
  `tests/unit/test_briefing_obsidian_paths.py`.
- Public surfaces added: `ObsidianConfig` (Pydantic, frozen), `validate_vault_path`,
  `is_owned_generated_note`, constants `GENERATED_ID_KEY`,
  `DEFAULT_VAULT_SUBDIR`, `DEFAULT_ALLOWED_SUBDIRS`.
- Test fixtures: none (tmp_path).
- Validation commands: `uv run pytest tests/unit/test_briefing_obsidian_paths.py -xvs`,
  `uv run ruff check src/research_pipeline/briefing tests/`,
  `uv run mypy src/research_pipeline/briefing`.
- Cases covered: defaults; missing/non-directory `vault_root`; valid Daily note;
  parent traversal `..`; outside-vault absolute path; subdir not in allowlist;
  outside-namespace path; non-`.md` extension; symlink escape; ownership
  match; ownership mismatch; human-note (no frontmatter); missing-file owned;
  custom allowlist (Weekly accepted, Topics rejected).
- Predicted outputs/exceptions: `ValueError` for every unsafe path with
  matching messages; `True/False` from `is_owned_generated_note`. No DB,
  no telemetry, no I/O outside tmp_path.
- No CLI/MCP/skill surfaces changed in this ticket.

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_obsidian_paths.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
uv run pytest tests/unit/test_briefing_*.py tests/integration_offline/test_briefing_phase_a_e2e.py tests/integration_offline/test_briefing_phase_b_memory_e2e.py -q
```

## Result
PASS

- 15/15 C01 unit tests pass.
- ruff: `All checks passed!`
- mypy: `Success: no issues found in 32 source files`
- Full Phase A+B+C-so-far regression: 202 passed.

## Obsidian Safety Evidence
- Writes are path-allowlisted: `validate_vault_path` resolves the target,
  requires it to be under `vault_root`, under the configured namespace
  subdir (default `AI-Intelligence`), and within
  `allowed_subdirs` (default `Daily`, `Topics`, `Sources`, `Weekly`,
  `Monthly`); only `.md` paths accepted; symlink escapes are rejected
  because resolution happens before the check.
- Dry-run is supported at config level (`ObsidianConfig.dry_run` flag);
  C01 only adds the surface — actual write commands honoring it land in
  C03/C04/C07.
- Wiki-links are preserved: C01 introduces no rendering; existing
  exporters keep their wiki-link-free Markdown unchanged. C05 will own
  the wiki-link/backlink tests.
- Human notes are not overwritten without matching generated ID:
  `is_owned_generated_note` returns `True` only when the file is missing
  or its YAML frontmatter contains `generated_id: <expected>`. Any other
  state (no frontmatter, mismatched id, unreadable file) returns `False`.
- Changed notes are reported: helpers return resolved paths so callers
  can list planned/actual changes; full reporting integrates in C07.

## Phase Compliance
- No Phase D+ functionality (no dossiers, feedback, social sources, MCP
  expansion, UI).
- No new dependencies introduced (Pydantic already used project-wide).
- No network calls.

## Next Ticket
`C02_obsidian_note_models_and_frontmatter`
