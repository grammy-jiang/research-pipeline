# Feature Proof Pack: C08_obsidian_offline_e2e

## Feature ID
`C08_obsidian_offline_e2e`

## Verification Summary
All acceptance verification commands run on the offline `obsidian_export/`
and `obsidian_existing_notes/` fixture decks:

```bash
uv run pytest tests/integration_offline/test_briefing_phase_c_obsidian_e2e.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

Results:
- pytest: **6 passed in 0.42s** (full export, dry-run, idempotency, owned-note
  rewrite, human-note refusal, unsafe-vault rejection).
- ruff: **All checks passed!**
- mypy: **Success: no issues found in 38 source files**.

## Evidence
- `tests/integration_offline/test_briefing_phase_c_obsidian_e2e.py` —
  drives `research-pipeline brief export-obsidian` end-to-end against a
  real briefing workspace seeded by `run_briefing(...)` from offline
  fixtures, asserting:
  1. `test_full_export_writes_validated_notes` — Daily/Topics/Sources
     notes appear under `<vault>/AI-Intelligence/`, all pass
     `validate_obsidian_export`.
  2. `test_dry_run_does_not_write` — `--dry-run` produces zero `.md`
     under the vault.
  3. `test_repeated_export_is_byte_idempotent` — second export leaves
     all owned notes byte-identical.
  4. `test_overwrites_owned_generated_note_in_place` — generator
     reclaims an owned note whose body was tampered with (`generated_id`
     intact).
  5. `test_refuses_to_overwrite_human_note` — pre-seeded human daily
     note (no `generated_id`) survives; CLI exits non-zero.
  6. `test_unsafe_vault_traversal_is_rejected` — non-existent vault root
     triggers `pydantic.ValidationError` at `ObsidianConfig` construction.
- `tests/fixtures/briefing/e2e/obsidian_export/` and
  `tests/fixtures/briefing/e2e/obsidian_existing_notes/` — offline
  registry + feed/release fixtures (RSS XML, GitHub releases JSON,
  registry TOML); no network access.

## Test Coverage
- New offline e2e module: 6 tests, all green.
- Reuses C01–C07 production code unchanged.

## Wiki-Link / Vault-Safety Preservation
- `validate_obsidian_export` enforces frontmatter integrity, required
  `## Agent Read Map` heading, and (via the underlying note schema) the
  Phase C wiki-link conventions on every produced note in the success
  test.
- Generated-note ownership rule exercised in scenarios 4 and 5; vault
  path safety exercised in scenario 6.

## Phase Compliance
- No Phase D+ functionality introduced.
- No new dependencies added.
- Tests are fully offline (no HTTP).
- Updates only files listed in `phase-status.yaml :: C08.owned_files`
  (the test module plus the two fixture directories).

## Next Ticket
Phase C is now feature-complete. Mark `phases.C.status: complete` and
clear `current_ticket`. Phase D remains `blocked_until_C_done` per
project convention; do not auto-advance.
