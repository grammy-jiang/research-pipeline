# Feature Acceptance Contract: Add offline Obsidian export e2e tests

## Feature ID
`C08_obsidian_offline_e2e`

## Goal
Add offline Obsidian export e2e tests for Phase C Obsidian archive/export.

## In Scope
Implement only this Phase C ticket; reuse A/B briefing surfaces; tests first; offline normal tests; preserve wiki-links; enforce vault path safety; write proof pack.

## Out of Scope
Phase D+ functionality, dossiers, feedback learning, social sources, MCP expansion, UI, automatic source expansion, new dependencies unless justified.

## Expected Owned Files
- `tests/integration_offline/test_briefing_phase_c_obsidian_e2e.py`
- `tests/fixtures/briefing/e2e/obsidian_export/`
- `tests/fixtures/briefing/e2e/obsidian_existing_notes/`

## Required Tests
- `tests/integration_offline/test_briefing_phase_c_obsidian_e2e.py`

## Required Fixtures
- `tests/fixtures/briefing/e2e/obsidian_export/`
- `tests/fixtures/briefing/e2e/obsidian_existing_notes/`

## Failure Cases To Cover
Missing vault config; unsafe path traversal; existing human note without generated ID; invalid frontmatter; missing headings; broken/mutated wiki-links; repeated export idempotency; no network in normal tests.

## Verification Commands
```bash
uv run pytest tests/integration_offline/test_briefing_phase_c_obsidian_e2e.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; no later-phase functionality; vault path safety enforced where applicable; wiki-links preserved where applicable; generated-note ownership respected where applicable; status/proof pack updated.
