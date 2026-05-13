# Feature Acceptance Contract: Add daily note export

## Feature ID
`C03_daily_note_export`

## Goal
Add daily note export for Phase C Obsidian archive/export.

## In Scope
Implement only this Phase C ticket; reuse A/B briefing surfaces; tests first; offline normal tests; preserve wiki-links; enforce vault path safety; write proof pack.

## Out of Scope
Phase D+ functionality, dossiers, feedback learning, social sources, MCP expansion, UI, automatic source expansion, new dependencies unless justified.

## Expected Owned Files
- `src/research_pipeline/briefing/obsidian_daily.py`
- `tests/unit/test_briefing_obsidian_daily.py`

## Required Tests
- `tests/unit/test_briefing_obsidian_daily.py`

## Required Fixtures
- `Add focused fixtures only if required by the test design.`

## Failure Cases To Cover
Missing vault config; unsafe path traversal; existing human note without generated ID; invalid frontmatter; missing headings; broken/mutated wiki-links; repeated export idempotency; no network in normal tests.

## Verification Commands
```bash
uv run pytest tests/unit/test_briefing_obsidian_daily.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; no later-phase functionality; vault path safety enforced where applicable; wiki-links preserved where applicable; generated-note ownership respected where applicable; status/proof pack updated.
