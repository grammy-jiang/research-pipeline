# Feature Acceptance Contract: Add evidence timeline from cluster events and topic memory

## Feature ID
`E05_evidence_timeline`

## Goal
Add evidence timeline from cluster events and topic memory for Phase E hot-topic dossiers.

## In Scope
Manual dossier workflow only; primary artifact gate; one-topic focus; evidence timeline; factuality labels; validation; tests first; offline tests; proof pack.

## Out of Scope
Automatic scheduling, Phase F source expansion, social sources, MCP expansion, UI, general literature review, raw source dump summarization.

## Expected Owned Files
- `src/research_pipeline/briefing/dossier_timeline.py`
- `tests/unit/test_briefing_dossier_timeline.py`

## Required Tests
- `tests/unit/test_briefing_dossier_timeline.py`

## Failure Cases
Missing cluster; missing primary artifact; missing evidence URL; unsupported factuality label; overlong dossier; multi-topic expansion; no network in normal tests.

## Verification Commands
```bash
uv run pytest tests/unit/test_briefing_dossier_timeline.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; manual dossier only; primary artifact required; claims labeled; validator passes/fails correctly; status and proof pack updated.
