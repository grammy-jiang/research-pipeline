# Feature Acceptance Contract: Add targeted Bluesky/AT Protocol source adapter, disabled by default

## Feature ID
`F06_bluesky_source`

## Goal
Add targeted Bluesky/AT Protocol source adapter, disabled by default for Phase F source expansion.

## In Scope
One source/governance unit only; registry policy; disabled-by-default behavior; offline fixtures; parser/evaluation tests; no network in normal tests; proof pack.

## Out of Scope
Browser scraping, social firehose, automatic source enablement, Phase G MCP/skill hardening, UI, behavioral tracking.

## Expected Owned Files
- `src/research_pipeline/briefing/sources/bluesky.py`
- `tests/unit/test_briefing_bluesky.py`
- `tests/fixtures/briefing/bluesky/feed.json`

## Required Tests
- `tests/unit/test_briefing_bluesky.py`

## Failure Cases
Missing registry entry; enabled-by-default noisy source; missing retention policy; missing cadence/rate-limit policy; malformed fixture; unsupported access method; report bloat; no side-by-side comparison where applicable.

## Verification Commands
```bash
uv run pytest tests/unit/test_briefing_bluesky.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; no scraping; source disabled by default where applicable; registry/retention/cadence/rate-limit policy defined; status and proof pack updated.
