# Feature Acceptance Contract: Add Hacker News discussion source adapter, disabled by default

## Feature ID
`F04_hacker_news_source`

## Goal
Add Hacker News discussion source adapter, disabled by default for Phase F source expansion.

## In Scope
One source/governance unit only; registry policy; disabled-by-default behavior; offline fixtures; parser/evaluation tests; no network in normal tests; proof pack.

## Out of Scope
Browser scraping, social firehose, automatic source enablement, Phase G MCP/skill hardening, UI, behavioral tracking.

## Expected Owned Files
- `src/research_pipeline/briefing/sources/hacker_news.py`
- `tests/unit/test_briefing_hacker_news.py`
- `tests/fixtures/briefing/hn/item.json`
- `tests/fixtures/briefing/hn/thread.json`

## Required Tests
- `tests/unit/test_briefing_hacker_news.py`

## Failure Cases
Missing registry entry; enabled-by-default noisy source; missing retention policy; missing cadence/rate-limit policy; malformed fixture; unsupported access method; report bloat; no side-by-side comparison where applicable.

## Verification Commands
```bash
uv run pytest tests/unit/test_briefing_hacker_news.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; no scraping; source disabled by default where applicable; registry/retention/cadence/rate-limit policy defined; status and proof pack updated.
