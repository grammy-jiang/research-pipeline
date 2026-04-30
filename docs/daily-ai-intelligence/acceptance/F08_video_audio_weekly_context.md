# Feature Acceptance Contract: Add YouTube/podcast weekly-context source adapter, disabled by default

## Feature ID
`F08_video_audio_weekly_context`

## Goal
Add YouTube/podcast weekly-context source adapter, disabled by default for Phase F source expansion.

## In Scope
One source/governance unit only; registry policy; disabled-by-default behavior; offline fixtures; parser/evaluation tests; no network in normal tests; proof pack.

## Out of Scope
Browser scraping, social firehose, automatic source enablement, Phase G MCP/skill hardening, UI, behavioral tracking.

## Expected Owned Files
- `src/research_pipeline/briefing/sources/video_audio.py`
- `tests/unit/test_briefing_video_audio.py`
- `tests/fixtures/briefing/video_audio/youtube_feed.xml`

## Required Tests
- `tests/unit/test_briefing_video_audio.py`

## Failure Cases
Missing registry entry; enabled-by-default noisy source; missing retention policy; missing cadence/rate-limit policy; malformed fixture; unsupported access method; report bloat; no side-by-side comparison where applicable.

## Verification Commands
```bash
uv run pytest tests/unit/test_briefing_video_audio.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; no scraping; source disabled by default where applicable; registry/retention/cadence/rate-limit policy defined; status and proof pack updated.
