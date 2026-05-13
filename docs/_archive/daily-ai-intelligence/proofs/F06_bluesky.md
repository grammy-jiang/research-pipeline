# Proof Pack: Bluesky Source

## Ticket
`F06_bluesky`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_bluesky.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS

- `tests/unit/test_briefing_bluesky.py`: 3 tests passed (feed parsed with author handle + cid; wrong access method; disabled by default).
- ruff and mypy clean.

## Source Safety Evidence
- `src/research_pipeline/briefing/sources/bluesky.py` requires `AccessMethod.BLUESKY_API` (official AT-Protocol API).
- Marked `SourcePolicy.DISCUSSION_ONLY`, `confidence="low"`, `evidence_type="speculation_or_watch_item"`.
- Offline fixture `tests/fixtures/briefing/bluesky/feed.json` drives all tests; no live HTTP.
- Disabled by default in registry templates (verified by `test_bluesky_source_disabled_by_default`).
- Title bounded to 160 chars; excerpt to 500 chars; rate limit and `max_events_per_run` honored.
- No browser scraping; no firehose.

## Next Ticket
`F07_x_api_policy_stub`
