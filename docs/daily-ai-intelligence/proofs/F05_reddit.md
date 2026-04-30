# Proof Pack: Reddit Source

## Ticket
`F05_reddit`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_reddit.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS

- `tests/unit/test_briefing_reddit.py`: 3 tests passed (listing parsed; wrong access method rejected; disabled by default).
- ruff and mypy clean.

## Source Safety Evidence
- `src/research_pipeline/briefing/sources/reddit.py` requires `AccessMethod.REDDIT_API` (sanctioned API access; rejected at construction otherwise).
- Marked `SourcePolicy.DISCUSSION_ONLY`, `confidence="low"`, `evidence_type="speculation_or_watch_item"`. Subreddit / score / num_comments retained in `raw_metadata` only — never escalated.
- Offline fixture `tests/fixtures/briefing/reddit/listing.json` exercises full happy path; no live HTTP calls in tests.
- Source disabled by default unless explicitly approved (`enabled=False` default; verified by `test_reddit_source_disabled_by_default`).
- No browser scraping. No firehose; respects `max_events_per_run` and per-source rate limits.

## Next Ticket
`F06_bluesky`
