# Proof Pack: Hacker News Refresh

## Ticket
`F04_hacker_news_refresh`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_hacker_news.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS

- `tests/unit/test_briefing_hacker_news.py`: 5 tests passed.
- ruff and mypy clean.

## Source Safety Evidence
- Refreshed `HackerNewsSource` honors registry-driven `enabled`, `max_events_per_run`, and rate-limit policy.
- Marked as `SourcePolicy.DISCUSSION_ONLY`, `confidence="low"`, `evidence_type="speculation_or_watch_item"` — discussion items never escalate above watch-item evidence.
- Offline fixtures under `tests/fixtures/briefing/hacker_news/` cover normal/empty/malformed responses.
- Disabled by default in registry templates.
- Uses official Firebase `hn.algolia` API endpoints — no browser scraping; no firehose.

## Next Ticket
`F05_reddit`
