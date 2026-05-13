# Proof Pack: Video / Audio Source

## Ticket
`F08_video_audio`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_video_audio.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS

- `tests/unit/test_briefing_video_audio.py`: 3 tests passed (YouTube Atom parsed with `item_type="video"`, `source_native_id="abc123XYZ"`, author "Example AI Channel", `confidence="medium"`; wrong access method rejected; disabled by default).
- ruff and mypy clean.

## Source Safety Evidence
- `src/research_pipeline/briefing/sources/video_audio.py` consumes YouTube Atom (`yt:videoId` namespace) and podcast RSS via `feed_url` / `api_url` only.
- Requires `AccessMethod.VIDEO_AUDIO`; rejected otherwise.
- Marked `SourcePolicy.PUBLIC_OFFICIAL`, `confidence="medium"`, `evidence_type="speculation_or_watch_item"`.
- Offline fixture `tests/fixtures/briefing/video_audio/youtube_feed.xml` drives tests; no live HTTP calls.
- Disabled by default in registry templates; 30-second timeout enforced when no fixture is provided.
- Reads only public Atom/RSS feeds (channel uploads, podcast RSS) — no transcript scraping, no auto-download, no browser automation, no firehose.

## Next Ticket
`F09_source_expansion_offline_e2e`
