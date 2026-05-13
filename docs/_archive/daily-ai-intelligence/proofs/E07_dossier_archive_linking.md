# Proof Pack: E07_dossier_archive_linking

## Ticket
`E07_dossier_archive_linking`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_dossier_linking.py -xvs
uv run ruff check src/research_pipeline/briefing/report.py src/research_pipeline/briefing/dossier.py
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 7/7 unit tests passing; ruff and mypy --strict clean.

## Dossier Safety Evidence
- Primary artifact gate enforced — archive writes (`write_dossier_with_archive`) only execute after `build_dossier` has accepted the cluster; archived bytes are byte-for-byte identical to the primary artifact.
- One-topic focus preserved — link tuples are `(title, link)` per dossier file; the daily brief renders one bullet per linked dossier with no cross-topic aggregation.
- Evidence URLs or inference/speculation labels present — links point to dossier files whose contents have already passed `validate_dossier_markdown`.
- No automatic dossier scheduling — `render_daily_brief(..., dossier_links=None)` defaults to None and only renders the "## Linked Dossiers" section when explicit links are passed in by the caller.
- No general literature-review expansion — linking is a static reference to existing manual dossiers; no search expansion is performed.

## Next Ticket
`E08_dossier_offline_e2e`
