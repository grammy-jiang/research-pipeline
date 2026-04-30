# Proof Pack: Paper Events Source

## Ticket
`F02_paper_events`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_paper_events.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS

- `tests/unit/test_briefing_paper_events.py`: 10 tests passed.
- ruff and mypy clean.

## Source Safety Evidence
- New adapter `PaperEventsSource` (`src/research_pipeline/briefing/sources/papers.py`) consumes structured arXiv (`.jsonl`) and Hugging Face daily-papers (`.json`) payloads via `map_arxiv_paper` / `map_hf_paper`.
- Access methods reuse the sanctioned `AccessMethod.ARXIV_API` and `AccessMethod.HUGGINGFACE_PAPERS` paths; no new firehose surface introduced.
- Offline fixtures under `tests/fixtures/briefing/papers/` drive every code path — no live HTTP calls in tests.
- Source adapter remains disabled-by-default in registries until explicit `enabled = true` is set per source.
- Workflow `_adapter_for` dispatches to `PaperEventsSource` only when the registry entry's fixture path ends in `.jsonl`/`.json`, preserving back-compat with existing arXiv/HF Atom adapters.
- No browser scraping.

## Next Ticket
`F03_academic_enrichment`
