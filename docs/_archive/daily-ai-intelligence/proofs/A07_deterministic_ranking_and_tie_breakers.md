# Proof Pack: Add deterministic ranking and tie-breakers

## Ticket

`A07_deterministic_ranking_and_tie_breakers`

## Implemented Files

- `docs/daily-ai-intelligence/acceptance/A07_deterministic_ranking_and_tie_breakers.md`
- `src/research_pipeline/briefing/rank.py`
- `tests/unit/test_briefing_rank.py`

## Acceptance Contract

`docs/daily-ai-intelligence/acceptance/A07_deterministic_ranking_and_tie_breakers.md`

## Verification Commands Run

```bash
uv run pytest tests/unit/test_briefing_rank.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result

PASS

## Evidence

- `uv run pytest tests/unit/test_briefing_rank.py -xvs`:
  - collected 5 tests
  - 5 passed
- `uv run ruff check src/research_pipeline/briefing tests/`:
  - All checks passed
- `uv run mypy src/research_pipeline/briefing`:
  - Success: no issues found in 27 source files

## Notes

- Added deterministic unit coverage for score ordering, low-information filtering,
  tie-break consistency, and `min_rank_score`/`max_items` handling.
- Updated ranking tie-breaker behavior so newer `published_at` values sort ahead
  when score and source-class tiers are equal.

## Status Update

- `phase-status.yaml` updated: yes
- new status: `verified`

## Next Ticket

`A08_markdown_daily_report_renderer`
