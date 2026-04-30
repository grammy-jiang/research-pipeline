# Proof Pack: Extend ranking with fatigue penalty and resurfaced-topic boost

## Ticket

`B05_ranking_fatigue_and_resurfaced_boost`

## Implemented Files

- `docs/daily-ai-intelligence/acceptance/B05_ranking_fatigue_and_resurfaced_boost.md`
- `src/research_pipeline/briefing/rank.py`
- `tests/unit/test_briefing_rank_memory.py`

## Acceptance Contract

`docs/daily-ai-intelligence/acceptance/B05_ranking_fatigue_and_resurfaced_boost.md`

## Verification Commands Run

```bash
uv run pytest tests/unit/test_briefing_rank_memory.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result

PASS

## Evidence

- `uv run pytest tests/unit/test_briefing_rank_memory.py -xvs`
  - collected 4 tests
  - 4 passed
- `uv run ruff check src/research_pipeline/briefing tests/`
  - all checks passed
- `uv run mypy src/research_pipeline/briefing`
  - success: no issues found in 30 source files

## Memory Safety Evidence

- Ranking remains read-only and does not create or mutate durable topic-memory records.
- Repeated topics now classify as `cooling` once fatigue reaches the same threshold used by ranking suppression behavior.
- Resurfaced topics continue to receive deterministic positive boost without allowing stale memory to override current evidence.

## Known Limitations

- B05 verifies ranking-time memory adjustments only; prior-context rendering remains deferred to B06.
- The verification run emitted an existing mypy note about an unused `research_pipeline.mcp_server.*` config section in `pyproject.toml`; this is unrelated to B05.

## Status Update

- `phase-status.yaml` updated: yes
- new status: `verified`

## Next Ticket

`B06_report_prior_context`
