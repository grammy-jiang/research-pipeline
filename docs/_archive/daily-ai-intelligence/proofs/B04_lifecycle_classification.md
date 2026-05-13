# B04 Proof Pack - Lifecycle Classification

## Scope Delivered

Implemented deterministic, read-only topic lifecycle classification for briefing clusters in:

- src/research_pipeline/briefing/lifecycle.py
- tests/unit/test_briefing_lifecycle.py

## Acceptance Coverage

Covered required lifecycle outcomes with explicit tests:

- missing memory -> `new`
- empty store -> `new`
- repeated low-novelty topic -> `cooling`
- dormant topic with fresh strong evidence -> `resurfaced`
- stale topic without strong evidence -> `dormant`
- explicit topic ID evidence overrides stale title-only ambiguity
- classification path is read-only (no topic-memory or alias-suggestion mutation)

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_lifecycle.py -xvs
uv run ruff check src/research_pipeline/briefing/lifecycle.py tests/unit/test_briefing_lifecycle.py
uv run mypy src/research_pipeline/briefing/lifecycle.py
```

## Verification Results

- `pytest`: 7 collected, 7 passed
- `ruff`: all checks passed on touched files
- `mypy`: no issues in touched lifecycle module

## Notes

- Classification is pure/read-only and does not perform durable writes.
- Resolution order preserves explicit topic ID authority before title/alias fallback.
