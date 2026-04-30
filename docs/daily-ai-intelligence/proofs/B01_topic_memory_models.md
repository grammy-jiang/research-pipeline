# Proof Pack: Add topic memory models

## Ticket

`B01_topic_memory_models`

## Implemented Files

- `docs/daily-ai-intelligence/acceptance/B01_topic_memory_models.md`
- `src/research_pipeline/briefing/models.py`
- `src/research_pipeline/briefing/__init__.py`
- `tests/unit/test_briefing_topic_memory.py`
- `docs/daily-ai-intelligence/phase-status.yaml`

## Acceptance Contract

`docs/daily-ai-intelligence/acceptance/B01_topic_memory_models.md`

## Verification Commands Run

```bash
uv run pytest tests/unit/test_briefing_topic_memory.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result

PASS

## Evidence

- `uv run pytest tests/unit/test_briefing_topic_memory.py -xvs`
  - collected 5 tests
  - 5 passed
- `uv run ruff check src/research_pipeline/briefing tests/`
  - all checks passed
- `uv run mypy src/research_pipeline/briefing`
  - success: no issues found in 28 source files

## Memory Safety Evidence

- Added `TopicMemoryWriteRecord` to require `trigger`, `effect`, `rollback`, source IDs, timestamp, and owning ticket/command before a durable memory write record can validate.
- Added `TopicAliasSuggestion` validation so approved or rejected suggestions must carry a `review_record`; pending suggestions may remain reviewable without one.
- Kept `TopicMemory` immutable and JSON-roundtrip safe with explicit unit coverage for construct → serialize → deserialize equality.

## Known Limitations

- B01 defines only the model-layer contract; it does not migrate the existing SQLite store to persist the expanded write-record metadata.
- The current `src/research_pipeline/briefing/topic_memory.py` file remains store-oriented and will need reconciliation under B02.

## Status Update

- `phase-status.yaml` updated: yes
- new status: `verified`

## Next Ticket

`B02_topic_memory_store`
