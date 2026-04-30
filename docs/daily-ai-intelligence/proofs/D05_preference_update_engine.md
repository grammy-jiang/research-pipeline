# Proof Pack: D05_preference_update_engine

## Ticket
`D05_preference_update_engine`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_preference_update.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 6/6 unit tests passing; ruff and mypy --strict clean.

## Feedback Safety Evidence
- Explicit feedback only — `apply_preference_updates(store, ...)` only consumes rows from `feedback_events`; no behavioural sources.
- Before/after weights recorded — every promoted row carries `before_weight`, `after_weight`, `procedure="explicit_feedback_promotion_v1"`, `observable_effect`, `trigger`, `rollback`, `review_record`.
- Rollback metadata exists — `rollback_preference_adjustment(db_path, adjustment_id)` removes the durable change and returns a rollback receipt.
- Conflicting/insufficient feedback does not silently change ranking — when `len(feedback) < min_feedback` the function returns `[]`; when a target has both positive and negative feedback the row is created with `conflict` set and is then rolled back in the same call so no durable change persists.
- No behavioral signals used — promotion key whitelist is `target_type` ∈ {event, cluster, topic, source, dossier}; no behavioural targets.

## Next Ticket
`D06_feedback_adjusted_ranking`
