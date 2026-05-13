# Proof Pack: D07_feedback_rollback_and_audit

## Ticket
`D07_feedback_rollback_and_audit`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_feedback_audit.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 11/11 unit tests passing; ruff and mypy --strict clean.

## Feedback Safety Evidence
- Explicit feedback only — `audit_feedback_sufficiency(events, ...)` operates on `FeedbackEvent` records only; signal classification uses the explicit POSITIVE/NEGATIVE sets.
- Before/after weights recorded — `audit_promotion_record(record)` requires `before_weight` and `after_weight` and rejects records where they are equal ("no observable change").
- Rollback metadata exists — `safe_rollback(db_path, adjustment_id, *, review_record="...")` calls `rollback_preference_adjustment` and returns a `rollback_envelope` containing `trigger`, `procedure="rollback_preference_adjustment_v1"`, `observable_effect`, `rollback`, and `review_record`.
- Conflicting/insufficient feedback does not silently change ranking — `audit_feedback_sufficiency` returns `(False, "insufficient feedback: N < min")` or `(False, "conflicting positive and negative feedback on same target")` when the gate fails; per-target conflict detection covers mixed-target batches.
- No behavioral signals used — `REQUIRED_PROMOTION_KEYS` does not include any behavioural fields; audit-time checks only validate explicit-feedback metadata.

## Next Ticket
`D08_feedback_weekly_section_and_e2e`
