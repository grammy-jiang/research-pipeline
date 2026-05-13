# Proof Pack: D08_feedback_weekly_section_and_e2e

## Ticket
`D08_feedback_weekly_section_and_e2e`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_weekly_feedback.py tests/integration_offline/test_briefing_phase_d_feedback_e2e.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 4 unit tests + 2 offline e2e tests passing; ruff and mypy --strict clean. No network calls.

## Feedback Safety Evidence
- Explicit feedback only — `render_feedback_section()` reads `BriefingFeedbackStore.list_feedback()` and `weights_by_target()`; tests assert no behavioural terms (`dwell`, `click`, `behavioural`, `behavioral`, `tracking`) appear in the rendered Markdown.
- Before/after weights recorded — the e2e test asserts `audit_promotion_record(record)` returns `ok=True` for the produced adjustment, confirming the full envelope is intact end-to-end.
- Rollback metadata exists — the e2e test calls `safe_rollback(db, adjustment_id)` and verifies the `preference_adjustments` table is empty afterwards.
- Conflicting/insufficient feedback does not silently change ranking — `test_phase_d_conflicting_feedback_does_not_promote` records 2 positive + 2 negative events on the same target, calls `apply_preference_updates(min_feedback=3)`, and asserts the result is `[]` and no rows persist.
- No behavioral signals used — the renderer references only signal-name counts and target-key weights; no clickstream or dwell columns are read.

## Next Ticket
`none` — Phase D complete, Phase E next.
