# Proof Pack: D06_feedback_adjusted_ranking

## Ticket
`D06_feedback_adjusted_ranking`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_rank_feedback.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 7/7 unit tests passing; ruff and mypy --strict clean.

## Feedback Safety Evidence
- Explicit feedback only — `RankingOptions.feedback_weights` keys are constrained to the explicit `cluster:|topic:|source:|event:` namespace produced by D02 `weights_by_target`.
- Before/after weights recorded — `explicit_feedback_components(cluster, weights)` returns the `(topic_adj, source_adj, negative_penalty)` triple consumed by ranking, enabling deterministic before/after deltas at audit time.
- Rollback metadata exists — rollback flows through D05/D07 to remove preference rows; the next ranking pass picks up the cleared weights automatically.
- Conflicting/insufficient feedback does not silently change ranking — only adjustments that survive the D05 gate end up in `weights_by_target()`; conflicting rows are rolled back before they reach the ranker.
- No behavioral signals used — `rank_clusters` only accepts the explicit `feedback_weights` dict; there is no behavioural-bonus path. The new `explicit_feedback_components` helper enforces the formula `rank_score = phase_c_rank_score + topic_adj + source_adj − negative_penalty`.

## Next Ticket
`D07_feedback_rollback_and_audit`
