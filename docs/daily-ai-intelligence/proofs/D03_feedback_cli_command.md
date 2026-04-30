# Proof Pack: D03_feedback_cli_command

## Ticket
`D03_feedback_cli_command`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_cli_feedback.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 6/6 CLI tests passing; ruff and mypy --strict clean.

## Feedback Safety Evidence
- Explicit feedback only — `brief feedback` requires exactly one of `--cluster|--topic|--source|--event|--dossier` and one explicit signal; raises `typer.BadParameter` otherwise.
- Before/after weights recorded — N/A at the CLI surface; persisted by the store and surfaced by D05.
- Rollback metadata exists — `--show` lists adjustments, `--conflicts` lists conflicting targets; rollback is performed by D07 `safe_rollback`.
- Conflicting/insufficient feedback does not silently change ranking — CLI never auto-promotes; it only records explicit signals into the store. Ranking adjustments are governed by D05 thresholds.
- No behavioral signals used — the `--signal` flag accepts only `more_like_this`, `less_like_this`, `seen_already`, etc. — no dwell/click/scroll values.

## Next Ticket
`D04_manual_review_labels_as_feedback`
