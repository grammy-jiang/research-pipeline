# Proof Pack: Source Evaluation Harness

## Ticket
`F01_source_evaluation_harness`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_source_evaluation.py -xvs
uv run pytest tests/unit/test_briefing_report_comparison.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS

- `tests/unit/test_briefing_source_evaluation.py`: 9 tests passed.
- `tests/unit/test_briefing_report_comparison.py`: 8 tests passed.
- ruff: All checks passed.
- mypy `src/research_pipeline/briefing`: Success: no issues found in 51 source files.

## Source Safety Evidence
- `evaluate_registry(registry)` validates each `BriefingSourceConfig` against governance rules: required policy fields (`access_method`, `source_policy`), retention/cadence/rate-limit, disabled-by-default for new sources, and X_API policy gate (`auth_required=True`, `last_reviewed_at`, `policy_gate_passed` tag).
- `compare_reports(baseline_md, candidate_md, *, max_item_growth_ratio=1.5, max_link_growth_ratio=2.0)` produces `ReportComparisonResult` with `noise_increase: bool` and `coverage_increase: bool`, blocking unbounded growth between baseline and candidate registries.
- Harness is offline-only and operates on local registry files / rendered markdown; no network access.
- No browser scraping; no firehose. The harness is the gate, not a source.

## Next Ticket
`F02_paper_events`
