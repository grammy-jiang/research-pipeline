# Proof Pack: Source Expansion Offline E2E

## Ticket
`F09_source_expansion_offline_e2e`

## Verification Commands Run
```bash
uv run pytest tests/integration_offline/test_briefing_phase_f_source_expansion_e2e.py -xvs
uv run pytest tests/unit/ tests/integration_offline/ -q
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS

- `tests/integration_offline/test_briefing_phase_f_source_expansion_e2e.py`: 3 tests passed (baseline runs; candidate runs and grows coverage with `noise_increase=False`; candidate registry passes governance).
- Full regression: `tests/unit/` + `tests/integration_offline/`: **4301 passed, 1 skipped** in 39.28s.
- ruff: All checks passed.
- mypy `src/research_pipeline/briefing`: Success: no issues found in 51 source files.

## Source Safety Evidence
- E2E test compares baseline registry (RSS only) vs. candidate registry (RSS + Bluesky) using F01's `compare_reports` and `evaluate_registry` harness, asserting `comparison.noise_increase is False` and every `SourceEvaluationResult.passed`.
- Side-by-side report comparison confirms candidate adds coverage without breaching `max_item_growth_ratio=1.5` / `max_link_growth_ratio=2.0`.
- All inputs are local fixtures under `tests/fixtures/briefing/e2e/source_expansion/` (`feed.xml`, `bluesky.json`, `registry_baseline.toml`, `registry_candidate.toml`); no network.
- Candidate Bluesky source has `last_reviewed_at = "2026-04-30"` and `tags = ["policy_gate_passed"]` — gates enforced by F01 governance.
- No browser scraping; no firehose. Demonstrates the full Phase F flow under disable-by-default + explicit-approval discipline.

## Next Ticket
`G01` (Phase G — remains blocked until explicit unblock; do not start)
