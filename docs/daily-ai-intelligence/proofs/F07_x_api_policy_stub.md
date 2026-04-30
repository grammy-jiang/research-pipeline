# Proof Pack: X API Policy Stub

## Ticket
`F07_x_api_policy_stub`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_x_api.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS

- `tests/unit/test_briefing_x_api.py`: 6 tests passed (disabled refused; missing auth_required refused; missing last_reviewed_at refused; missing policy tag refused; all gates pass returns `[]`; wrong `access_method` raises `ValueError`).
- ruff and mypy clean.

## Source Safety Evidence
- `src/research_pipeline/briefing/sources/x_api.py` is a **policy stub**: `poll()` raises `XApiPolicyError` unless ALL four gates pass:
  1. `source.enabled is True`
  2. `source.auth_required is True`
  3. `source.last_reviewed_at` is set
  4. `"policy_gate_passed"` is in `source.tags`
- When all gates pass, `poll()` returns `[]` — live polling is deliberately deferred.
- Construction enforces `access_method == AccessMethod.X_API` (rejected otherwise).
- No fixture coupling — `validate_access_fields` excludes `X_API` from fixture/url/query requirements.
- F01 governance harness `evaluate_registry` enforces the same gates at registry-load time.
- No browser scraping; no firehose. Implements explicit deny-by-default for an unsanctioned firehose surface.

## Next Ticket
`F08_video_audio`
