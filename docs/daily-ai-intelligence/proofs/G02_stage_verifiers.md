# Proof Pack: G02_stage_verifiers

## Ticket
`G02_stage_verifiers`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_workflow_verification.py -xvs
uv run ruff check src/research_pipeline/briefing src/research_pipeline/mcp_server tests/
uv run mypy src/research_pipeline/briefing/workflow_verification.py
```

## Result
PASS — 19/19 tests pass; ruff clean; mypy clean.

## Harness Safety Evidence
- Frozen `StageVerification` model with sorted `issues` tuple.
- `_REGISTRY` covers all six pipeline stages: planned, polled, ranked, generated, validated, archived.
- Public API: `stage_verifiers()`, `get_stage_verifier()`, `verify_stage()`, `verify_completed_stages()`.
- Replay scenario (G09) uses `verify_completed_stages` to detect artifact drift.
- A-F governance unchanged.

## Next Ticket
`G03_mcp_schemas`
