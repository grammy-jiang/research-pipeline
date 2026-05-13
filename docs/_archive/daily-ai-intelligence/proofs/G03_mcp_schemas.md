# Proof Pack: G03_mcp_schemas

## Ticket
`G03_mcp_schemas`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_mcp_briefing_schemas.py -xvs
uv run ruff check src/research_pipeline/briefing src/research_pipeline/mcp_server tests/
uv run mypy src/
```

## Result
PASS — 12/12 tests pass; ruff clean; mypy clean.

## Harness Safety Evidence
- 10 namespaced `Brief*Input` Pydantic schemas exist on `mcp_server.schemas`.
- All inherit from `BriefCommonInput` (workspace + date defaults).
- Required fields raise `ValidationError` when missing (vault_path, target_type/id/signal, cluster_id, week).
- No collision with academic-paper schemas.
- A-F governance unchanged.

## Next Ticket
`G04_mcp_tools`
