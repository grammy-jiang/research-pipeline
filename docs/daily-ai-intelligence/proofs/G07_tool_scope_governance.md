# Proof Pack: G07_tool_scope_governance

## Ticket
`G07_tool_scope_governance`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_tool_governance.py -xvs
uv run ruff check src/research_pipeline/briefing src/research_pipeline/mcp_server tests/
uv run mypy src/research_pipeline/briefing/tool_governance.py
```

## Result
PASS — 11/11 tests pass; ruff clean; mypy clean.

## Harness Safety Evidence
- `ToolPolicy` frozen Pydantic model encodes capability per tool: `kind` (network|local), `effects`, `deterministic`, `source_allowlisted`, `write_paths`, `notes`.
- `_POLICIES` covers all 9 `brief_*` tools.
- `brief_poll_sources_tool` and `brief_run_tool` are `kind="network"` and `source_allowlisted=True` (Phase G non-goal: no new source expansion).
- `brief_rank_events_tool`, `brief_validate_report_tool`, `brief_generate_daily_tool`, `brief_generate_dossier_tool`, `brief_weekly_synthesis_tool` are `kind="local"` and `deterministic=True`.
- `brief_export_obsidian_tool` and `brief_record_feedback_tool` are local writes confined to allowlisted paths.
- `is_unsupported_source(url, allowlist)` enables agent-side refusal.
- A-F governance unchanged.

## Next Ticket
`G08_held_out_agent_evals`
