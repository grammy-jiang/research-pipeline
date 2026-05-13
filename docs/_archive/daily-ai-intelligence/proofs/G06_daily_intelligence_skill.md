# Proof Pack: G06_daily_intelligence_skill

## Ticket
`G06_daily_intelligence_skill`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_skill_daily_ai_intelligence.py -xvs
uv run ruff check src/research_pipeline/briefing src/research_pipeline/mcp_server tests/
uv run mypy src/
```

## Result
PASS — 10/10 tests pass; ruff clean; mypy clean.

## Harness Safety Evidence
- `SKILL.md` frontmatter contains required keys: `name`, `description`, `license`, `compatibility`.
- Trigger phrases and non-trigger handoff rules present (paper-only requests are handed off to academic-paper-research).
- Explicit prohibition on sending raw / unclustered source dumps to cloud models.
- Preference for `brief_*` MCP tools or `research-pipeline brief` CLI.
- `config.toml` parses with `tomllib`.
- All five reference docs exist; `agent-evaluation.md` includes machine-readable task IDs (`run_daily_brief`, `validate_malformed_report`, `record_feedback`, `export_obsidian`, `refuse_unsupported_source`, `generate_dossier`, `paper_request_handoff`).
- A-F governance unchanged.

## Next Ticket
`G07_tool_scope_governance`
