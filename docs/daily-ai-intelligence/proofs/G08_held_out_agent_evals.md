# Proof Pack: G08_held_out_agent_evals

## Ticket
`G08_held_out_agent_evals`

## Verification Commands Run
```bash
uv run pytest tests/integration_offline/test_briefing_phase_g_agent_evals.py -xvs
uv run ruff check src/research_pipeline/briefing src/research_pipeline/mcp_server tests/
uv run mypy src/
```

## Result
PASS — 9/9 tests pass; ruff clean; mypy clean.

## Harness Safety Evidence
Held-out scenarios cover the agent contract:

- `test_run_daily_brief` — agent drives `brief_run_tool` end-to-end.
- `test_validate_malformed_report` — deterministic validation failure.
- `test_record_feedback` — local feedback write only.
- `test_export_obsidian_to_configured_vault` — vault-confined write.
- `test_refuse_unsupported_source` — agent refuses non-allowlisted URLs (substring match against registry allowlist).
- `test_generate_dossier_is_local_and_deterministic` — manual dossier path is local + deterministic.
- `test_paper_request_handoff_documented` — SKILL.md hands paper-only requests to the academic-paper-research skill.
- `test_every_brief_tool_has_governance` — `policy_for(name)` resolves for every `brief_*` tool.
- `test_brief_namespace_does_not_collide_with_academic_research` — namespace separation enforced.

A-F governance unchanged. No new external sources, no UI, no behavioral tracking.

## Next Ticket
`G09_replay_diagnosis_and_final_gate`
