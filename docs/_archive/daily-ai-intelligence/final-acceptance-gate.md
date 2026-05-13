# Final Acceptance Gate

Run after G09 and after the final traceability matrix and gap register are written.

## Required commands

```bash
uv run pytest tests/unit/test_briefing_*.py -xvs
uv run pytest tests/unit/test_mcp_briefing_schemas.py tests/unit/test_mcp_briefing_tools.py tests/unit/test_mcp_briefing_resources.py -xvs
uv run pytest tests/unit/test_skill_daily_ai_intelligence.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_a_e2e.py tests/integration_offline/test_briefing_phase_b_memory_e2e.py tests/integration_offline/test_briefing_phase_c_obsidian_e2e.py tests/integration_offline/test_briefing_phase_d_feedback_e2e.py tests/integration_offline/test_briefing_phase_e_dossier_e2e.py tests/integration_offline/test_briefing_phase_f_source_expansion_e2e.py tests/integration_offline/test_briefing_phase_g_agent_evals.py tests/integration_offline/test_briefing_phase_g_replay_diagnosis.py -xvs
uv run ruff format .
uv run ruff check . --fix
uv run mypy src/
uv run pre-commit run --all-files
```

## Required documents

- `docs/daily-ai-intelligence/final-traceability-matrix.md`
- `docs/daily-ai-intelligence/final-gap-register.md`
- `docs/daily-ai-intelligence/final-completeness-audit-report.md`

## Completion condition

The final gate passes only when:

- all commands pass;
- gap register is empty;
- traceability matrix marks all planned features pass or justified not-applicable;
- no A-G governance rule is weakened.
