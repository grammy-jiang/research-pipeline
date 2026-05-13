# Phase G Acceptance Gate Addendum

Append to `acceptance-gates.md`.

## Phase G Gate

G01-G09 audit_pass; no pending/in_progress/failed/reopened/blocked G tickets; A-F tests still pass; workflow state is replayable; stage verifiers run; MCP `brief_*` schemas/tools/resources mirror stable CLI behavior; tool annotations reflect network/write behavior; dedicated skill trigger/non-trigger rules work; held-out agent evals pass; final all-phase acceptance gate passes.

Required commands:

```bash
uv run pytest tests/unit/test_briefing_workflow_state.py tests/unit/test_briefing_workflow_verification.py -xvs
uv run pytest tests/unit/test_mcp_briefing_schemas.py tests/unit/test_mcp_briefing_tools.py tests/unit/test_mcp_briefing_resources.py -xvs
uv run pytest tests/unit/test_skill_daily_ai_intelligence.py tests/unit/test_briefing_tool_governance.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_g_agent_evals.py tests/integration_offline/test_briefing_phase_g_replay_diagnosis.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_a_e2e.py tests/integration_offline/test_briefing_phase_b_memory_e2e.py tests/integration_offline/test_briefing_phase_c_obsidian_e2e.py tests/integration_offline/test_briefing_phase_d_feedback_e2e.py tests/integration_offline/test_briefing_phase_e_dossier_e2e.py tests/integration_offline/test_briefing_phase_f_source_expansion_e2e.py -xvs
uv run ruff format .
uv run ruff check . --fix
uv run mypy src/
```
