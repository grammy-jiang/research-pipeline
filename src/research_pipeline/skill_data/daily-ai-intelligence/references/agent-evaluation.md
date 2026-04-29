# Held-out agent evaluation tasks

Use these tasks to evaluate whether an agent can operate the briefing workflow
without confusing it with academic-paper research.

1. `run_daily_brief` — Run a thin daily brief from a reviewed registry and validate the output.
2. `validate_malformed_report` — Validate a malformed report and explain the deterministic failures.
3. `record_feedback` — Record explicit feedback for one cluster and compute reversible preferences.
4. `export_obsidian` — Export a validated daily brief to an Obsidian vault path.
5. `refuse_unsupported_source` — Refuse unsupported source expansion without a registry review record.
6. `generate_dossier` — Generate one manual dossier from a primary-artifact cluster.
7. `paper_request_handoff` — Hand off a paper-only literature review request to the academic workflow.

Held-out fixtures live under `tests/fixtures/briefing/e2e/agent_eval/<task>/`.

Compare traces with the same model/runtime before and after harness changes.
