# Final Traceability Matrix

Generated from `docs/daily-ai-intelligence/phase-status.yaml` — all 63 tickets `audit_pass`.

| Feature / Function | Phase | Ticket | Implementation Files | Tests | Surface | Proof Pack | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| Briefing package skeleton and models | A | A01 | `src/research_pipeline/briefing/__init__.py`, `briefing/models.py` | `tests/unit/test_briefing_models.py` | internal | proofs/A01 | pass | |
| Source registry loader and validation | A | A02 | `briefing/registry.py` | `tests/unit/test_briefing_registry.py` | CLI | proofs/A02 | pass | |
| GitHub releases adapter | A | A03 | `briefing/sources/` | `tests/unit/test_briefing_github_releases.py` | internal | proofs/A03 | pass | |
| RSS/Atom adapter | A | A04 | `briefing/sources/` | `tests/unit/test_briefing_rss.py` | internal | proofs/A04 | pass | |
| Normalization and stable ID generation | A | A05 | `briefing/normalize.py` | `tests/unit/test_briefing_normalize.py` | internal | proofs/A05 | pass | |
| Exact deduplication | A | A06 | `briefing/dedup.py` | `tests/unit/test_briefing_dedup.py` | internal | proofs/A06 | pass | |
| Deterministic ranking and tie-breakers | A | A07 | `briefing/rank.py` | `tests/unit/test_briefing_rank.py` | internal | proofs/A07 | pass | |
| Markdown daily report renderer | A | A08 | `briefing/report.py` | `tests/unit/test_briefing_report.py` | CLI | proofs/A08 | pass | |
| Report and event validator | A | A09 | `briefing/validate.py` | `tests/unit/test_briefing_validate.py` | CLI | proofs/A09 | pass | |
| CLI command group and artifact layout | A | A10 | `briefing/layout.py`, `cli/brief_commands.py` | `tests/unit/test_briefing_layout.py` | CLI | proofs/A10 | pass | |
| Telemetry JSONL | A | A11 | `briefing/telemetry.py` | `tests/unit/test_briefing_telemetry.py` | internal | proofs/A11 | pass | |
| Offline e2e tests (normal/low-signal/no-news) | A | A12 | `tests/integration_offline/test_briefing_phase_a_e2e.py` | same | test | proofs/A12 | pass | |
| TopicMemory models | B | B01 | `briefing/topic_memory.py` | `tests/unit/test_briefing_topic_memory.py` | internal | proofs/B01 | pass | |
| SQLite topic memory store | B | B02 | `briefing/topic_memory_store.py` | `tests/unit/test_briefing_topic_memory_store.py` | internal | proofs/B02 | pass | |
| Memory lookup | B | B03 | `briefing/memory_lookup.py` | `tests/unit/test_briefing_memory_lookup.py` | internal | proofs/B03 | pass | |
| Lifecycle classification | B | B04 | `briefing/lifecycle.py` | `tests/unit/test_briefing_lifecycle.py` | internal | proofs/B04 | pass | |
| Fatigue penalty and resurfaced boost | B | B05 | `briefing/rank.py` | `tests/unit/test_briefing_rank_memory.py` | internal | proofs/B05 | pass | |
| Prior-context report references | B | B06 | `briefing/report.py` | `tests/unit/test_briefing_report_prior.py` | CLI | proofs/B06 | pass | |
| Alias/merge review queue | B | B07 | `briefing/topic_review.py` | `tests/unit/test_briefing_topic_review.py` | CLI | proofs/B07 | pass | |
| Memory validation and e2e | B | B08 | `tests/integration_offline/test_briefing_phase_b_memory_e2e.py` | same | test | proofs/B08 | pass | |
| Obsidian config and path allowlist | C | C01 | `briefing/obsidian.py` | `tests/unit/test_briefing_obsidian_config.py` | internal | proofs/C01 | pass | |
| Obsidian note models and YAML frontmatter | C | C02 | `briefing/obsidian_notes.py` | `tests/unit/test_briefing_obsidian_notes.py` | internal | proofs/C02 | pass | |
| Daily note export | C | C03 | `briefing/obsidian_daily.py` | `tests/unit/test_briefing_obsidian_daily.py` | CLI | proofs/C03 | pass | |
| Topic and source note export | C | C04 | `briefing/obsidian_topics.py`, `briefing/obsidian_sources.py` | `tests/unit/test_briefing_obsidian_topics.py` | CLI | proofs/C04 | pass | |
| Wiki-link/backlink handling | C | C05 | `briefing/obsidian_links.py` | `tests/unit/test_briefing_obsidian_links.py` | internal | proofs/C05 | pass | |
| Obsidian export validation | C | C06 | `briefing/validate_obsidian.py` | `tests/unit/test_briefing_validate_obsidian.py` | CLI | proofs/C06 | pass | |
| brief export-obsidian CLI | C | C07 | CLI surface | `tests/unit/test_briefing_cli_obsidian.py` | CLI | proofs/C07 | pass | |
| Obsidian offline e2e | C | C08 | `tests/integration_offline/test_briefing_phase_c_obsidian_e2e.py` | same | test | proofs/C08 | pass | |
| FeedbackEvent models | D | D01 | `briefing/feedback.py` | `tests/unit/test_briefing_feedback_models.py` | internal | proofs/D01 | pass | |
| Feedback store | D | D02 | `briefing/feedback_store.py` | `tests/unit/test_briefing_feedback_store.py` | internal | proofs/D02 | pass | |
| brief feedback CLI | D | D03 | CLI surface | `tests/unit/test_briefing_feedback_cli.py` | CLI | proofs/D03 | pass | |
| Manual review labels as feedback | D | D04 | `briefing/manual_review.py` | `tests/unit/test_briefing_manual_review.py` | CLI | proofs/D04 | pass | |
| Reversible preference updates | D | D05 | `briefing/preference_update.py` | `tests/unit/test_briefing_preference_update.py` | internal | proofs/D05 | pass | |
| Feedback-adjusted ranking | D | D06 | `briefing/rank.py` | `tests/unit/test_briefing_rank_feedback.py` | internal | proofs/D06 | pass | |
| Feedback rollback and audit | D | D07 | `briefing/feedback_audit.py` | `tests/unit/test_briefing_feedback_rollback.py` | CLI | proofs/D07 | pass | |
| Weekly feedback section and e2e | D | D08 | `briefing/weekly.py` | `tests/integration_offline/test_briefing_phase_d_feedback_e2e.py` | CLI | proofs/D08 | pass | |
| TopicDossier models | E | E01 | `briefing/dossier.py` | `tests/unit/test_briefing_dossier_models.py` | internal | proofs/E01 | pass | |
| Single-topic dossier renderer | E | E02 | `briefing/dossier.py` | `tests/unit/test_briefing_dossier_renderer.py` | CLI | proofs/E02 | pass | |
| brief generate-dossier CLI | E | E03 | CLI surface | `tests/unit/test_briefing_dossier_cli.py` | CLI | proofs/E03 | pass | |
| Primary artifact gate | E | E04 | `briefing/dossier.py` | `tests/unit/test_briefing_dossier_gate.py` | internal | proofs/E04 | pass | |
| Evidence timeline | E | E05 | `briefing/dossier_timeline.py` | `tests/unit/test_briefing_dossier_timeline.py` | internal | proofs/E05 | pass | |
| Dossier validation | E | E06 | `briefing/validate_dossier.py` | `tests/unit/test_briefing_validate_dossier.py` | CLI | proofs/E06 | pass | |
| Dossier archive linking | E | E07 | `briefing/dossier.py` | `tests/unit/test_briefing_dossier_links.py` | internal | proofs/E07 | pass | |
| Dossier offline e2e | E | E08 | `tests/integration_offline/test_briefing_phase_e_dossier_e2e.py` | same | test | proofs/E08 | pass | |
| Source evaluation harness | F | F01 | `briefing/source_evaluation.py` | `tests/unit/test_briefing_source_evaluation.py` | CLI | proofs/F01 | pass | |
| Paper event mapping | F | F02 | `briefing/normalize.py` | `tests/unit/test_briefing_paper_events.py` | internal | proofs/F02 | pass | |
| Academic enrichment | F | F03 | `briefing/sources/` | `tests/unit/test_briefing_academic_enrichment.py` | internal | proofs/F03 | pass | |
| Hacker News adapter (disabled by default) | F | F04 | `briefing/sources/` | `tests/unit/test_briefing_hn.py` | internal | proofs/F04 | pass | disabled |
| Reddit adapter (disabled by default) | F | F05 | `briefing/sources/` | `tests/unit/test_briefing_reddit.py` | internal | proofs/F05 | pass | disabled |
| Bluesky adapter (disabled by default) | F | F06 | `briefing/sources/` | `tests/unit/test_briefing_bluesky.py` | internal | proofs/F06 | pass | disabled |
| X/Twitter API policy stub (disabled) | F | F07 | `briefing/sources/` | `tests/unit/test_briefing_x_stub.py` | internal | proofs/F07 | pass | disabled |
| Video/audio weekly context (disabled) | F | F08 | `briefing/sources/` | `tests/unit/test_briefing_video.py` | internal | proofs/F08 | pass | disabled |
| Source expansion offline e2e | F | F09 | `tests/integration_offline/test_briefing_phase_f_source_expansion_e2e.py` | same | test | proofs/F09 | pass | |
| Workflow state model | G | G01 | `briefing/workflow_state.py` | `tests/unit/test_briefing_workflow_state.py` | internal | proofs/G01 | pass | |
| Stage verifier registry | G | G02 | `briefing/workflow_verification.py` | `tests/unit/test_briefing_workflow_verification.py` | internal | proofs/G02 | pass | |
| MCP brief_* schemas | G | G03 | `mcp_server/schemas.py` | `tests/unit/test_mcp_briefing_schemas.py` | MCP | proofs/G03 | pass | |
| MCP brief_* tools | G | G04 | `mcp_server/tools.py` | `tests/unit/test_mcp_briefing_tools.py` | MCP | proofs/G04 | pass | |
| MCP briefing resources | G | G05 | `mcp_server/tools.py` | `tests/unit/test_mcp_briefing_resources.py` | MCP | proofs/G05 | pass | |
| Daily AI intelligence skill | G | G06 | `SKILL.md` / `.agents/skills/` | `tests/unit/test_skill_daily_ai_intelligence.py` | skill | proofs/G06 | pass | |
| Tool-scope governance | G | G07 | `briefing/tool_governance.py` | `tests/unit/test_briefing_tool_governance.py` | internal | proofs/G07 | pass | |
| Held-out agent evals | G | G08 | `tests/integration_offline/test_briefing_phase_g_agent_evals.py` | same | test | proofs/G08 | pass | |
| Replay/diagnosis and final gate | G | G09 | `docs/daily-ai-intelligence/replay-diagnosis.md` | `tests/integration_offline/test_briefing_phase_g_replay_diagnosis.py` | docs/test | proofs/G09 | pass | |
