# Phase F Acceptance Gate Addendum

Append to `acceptance-gates.md`.

## Phase F Gate

F01-F09 audit_pass; no pending/in_progress/failed/reopened/blocked F tickets; A/B/C/D/E tests still pass; new sources have registry entries, offline fixtures, retention/cadence/rate-limit policy, parser tests, disabled-by-default behavior, and side-by-side report comparison; no scraping; no firehose; X/Twitter remains official-API-only and disabled unless strict policy gate passes.

Required commands:

```bash
uv run pytest tests/unit/test_briefing_source_evaluation.py tests/unit/test_briefing_paper_events.py tests/unit/test_briefing_academic_enrichment.py tests/unit/test_briefing_hacker_news.py tests/unit/test_briefing_reddit.py tests/unit/test_briefing_bluesky.py tests/unit/test_briefing_x_api_policy.py tests/unit/test_briefing_video_audio.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_f_source_expansion_e2e.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_a_e2e.py tests/integration_offline/test_briefing_phase_b_memory_e2e.py tests/integration_offline/test_briefing_phase_c_obsidian_e2e.py tests/integration_offline/test_briefing_phase_d_feedback_e2e.py tests/integration_offline/test_briefing_phase_e_dossier_e2e.py -xvs
uv run ruff format .
uv run ruff check . --fix
uv run mypy src/
```
