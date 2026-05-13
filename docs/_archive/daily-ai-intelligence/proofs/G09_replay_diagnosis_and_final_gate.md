# Proof Pack: G09_replay_diagnosis_and_final_gate

## Ticket
`G09_replay_diagnosis_and_final_gate`

## Verification Commands Run
```bash
uv run pytest tests/integration_offline/test_briefing_phase_g_replay_diagnosis.py -xvs
uv run ruff check src/research_pipeline/briefing src/research_pipeline/mcp_server tests/
uv run mypy src/
```

## Result
PASS — 5/5 tests pass; ruff clean; mypy clean.

## Harness Safety Evidence
- `docs/daily-ai-intelligence/replay-diagnosis.md` documents state model, stage verifiers, replay procedure, diagnosis checklist, and tools to use.
- `docs/daily-ai-intelligence/final-acceptance-gate.md` enumerates required commands, required documents, and completion condition.
- Integration tests:
  - `test_replay_loads_state_and_verifies_completed_stages` — state advances through every stage; `verify_completed_stages` returns `ok=True` for all.
  - `test_replay_detects_artifact_drift` — deleting a stage artifact post-run flips the verifier to `ok=False` with a sorted issues tuple.
  - `test_failed_run_state_is_replayable` — failed-state `last_error` round-trips through JSON.
- Phase G total tests passing: 93/93 across G01–G09 unit + integration suites.
- A-F governance unchanged.

## Next Ticket
None — Phase G complete.
