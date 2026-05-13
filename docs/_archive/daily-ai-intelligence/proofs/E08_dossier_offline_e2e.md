# Proof Pack: E08_dossier_offline_e2e

## Ticket
`E08_dossier_offline_e2e`

## Verification Commands Run
```bash
uv run pytest tests/integration_offline/test_briefing_phase_e_dossier_e2e.py -xvs
uv run pytest tests/unit/test_briefing_dossier_models.py tests/unit/test_briefing_dossier_renderer.py tests/unit/test_briefing_dossier_primary_artifact.py tests/unit/test_briefing_dossier_timeline.py tests/unit/test_briefing_validate_dossier.py tests/unit/test_briefing_cli_dossier.py tests/unit/test_briefing_dossier_linking.py -q
uv run ruff check src/research_pipeline/briefing tests/integration_offline/test_briefing_phase_e_dossier_e2e.py
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 4/4 offline e2e tests + 59 Phase E unit tests passing (63 total Phase E tests); full unit suite 4189 passed / 1 skipped; ruff and mypy --strict clean. No network calls — `test_dossier_offline_no_network` patches `socket.create_connection` to raise.

## Dossier Safety Evidence
- Primary artifact gate enforced — `test_dossier_no_primary_artifact_rejected` runs the CLI against a cluster with `primary_artifact_present=False` and asserts non-zero exit code with no dossier file produced.
- One-topic focus preserved — fixture clusters each declare a single `topic_id`; happy-path dossier markdown is validated with `validate_dossier_markdown`, which rejects multi-topic markdown.
- Evidence URLs or inference/speculation labels present — `test_dossier_manual_happy_path` asserts `https://example.com/release-a` and `factuality_label=supported_fact` are present in the produced dossier.
- No automatic dossier scheduling — every test invokes the dossier command directly; no scheduler hook is triggered.
- No general literature-review expansion — `test_dossier_long_rejection_via_validator` confirms the validator rejects an overlong dossier (>1500 words), bounding scope.

## Next Ticket
`none` — Phase E complete; do not start Phase F.
