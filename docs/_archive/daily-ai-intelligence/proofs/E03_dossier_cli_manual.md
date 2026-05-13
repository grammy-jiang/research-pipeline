# Proof Pack: E03_dossier_cli_manual

## Ticket
`E03_dossier_cli_manual`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_cli_dossier.py -xvs
uv run ruff check src/research_pipeline/cli/cmd_brief.py
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 5/5 unit tests passing; ruff and mypy --strict clean.

## Dossier Safety Evidence
- Primary artifact gate enforced — `brief dossier --cluster <id>` resolves the cluster from `ranked_clusters.jsonl` and runs `build_dossier`, which raises a `ValueError` (non-zero CLI exit) when `primary_artifact_present` is false.
- One-topic focus preserved — the CLI requires a single `--cluster` argument; multi-topic clusters are rejected by the model invariant `len(topic_ids) <= 1`.
- Evidence URLs or inference/speculation labels present — output is validated via `validate_dossier_report` before being written to the workspace.
- No automatic dossier scheduling — invocation is strictly user-driven via Typer command; there is no orchestrator hook in the daily `brief run` flow that calls `dossier`.
- No general literature-review expansion — the command operates on one cluster only; it does not query external search sources or fan out to related topics.

## Next Ticket
`E04_primary_artifact_gate`
