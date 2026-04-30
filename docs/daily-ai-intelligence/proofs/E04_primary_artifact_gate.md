# Proof Pack: E04_primary_artifact_gate

## Ticket
`E04_primary_artifact_gate`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_dossier_primary_artifact.py -xvs
uv run ruff check src/research_pipeline/briefing/dossier.py
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 6/6 unit tests passing; ruff and mypy --strict clean.

## Dossier Safety Evidence
- Primary artifact gate enforced — `build_dossier` raises a `ValueError` (with the cluster_id in the message) when any of: `primary_artifact_present is False`, `canonical_urls` empty, `events` empty, or `len(topic_ids) > 1`.
- One-topic focus preserved — explicit invariant test `test_build_dossier_rejects_multi_topic_cluster` covers the multi-topic reject path.
- Evidence URLs or inference/speculation labels present — gate ensures at least one canonical_url exists before any rendering occurs.
- No automatic dossier scheduling — gate is invoked only when a user runs the CLI command; no scheduler.
- No general literature-review expansion — the gate's failure paths short-circuit before any external lookup.

## Next Ticket
`E05_evidence_timeline`
