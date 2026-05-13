# Proof Pack: E05_evidence_timeline

## Ticket
`E05_evidence_timeline`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_dossier_timeline.py -xvs
uv run ruff check src/research_pipeline/briefing/dossier_timeline.py
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 8/8 unit tests passing; ruff and mypy --strict clean.

## Dossier Safety Evidence
- Primary artifact gate enforced — timeline is only built after `build_dossier` has gated on primary-artifact presence; cluster-event entries require a non-empty `canonical_url`.
- One-topic focus preserved — the function takes a single `BriefingCluster` plus optional `topic_memories` for that cluster's `topic_ids`; no cross-topic merging.
- Evidence URLs or inference/speculation labels present — every entry carries a non-empty http(s) or obsidian:// `evidence_url`; topic-memory entries fall back to a deterministic `obsidian://open?vault=...&file={topic_id}` URL.
- No automatic dossier scheduling — `build_evidence_timeline` is a pure function with no side effects; no I/O.
- No general literature-review expansion — function only consumes events already in the cluster plus the named topic memories; no external fetches.

## Next Ticket
`E06_dossier_validation`
