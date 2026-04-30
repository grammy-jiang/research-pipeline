# Proof Pack: E01_dossier_models

## Ticket
`E01_dossier_models`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_dossier_models.py -xvs
uv run ruff check src/research_pipeline/briefing/dossier.py
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 17/17 unit tests passing; ruff and mypy --strict clean.

## Dossier Safety Evidence
- Primary artifact gate enforced — `FactualityLabel` StrEnum (SUPPORTED_FACT, INFERENCE, SPECULATION_OR_WATCH_ITEM) tags every claim; `DossierClaim._validate` rejects SUPPORTED_FACT without a non-empty http(s) `evidence_url`.
- One-topic focus preserved — `EvidenceTimelineEntry.origin` is a `Literal["cluster_event","topic_memory"]`; topic-memory entries must carry an obsidian:// URL or http(s) URL.
- Evidence URLs or inference/speculation labels present — every claim instance must declare a label; URL scheme is validated.
- No automatic dossier scheduling — models are pure data, no scheduler hook.
- No general literature-review expansion — model surface is bounded to a single cluster's evidence.

## Next Ticket
`E02_dossier_renderer`
