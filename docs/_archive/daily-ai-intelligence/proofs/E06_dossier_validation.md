# Proof Pack: E06_dossier_validation

## Ticket
`E06_dossier_validation`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_validate_dossier.py -xvs
uv run ruff check src/research_pipeline/briefing/validate_dossier.py
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 8/8 unit tests passing; ruff and mypy --strict clean.

## Dossier Safety Evidence
- Primary artifact gate enforced — validator rejects markdown lacking required sections (Evidence Timeline, Artifacts To Open) and demands at least one http(s) URL.
- One-topic focus preserved — validator rejects markdown that contains more than one `topic_id:` line in front-matter.
- Evidence URLs or inference/speculation labels present — validator requires the literal `factuality_label=supported_fact` substring (or an explicit inference/speculation alternative) and at least one http(s) link.
- No automatic dossier scheduling — validator is a pure function; no scheduling side effects.
- No general literature-review expansion — link cap (`max_links=30`) and word cap (`max_words=1500`) prevent the dossier from sprawling into a literature review.

## Next Ticket
`E07_dossier_archive_linking`
