# Proof Pack: C06 Obsidian export validation

## Ticket
`C06_obsidian_export_validation`

## Implemented Files
- `src/research_pipeline/briefing/validate_obsidian.py`
- `tests/unit/test_briefing_validate_obsidian.py`

## Verification Commands Run
```bash
uv run pytest tests/unit/test_briefing_validate_obsidian.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Result
PASS — 10/10 unit tests pass; ruff clean; mypy clean (38 source files).

## Obsidian Safety Evidence
- `validate_obsidian_note_file` confirms each export has the expected
  `type`, owns `generated_id`, and contains the required `## Agent Read
  Map` heading.
- A wiki-to-markdown rewrite regression
  (`[Topic Alpha](Topic-Alpha.md)`) is detected and reported by the
  validator.
- Aggregator `validate_obsidian_export` rolls up daily + topic + source
  results and counts wiki-links per bundle for telemetry.
- Validators read files from disk and do not mutate them; safe to run
  before and after CLI export.

## Next Ticket
`C07_cli_export_obsidian`
