# Prompt 00 — Assess Input Quality

You are performing the blueprint skill's intake and input-quality assessment.
This prompt owns the strong / usable / weak / insufficient classification.
Later prompts consume that result; they must not recompute it.

## Intake

1. Read the source `<topic-slug>-research-report.md` in full.
2. Load supplementary artifacts if present (`gaps.json`,
   `synthesis_report.json`, `screened.jsonl`, `query_plan.json`). The Markdown
   report is authoritative; JSON is supplementary.
3. Detect which canonical sections exist using `references/input-mapping.md`.
4. Detect domain count. If the report spans multiple unrelated product domains,
   choose the highest-coverage domain as a documented default; ask the user only
   when domains have similar coverage and would yield materially different
   product theses.

## Quality classification

Classify input quality using the thresholds in `references/input-mapping.md`:

- `insufficient` -> STOP. Emit the standardized insufficient-input failure
  document from `references/troubleshooting.md`. Do not fabricate a blueprint.
- `weak` -> proceed, but record missing areas to surface later as assumptions or
  open questions.
- `usable` / `strong` -> proceed.

Record the result in `intermediate/input_quality.json` with:

- quality: strong / usable / weak / insufficient
- detected_sections
- missing_sections
- source_research_question
- target_domain_default and rationale
- input_constraints and caveats

## Ownership boundaries

- This prompt is the only owner of input-quality classification.
- `prompts/01_extract_research_items.md` extracts research items after this
  gate passes; it does not decide whether input quality is sufficient.
- `prompts/03_resolve_ideas.md` owns final MVP inclusion decisions.
