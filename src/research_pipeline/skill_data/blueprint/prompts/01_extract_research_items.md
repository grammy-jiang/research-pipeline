# Prompt 01 — Extract Research Items

You are extracting product-relevant research items from a research
synthesis report produced by the `research-pipeline` skill.

The report uses confidence grades: 🟢 HIGH (3+ papers), 🟡 MEDIUM (1–2
papers or with caveats), 🔴 LOW (preliminary or contradicted). Gaps are
classified as `ACADEMIC`, `ENGINEERING`, or `OUT_OF_SCOPE`.

## Step 0 — Intake and quality check

1. Read the source `<topic-slug>-research-report.md` in full.
2. Load supplementary artifacts if present (`gaps.json`,
   `synthesis_report.json`, `screened.jsonl`, `query_plan.json`). The
   Markdown report is authoritative; JSON is supplementary.
3. Detect which canonical sections exist (see
   `references/input-mapping.md`).
4. Classify input quality using the thresholds in
   `references/input-mapping.md`:
   - `insufficient` → STOP. Emit the standardized insufficient-input
     failure document (see `references/troubleshooting.md`). Do not
     continue.
   - `weak` → proceed, but record missing areas to surface later as
     assumptions or open questions.
   - `usable` / `strong` → proceed.
5. Detect domain count. If the report spans multiple unrelated product
   domains, choose the highest-coverage domain as a documented default;
   ask the user only when domains have similar coverage and would yield
   materially different theses.

## Step 1 — Extract

Extract:

- research question and scope
- mechanisms (with confidence grade and citation)
- methodologies
- recurring patterns (with citation)
- benchmarks, methodology comparisons, benchmark tables, paper-specific
  performance data, and evaluation findings
- assumptions
- contradictions (from a Points of Contradiction section)
- evidence-strength claims (from the Confidence-Graded Findings section)
- academic gaps (Research Gaps, type `ACADEMIC`)
- engineering gaps (Research Gaps, type `ENGINEERING`)
- risks (from any Risk Register or Security Considerations section)
- operational implications
- architecture hints (from Readiness Assessment or Recommendations)
- the Round History length and the Readiness Assessment verdict
  (`IMPLEMENTATION_READY` / `HAS_GAPS`) — these size the MVP later

## Step 2 — Classify each item

For each item, assign:

- item id
- item name
- source section (e.g. "Confidence-Graded Findings §3")
- source citation (`[arxiv_id]` or `[Author, Year]`)
- item type: one of taxonomy, mechanism, algorithm, workflow_pattern,
  benchmark, security_method, data_structure, empirical_result,
  assumption, contradiction, academic_gap, engineering_gap, risk,
  operational_implication, architecture_hint
- confidence grade: HIGH / MEDIUM / LOW / unknown
- product relevance: critical / useful / optional / weak / out_of_scope
- one-sentence summary

## Constraints

- Do **not** design the product yet.
- Do **not** select a tech stack.
- Preserve every citation exactly as written in the source report so it
  stays traceable to the `## References` section.
- Treat these schemas as reasoning aids, not output formats — the final
  blueprint is prose and tables, never raw YAML/JSON.
