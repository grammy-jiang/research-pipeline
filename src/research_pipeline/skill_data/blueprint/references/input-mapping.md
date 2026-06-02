# Input Mapping & Quality Thresholds

## Research-pipeline report sections → blueprint inputs

| Research-Pipeline Section | Blueprint Input Role |
|---|---|
| `## Executive Summary` | Product opportunity framing; confidence baseline |
| `## Research Question` | Product thesis seed; scope boundaries |
| `## Methodology` | Evidence quality signal (how many papers, sources) |
| `## Papers Reviewed` | Evidence quality; citation traceability |
| `## Research Landscape` (themes) | Core mechanism extraction per theme |
| `## Confidence-Graded Findings` (🟢🟡🔴) | ADOPT/DEFER/REJECT signal per idea |
| `## Research Gaps` (ACADEMIC / ENGINEERING) | Gap-type-specific product actions |
| `## Practical Recommendations` | Priority product requirements |
| `## Points of Contradiction` | Design decisions; contradiction resolution |
| `## Trade-Off Analysis` | Decision policy inputs |
| `## Readiness Assessment` (`IMPLEMENTATION_READY` / `HAS_GAPS`) | MVP feasibility signal |
| `## Evidence Map` | Traceability matrix inputs |
| `## References` | Authoritative citation list for the traceability appendix |
| `## Round History` | MVP sizing signal (see below) |

A long Round History with many remaining gaps signals a speculative
product space that requires more conservative MVP scoping.

## Supplementary artifacts (optional)

| Artifact | Purpose |
|---|---|
| `runs/<run_id>/summarize/synthesis_report.json` | Machine-readable findings and gap objects |
| `gaps.json` | Structured gap classification (`ACADEMIC` / `ENGINEERING` / `OUT_OF_SCOPE`) with priorities |
| `runs/<run_id>/screen/screened.jsonl` | Paper metadata for traceability citations |
| `runs/<run_id>/plan/query_plan.json` | Original topic framing |

The Markdown report is authoritative. If JSON conflicts with it, prefer
the Markdown report and note the conflict in §2. If the JSON has detail
missing from the Markdown report, use it but mark it supplementary.

## Input-quality detection

Record these booleans, then classify `overall`:

```yaml
input_quality:
  has_research_question: true/false
  has_confidence_graded_findings: true/false
  has_mechanisms: true/false
  has_evidence: true/false
  has_assumptions: true/false
  has_contradictions: true/false
  has_gaps_classified: true/false     # ACADEMIC/ENGINEERING labels present
  has_risks: true/false
  has_operational_implications: true/false
  has_architecture_hints: true/false
  has_readiness_assessment: true/false
  overall: strong | usable | weak | insufficient
```

## Classification thresholds

| Overall | Condition |
|---|---|
| `strong` | `has_research_question` AND `has_confidence_graded_findings` AND `has_gaps_classified` AND `has_evidence` |
| `usable` | `has_research_question` AND (`has_mechanisms` OR `has_confidence_graded_findings`) AND `has_evidence` |
| `weak` | `has_research_question` AND (`has_mechanisms` OR `has_evidence`) |
| `insufficient` | No research question, no identifiable mechanisms/findings, or no evidence. The skill MUST stop. |

- `insufficient` → emit the standardized failure (see
  `troubleshooting.md`) and stop.
- `weak` but not `insufficient` → proceed, and explicitly mark missing
  areas as assumptions or open questions in the blueprint.

## Multi-domain reports

If the report covers multiple unrelated product domains:

1. Detect the split from the research-landscape themes.
2. Scope the blueprint to one domain. Use the highest-coverage domain
   (most papers / highest-confidence findings) as the default and
   document the assumption in §2.
3. Note cross-domain features only as future extensions.
4. Ask the user only when domains have similar coverage and would lead to
   materially different product theses.

Never produce one blueprint covering all domains — it yields an incoherent
thesis.
