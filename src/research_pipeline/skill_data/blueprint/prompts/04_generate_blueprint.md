# Prompt 04 — Generate Blueprint

You are composing an implementation-neutral product blueprint.

Use `templates/product_blueprint_template.md` as the skeleton. The
document must start with a `## Contents` section containing internal
Markdown links to all 18 sections.

## Required sections (in order)

1. Executive Product Thesis
2. Source Research Interpretation
3. Target Users and System Actors
4. Product Goals and Non-Goals
5. Research-to-Product Translation Map
6. Adopt / Adapt / Merge / Defer / Reject Decisions
7. Core Product Capabilities
8. Workflow Model
9. Logical Architecture
10. Conceptual Information Model
11. Decision Policies
12. Risk, Governance, and Safety Model
13. Evaluation Strategy
14. MVP Scope
15. Roadmap and Future Extensions
16. Open Questions and Validation Plan
17. Handoff Notes for Technical Design
18. Traceability Appendix

## Thesis

Generate a specific, short, product-oriented thesis using one of the
templates in the output template (single-domain / multi-domain /
research-validation). It must not be a paper summary, a list of
technologies, or a vague research ambition.

**Emphasis control.** Lead with the *primary* research-backed architecture,
not a conditional or secondary mechanism. If the report frames a mechanism
as conditional, bounded, escalation-only, or secondary, describe it as a
supporting mechanism — do not make it the product identity. (E.g. if the
evidence says "backbone-first generation with conditional review", lead
with the backbone-first engine, not with "multi-agent".)

## Workflows (§8)

Every major workflow needs: purpose, trigger, actors, inputs,
preconditions, decision gates, steps, outputs, failure modes, success
criteria, related capabilities, and traceability. See
`templates/workflow_template.md`.

## Logical architecture (§9)

Describe conceptual responsibilities and boundaries, **not**
implementation components. Conceptual names (e.g. "Admission Controller")
describe responsibility boundaries only — they must not imply classes,
services, packages, processes, or deployable units. See
`templates/logical_architecture_template.md`.

## Formatting requirements

- `## Contents` with valid internal links to every numbered section **and
  every appendix actually present** — scan the final headings and add any
  `## Appendix …` (the Appendix A self-check is always present; the optional
  Appendix B register only when you include it).
- At least one **Mermaid** diagram for the main end-to-end workflow.
- At least one **Mermaid** diagram for the logical architecture.
- Additional Mermaid workflow diagrams only for complex, safety-critical,
  or high-risk workflows.
- Markdown tables for the translation map (§5), decisions (§6), risks
  (§12), evaluations (§13), and policies (§11).
- Cite research evidence as `[arxiv_id]` or `[Author, Year]`, traceable to
  the source report's `## References`.
- Apply the length budget for the active `output_detail` setting. For
  `standard`, prefer: ≤ 8 core capabilities, ≤ 3 workflows, ≤ 10 risks,
  ≤ 8 evaluation scenarios, ≤ 8 decision policies, ≤ 10 open questions.
  Keep the main body scannable in one pass — move large tables (the full
  §5 translation map, the §18 traceability appendix) to appendices rather
  than the main flow. If content exceeds a budget, compress or switch to
  `detailed`; do not silently expand `standard` into an architecture dossier.
- End with `## Appendix A: Blueprint Quality-Gate Self-Check` — a compact
  table (Gate · Status · Finding · Required Action · Blocks Technical
  Design?). Every WARNING must carry a concrete required action and a
  blocks-technical-design verdict, not a passive note (see
  `prompts/05_quality_gate.md`).

## Traceability discipline

Every major capability must trace to a research mechanism, recurring
pattern, evidence-backed finding, engineering gap, risk item, assumption,
contradiction resolution, or a constrained explicit design decision. If no
trace exists, mark it **"Design hypothesis — requires validation."** An
explicit design decision may be used only to connect, operationalize, or
govern research-backed capabilities; it needs a rationale and must not
replace research traceability for core product claims.

**Gap-citation fallback.** Do not leave a citation cell blank. Paper-derived
items cite `[arxiv_id]` / `[Author, Year]`. An item derived from a
source-report gap with no paper citation cites
`[Source Report: Research Gaps — <gap name>]`. Leave a citation blank only
when the row is explicitly an internal design hypothesis.

## Metadata integrity (§1.5–1.6)

Copy metadata from the source report and skill metadata; never infer,
normalise, invent, or upgrade a value.

- The **blueprint skill version** comes from `manifest.json` (`version`).
  If you cannot read it, write `unknown` — do not invent a number
  (e.g. "1.0").
- Keep **pipeline runs integrated** and **gap-closure rounds** as separate
  fields. A report that "consolidates N pipeline runs" states a run count,
  not a gap-closure round count; if no round count is given, write
  `unknown` for rounds.
- Any field not explicitly available is `unknown` / `not specified`.

## Scope discipline (§3, §4, §14)

Classify every actor, domain, and use case as Primary / Secondary /
Future / Evidence-only / Out-of-scope.

- Only Primary-scope actors and domains — those the **thesis** names —
  appear as first-class users and MVP requirements.
- High-stakes or adjacent domains that appear only as research evidence
  (e.g. legal/medical when the thesis targets technical/academic content)
  are Secondary or Future: put them in §15 roadmap or §4 non-goals, not
  the primary actor table or MVP.

## MVP scope (§14)

Build §14 as MVP-0 / MVP-1 / Safety Baseline / Evaluation Baseline /
Explicitly Deferred / Success Definition.

- **MVP-0** is the smallest end-to-end slice that demonstrates the thesis
  with one realistic input and one realistic output. Keep it minimal — do
  not put every routing, memory, governance, and evaluation mechanism in
  MVP-0 unless the product cannot function or be safely evaluated without
  it. Do not call a large Phase-1 system an MVP-0.
- **MVP-1** is the first usable production version: production hardening,
  richer evaluation, and optional escalation added on top of MVP-0. If the
  product has more than 4 major capabilities the MVP-0/MVP-1 split is
  mandatory; for a genuinely simple product MVP-0 and MVP-1 may coincide —
  say so rather than inventing a split.
- List and justify safety and evaluation baselines **separately** from the
  core path, noting which are required already at MVP-0.
- Do not translate research completeness into MVP inclusion. Anything not
  required moves to a §15 phase with a one-line reason.
- `ACADEMIC`-gap items stay out of MVP unless the product's purpose is to
  validate that gap.

## Release-gate discipline (§12)

A release gate derived from a MEDIUM- or LOW-confidence mechanism is
justified only when the risk impact is HIGH, no cheaper baseline control
exists, and the blueprint states why it is required now. Otherwise
downgrade it to a warning, an evaluation/monitoring requirement, or a
Phase 2 release gate. State each gate's confidence and risk impact.

## Optional Appendix B — Design Decision Register

When the blueprint is intended for technical-architecture handoff (and is
detailed enough to warrant it), you may add `## Appendix B: Design Decision
Register` with columns Decision · Type · Rationale · Evidence · Reversible?
· Revisit Trigger. Include only decisions that materially affect downstream
architecture; do **not** duplicate the §6 table — the value is the
reversibility / revisit-trigger view §6 lacks. If you add it, list it in
Contents.

## Hard constraints

- Do **not** select a tech stack (language, framework, database, vector
  database, cloud provider, vendor, UI library, deployment model).
- Do **not** write code, a database schema, or implementation tickets.
- Do **not** treat unresolved research gaps as solved.
- Make the document actionable for a later technical-design skill without
  requiring it to re-read the original papers.
