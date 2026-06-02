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

- `## Contents` with valid internal links to every section.
- At least one **Mermaid** diagram for the main end-to-end workflow.
- At least one **Mermaid** diagram for the logical architecture.
- Additional Mermaid workflow diagrams only for complex, safety-critical,
  or high-risk workflows.
- Markdown tables for the translation map (§5), decisions (§6), risks
  (§12), evaluations (§13), and policies (§11).
- Cite research evidence as `[arxiv_id]` or `[Author, Year]`, traceable to
  the source report's `## References`.
- Apply the section-length guidance for the active `output_detail`
  setting (concise / standard / detailed; default standard).

## Traceability discipline

Every major capability must trace to a research mechanism, recurring
pattern, evidence-backed finding, engineering gap, risk item, assumption,
contradiction resolution, or a constrained explicit design decision. If no
trace exists, mark it **"Design hypothesis — requires validation."** An
explicit design decision may be used only to connect, operationalize, or
govern research-backed capabilities; it needs a rationale and must not
replace research traceability for core product claims.

## Hard constraints

- Do **not** select a tech stack (language, framework, database, vector
  database, cloud provider, vendor, UI library, deployment model).
- Do **not** write code, a database schema, or implementation tickets.
- Do **not** treat unresolved research gaps as solved.
- Make the document actionable for a later technical-design skill without
  requiring it to re-read the original papers.
