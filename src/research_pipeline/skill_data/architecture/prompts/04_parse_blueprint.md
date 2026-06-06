# Prompt 04 — Parse Blueprint

You are extracting the structured facts the architecture passes will consume.

## Inputs

- `intermediate/blueprint_architecture_extract.md` (or the full blueprint).

## Instructions

Parse and record:

- **thesis** — the product identity and primary architecture.
- **mvp** — MVP-0 (smallest end-to-end slice) and MVP-1 (first usable version).
- **actors** — primary/secondary users, external systems, human approval actors.
- **capabilities** — core product capabilities.
- **workflows** — each major workflow (trigger, steps, decision gates, outputs,
  failure modes).
- **product_experience** — the §9 Product Experience Direction and its handoff:
  primary interaction mode, trust/control/transparency requirements,
  human-in-the-loop experience, failure/recovery expectations, and the UX
  assumptions handed to architecture. Treat these as UX intent the architecture
  must preserve (surfaces, states, review/audit flows) — do not invent UX
  assumptions the blueprint did not state. (Drives §23 Experience Architecture.)
- **recommended_next_stages** — the §19 Recommended Next Stages routing: the
  RUN / SKIP / DEFER / ASK_USER decision for each of tech-stack-selection,
  ux-design, security-review, test-design, architecture-update, and
  architecture-reconciliation, with each stage's reason and revisit trigger.
  Treat these as overrideable routing the architecture must reflect — do not
  silently expand or collapse the pipeline. (Drives §7 provisional-tech
  discipline and §24 Recommended Next Stages and Downstream Handoffs.) If the
  blueprint has no §19, write `not present in blueprint` and default
  tech-stack-selection to RUN (keep §7 provisional).
- **logical_architecture** — the blueprint's conceptual components and
  boundaries.
- **conceptual_objects** — the information model objects.
- **decision_policies** — policy rules that imply validation/escalation.
- **risks** — risk/governance/safety items and any release gates.
- **evaluation** — evaluation strategy and requirements.
- **handoff_notes** — the technical-design handoff section.
- **decision_register** — the design decision register, if present.
- **open_questions** — unresolved questions.

Cite the blueprint section for each item. If a field is absent, write
`not present in blueprint` — do not invent it.

## Output

`intermediate/blueprint_parse.json` with the keys above.

## Validation / failure policy

- Gate: thesis, MVP, workflows, and handoff notes are present (or explicitly
  marked absent with the consequence noted).
- Failure policy: `ask_or_limited_architecture` — if core fields are missing,
  ask the user, or proceed with an explicitly limited architecture.
