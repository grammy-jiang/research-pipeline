# Prompt 05 — Skill Operator UX

You are defining **Skill Operator UX** — how the user interacts with the **AI
skill workflow itself** (the product-design skills, not the end product). Keep
this separate from Target Software UX (prompt 06).

## Inputs

- `intermediate/architecture_parse.json`, `intermediate/blueprint_parse.json`,
  `intermediate/clarifications.md`.

## Instructions

Define, as the §5 content:

- **Invocation** — how the user invokes the skill (default `ux-design`, and the
  explicit forms with an architecture/blueprint/stack argument and a `--mode`).
- **Input auto-detection** — what file the skill auto-detects and the search
  order; how the topic slug is derived.
- **Missing-input behaviour** — what happens if no architecture document is
  found (the STOP message) and how missing architecture sections are reported
  (continue-with-warning vs fail).
- **Clarification behaviour** — interactive vs automatic vs hybrid; how questions
  are asked (one at a time, recommended answer, default assumption).
- **Assumption recording** — how assumptions and high-risk flags are surfaced.
- **Resume / update behaviour** — how the skill resumes or updates a prior
  `<topic-slug>-ux-design.md` (new / regenerate / patch / resume; Update History).
- **Output** — the generated file name and where it is written; inline fallback.
- **Uncertainty / escalation** — how uncertainty is reported and how missing
  information is escalated (ASK_USER) rather than fabricated.

Keep it concrete but at the workflow level — this is operator experience, not
visual design. This section generalizes to the other design-chain skills
(blueprint / architecture / ux-design); describe the pattern, noting where
ux-design differs.

## Output

`intermediate/skill_operator_ux.md` (the §5 content).

## Validation / failure policy

- Gate: invocation, auto-detection, missing-input, clarification, assumptions,
  resume/update, output, and escalation behaviours are all defined.
- Failure policy: `revise`.
