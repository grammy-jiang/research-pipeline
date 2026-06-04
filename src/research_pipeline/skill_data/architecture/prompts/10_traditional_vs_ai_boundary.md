# Prompt 10 — Traditional Software vs AI Responsibility Matrix

You are producing the mandatory responsibility matrix (§6).

## Inputs

- `intermediate/blueprint_parse.json`,
  `intermediate/provisional_tech_stack_decisions.md`.
- `templates/ai_responsibility_matrix_template.md`.

## Instructions

1. For each significant responsibility, mark the owner(s): Traditional Software
   / AI / LLM / Skill / MCP Server / Human, plus a Notes column.
2. Apply the core rule: deterministic control, state, storage, audit, security,
   workflow transitions, and interface contracts belong to traditional software
   unless explicitly justified. AI owns only judgment-heavy, language-heavy, or
   reasoning-heavy tasks.
3. Enforce the quality gate: AI components must not mutate durable state
   directly unless a deterministic validation gate + audit controls exist. If a
   row assigns state mutation to AI, show the downstream validation gate.
4. Be explicit about which steps are AI and how their output is validated
   before affecting state.

## Output

`intermediate/ai_responsibility_matrix.md` → populates §6 and seeds §11.

## Validation / failure policy

- Gate: deterministic and AI boundaries are explicit; no unvalidated AI state
  mutation.
- Failure policy: `revise`.
