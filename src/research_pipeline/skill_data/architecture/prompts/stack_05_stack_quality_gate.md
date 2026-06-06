# Prompt stack_05 — Stack Quality Gate

You are running the `stack` mode quality gate before writing the document. Be
skeptical: use PASS / WARNING / FAIL, and never mark a clean PASS over a known
contradiction.

## Inputs

- `intermediate/stack_selection.md`, `intermediate/stack_architecture_impact.md`,
  `intermediate/stack_decision_drivers.md`.
- `templates/architecture_tech_stack_template.md`,
  `references/tech-stack-selection-guide.md`.

## Instructions

1. Fill the §19 Tech Stack Quality-Gate Self-Check table:

   | Gate | Status | Finding | Required Action | Blocks Architecture Update? |
   |---|---|---|---|---|
   | Architecture requirements consumed | PASS/WARNING/FAIL | … | … | yes/no |
   | Alternatives considered | PASS/WARNING/FAIL | … | … | yes/no |
   | Risk and reversibility stated | PASS/WARNING/FAIL | … | … | yes/no |
   | Security/privacy implications included | PASS/WARNING/FAIL | … | … | yes/no |
   | Architecture impact notes produced | PASS/WARNING/FAIL | … | … | yes/no |
   | Architecture update requirement explicit | PASS/WARNING/FAIL | … | … | yes/no |

2. **Fail conditions:** technology choices made without architecture
   requirements; no alternatives considered; selected technologies contradict
   architecture constraints; security/privacy implications ignored; architecture
   impact missing.
3. **Stay in scope:** if the gate reveals the *architecture* is wrong (not the
   stack), record it as `Architecture Update Required?` — do not rewrite the
   architecture and do not change the product thesis / MVP / UX intent.
4. On FAIL, revise §4–§18 and re-run (max 3 attempts; then surface the failing
   gates and stop — do not deliver an unvalidated tech-stack document).

## Output

`intermediate/stack_quality_gate.md`.

## Validation / failure policy

- Gate: no failed stack quality gates (or failures surfaced after 3 attempts).
- Failure policy: `revise_max_3_then_stop`.
