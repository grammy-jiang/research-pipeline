# Prompt 05 — Blueprint-to-Architecture Traceability Map

You are building the explicit map from blueprint sections to architecture
sections so the handoff is never reinvented.

## Inputs

- `intermediate/blueprint_parse.json`.
- `templates/blueprint_to_architecture_map_template.md`.

## Instructions

1. Produce the mapping table using the template as the baseline, adjusting rows
   to the actual blueprint sections present.
2. For each blueprint section, name the architecture section(s) it drives and
   one line on how it is used.
3. If a standard blueprint section is absent, keep the row and write
   "not present in blueprint" in the source column.
4. State the traceability quality gate: every major architecture decision must
   trace back to a blueprint section, a user clarification, a rule-pack
   decision, or an explicit recorded assumption.

## Output

`intermediate/traceability_map.md` (a Markdown table + the gate statement).

## Validation / failure policy

- Gate: the major architecture sections each have at least one blueprint
  source (or an explicit "not present" note).
- Failure policy: `revise`.
