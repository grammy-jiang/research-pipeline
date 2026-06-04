# Prompt 21 — Rule-Pack Review

You are applying the architecture rule packs as review gates — not pasting them
into the output.

## Inputs

- All intermediate artifacts produced so far.
- `rule-packs/` (boundary, data, interface, reliability, ai_boundary,
  observability, security).

## Instructions

1. Run each rule pack against the relevant intermediate artifacts.
2. For each rule pack, record PASS / WARN / FAIL with a one-line finding and,
   for any non-PASS, the section to revise and the required action.
3. Pay special attention to the hard fails: AI mutating durable state without
   deterministic validation; MCP introduced without passing the adoption gate;
   secrets in artifacts; a critical workflow with no failure handling; every
   conceptual component promoted to a service without rationale.

## Output

`intermediate/rule_pack_review.md` (a table: Rule Pack · Status · Finding ·
Section · Required Action).

## Validation / failure policy

- Gate: no failed rule-pack gates remain.
- Failure policy: `revise_max_3_then_stop` — after 3 failed revisions, surface
  the failing rules to the user and stop.
