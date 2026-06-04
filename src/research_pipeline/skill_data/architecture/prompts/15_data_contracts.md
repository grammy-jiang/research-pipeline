# Prompt 15 — Data Contracts and Storage Lifecycle

You are defining schema-level data objects and their lifecycle — not final
database migrations.

## Inputs

- `intermediate/blueprint_parse.json` (conceptual information model),
  `intermediate/interface_contracts.md`, `intermediate/final_tech_stack_decisions.md`.

## Instructions

1. Map blueprint objects to storage: Blueprint Object · Storage Owner ·
   Suggested Storage · Retention · Notes (schema-evolution / immutability).
2. Define storage ownership (which module owns each store).
3. Define retention for every store and artifact.
4. Define a schema-evolution strategy (additive-first, migration path).
5. Separate metadata from large artifacts.
6. Define audit immutability semantics (append-only).
7. Describe the state machine(s) for the primary entity (§14).

Do not author concrete migrations or DDL.

## Output

`intermediate/data_contracts.md` → populates §13 and §14.

## Validation / failure policy

- Gate: storage owner, retention, and schema-evolution strategy are present for
  every object; audit immutability is defined.
- Failure policy: `revise`.
