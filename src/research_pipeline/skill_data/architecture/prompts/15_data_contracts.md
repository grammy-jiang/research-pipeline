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
7. Describe the §14 state machine(s) as a **canonical state model** with three
   distinct categories so terms are not conflated:
   - **Lifecycle states** — the persisted states a unit of work moves through
     (e.g. queued → … → completed / failed). These are the only values the
     persistence schema and API status should use.
   - **Operational condition flags** — orthogonal runtime conditions (e.g.
     degraded, fallback-used, probe-unavailable, discourse-unscored), not
     lifecycle states.
   - **Audit events** — things that happened (e.g. escalated, job_failed), not
     states.
   Every status/state/condition term used elsewhere (API contracts, failure
   handling, observability, human-review, probe policy) must resolve to one of
   these three categories. Do not use a condition word (e.g. "degraded") as a
   lifecycle state unless it is in the lifecycle list.

Do not author concrete migrations or DDL.

## Output

`intermediate/data_contracts.md` → populates §13 and §14.

## Validation / failure policy

- Gate: storage owner, retention, and schema-evolution strategy are present for
  every object; audit immutability is defined.
- Failure policy: `revise`.
