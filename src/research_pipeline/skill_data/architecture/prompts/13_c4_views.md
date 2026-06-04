# Prompt 13 — C4 Views

You are generating the C4 architecture views as Mermaid diagrams.

## Inputs

- `intermediate/final_tech_stack_decisions.md`,
  `intermediate/ai_responsibility_matrix.md`, `intermediate/blueprint_parse.json`.
- `references/c4_model_summary.md`.

## Instructions

Generate, as Mermaid:

1. **System Context** (required) — users, system boundary, external systems,
   external AI services, file inputs, monitoring/audit consumers, human
   approval actors.
2. **Container / Runtime** (required) — entrypoints, backend/worker, stores,
   artifact storage, queue (if needed), agent runtime, MCP server (only if
   justified), external providers, observability backend. Give each container a
   responsibility and owner.
3. **Dynamic / sequence** (required) — the main workflow end to end, including
   the AI→validation→commit path and the primary failure branch.
4. **Component** (conditional) — only for the most complex container; name it.
5. **Deployment** (conditional) — only when topology affects security, privacy,
   data locality, scaling, availability, or operations. Do not add it merely
   because an external LLM is used.

Do not convert every conceptual blueprint component into a separate container
without rationale.

## Output

`intermediate/c4_views.md` → populates §8, §9, §10 (if any), §15.

## Validation / failure policy

- Gate: System Context, Container/Runtime, and a main-workflow Dynamic view are
  present.
- Failure policy: `revise`.
