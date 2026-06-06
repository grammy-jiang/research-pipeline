# Prompt stack_02 — Technology Decision Drivers

You are extracting the architecture requirements that **drive** technology
choices. Stack mode satisfies the architecture; this pass turns the architecture
into a list of decision drivers.

## Inputs

- `intermediate/stack_inputs.json` (architecture design + blueprint paths).
- The architecture design document.
- `references/tech-stack-selection-guide.md`.

## Instructions

1. Read the architecture and extract the constraints that bind the stack, from:
   - §4 Architecture Goals and Constraints (NFRs, security/privacy, cost/latency,
     team, MVP-0/MVP-1);
   - §6 Traditional-vs-AI boundary and §11 AI / Skill / MCP architecture (what AI
     orchestration / MCP the stack must support);
   - §12 Interface Contracts and §13 Data Contracts (what the storage/runtime
     must express);
   - §14 State, Storage, and Data Lifecycle (state model the storage must hold);
   - §17 Security and Trust Boundaries incl. the data-egress decision (which
     constrains the LLM provider abstraction and logging stack);
   - §16 Observability (correlation IDs, audit trail, redaction);
   - §20 Deployment Architecture (deployment target, locality).
2. Produce a driver table: Driver · Source (architecture §) · Why It Constrains
   the Stack.
3. Do **not** select technologies yet. Do **not** restate the whole
   architecture — capture only what drives a stack decision.
4. Note any architecture requirement that *no* reasonable technology can satisfy
   — that is an early signal of *Architecture Update Required?* (handled in
   `stack_04`).

## Output

`intermediate/stack_decision_drivers.md` (the §3 Technology Decision Drivers
table for the stack document).

## Validation / failure policy

- Gate: architecture requirements are consumed as explicit, sourced drivers.
- Failure policy: `revise`.
