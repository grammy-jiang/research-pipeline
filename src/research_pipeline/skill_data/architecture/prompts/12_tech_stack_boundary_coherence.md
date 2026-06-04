# Prompt 12 — Tech-Stack / AI-Boundary Coherence Review

You are finalizing the tech stack now that the AI boundary and MCP decision
exist. Tech-stack selection and AI-boundary design are co-dependent.

## Inputs

- `intermediate/provisional_tech_stack_decisions.md`
- `intermediate/ai_responsibility_matrix.md`
- `intermediate/skill_mcp_decisions.md`

## Instructions

1. Revisit every provisional tech-stack choice that affects agent
   orchestration, workflow execution, storage, observability, security, or MCP
   exposure.
2. Resolve incoherences: e.g. if MCP is deferred, remove MCP-server
   infrastructure from the stack; if AI output must be validated
   deterministically, ensure the stack supports a validation/gateway layer; if
   audit is append-only, ensure the storage choice supports it.
3. Produce the **final** tech-stack table (same columns) and note what changed
   from the provisional version and why.

## Output

`intermediate/final_tech_stack_decisions.md` → populates §7.

## Validation / failure policy

- Gate: the tech stack is consistent with the AI boundary and MCP strategy.
- Failure policy: `revise`.
