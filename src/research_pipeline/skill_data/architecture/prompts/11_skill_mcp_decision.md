# Prompt 11 — Skill vs MCP Decision

You are deciding the skill / MCP / server boundaries. MCP must be justified,
not fashionable.

## Inputs

- `intermediate/ai_responsibility_matrix.md`.
- `references/mcp_adoption_guide.md`.

## Instructions

1. For each AI-/tool-facing capability, decide: internal module, **skill**, or
   **MCP server**, following the decision procedure in the MCP adoption guide.
2. If MCP is proposed, pass the **adoption gate**: clear external clients;
   clear resources/tools exposed; permission boundary; audit requirements;
   error model; versioning approach; non-MCP alternative considered. If any
   item is missing, either supply it or **DEFER** MCP.
3. Record the outcome (adopt or defer) even when the answer is defer — it
   becomes ADR-00xx (mcp-adoption).

## Output

`intermediate/skill_mcp_decisions.md` → populates §11.2 and seeds the MCP ADR.

## Validation / failure policy

- Gate: MCP is introduced only if every adoption-gate item is satisfied;
  otherwise it is explicitly deferred.
- Failure policy: `revise`.
