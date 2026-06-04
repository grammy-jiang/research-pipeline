# Prompt 09 — Provisional Tech Stack Selection

You are selecting a **provisional** tech stack. It is not final until the
AI-boundary and skill/MCP decisions are made and the coherence review (prompt
12) runs.

## Inputs

- `intermediate/goals_constraints.md`, `intermediate/solution_strategy.md`.
- `templates/tech_stack_decision_table.md`.

## Instructions

1. For each required choice, produce a row: Decision · Recommendation ·
   Alternatives · Rationale · Risks · Reversible?
2. Cover: primary language; backend framework; CLI framework (if needed); API
   style; job execution model; storage system; artifact storage;
   search/retrieval; queue/background jobs; observability stack; testing
   framework; configuration/secrets; LLM provider abstraction; agent
   orchestration; MCP strategy; deployment target.
3. State the decision criteria that drove each choice (fit, ecosystem maturity,
   familiarity, contract support, observability, security, local-dev/deploy
   simplicity, performance, cost, maintainability, reversibility).
4. **Anti-default rule:** never apply a fixed stack as a universal default.
   Justify every choice from this blueprint's context.
5. Mark the whole table **provisional**.

## Output

`intermediate/provisional_tech_stack_decisions.md`.

## Validation / failure policy

- Gate: all tech choices have rationale and alternatives, and the table is
  marked provisional.
- Failure policy: `revise`.
