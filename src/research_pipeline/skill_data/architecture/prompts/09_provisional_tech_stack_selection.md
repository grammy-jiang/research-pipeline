# Prompt 09 — Provisional Tech Stack Selection

You are selecting a **provisional** tech stack. It is not final until the
AI-boundary and skill/MCP decisions are made and the coherence review (prompt
12) runs.

## Inputs

- `intermediate/goals_constraints.md`, `intermediate/solution_strategy.md`.
- `intermediate/blueprint_parse.json` (the `recommended_next_stages` routing).
- `templates/tech_stack_decision_table.md`,
  `references/next-stages-and-handoffs-guide.md`.

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
5. **Technology-specific validity:** only credit a chosen technology with
   properties it actually provides. Do not describe one technology's
   enforcement model in terms of another's. When an enforcement or guarantee
   depends on implementation discipline rather than the technology itself,
   downgrade absolute wording ("guaranteed", "immutable", "no grants") to
   "application-enforced", "tamper-evident", "best-effort", or "requires
   operational control", and add a risk or ADR note. (Example: an append-only
   audit log on an embedded store with no role/grant model is
   *application-enforced + hash-chain tamper-evident*, not enforced by database
   grants — say so. This is an illustration, not a tech recommendation.)
6. Mark the whole table **provisional**.
7. **Provisional-tech discipline (design vs stack mode):** read the blueprint's
   §19 `recommended_next_stages` routing for `tech-stack-selection`.
   - If `RUN` or `DEFER`: final selection belongs to `stack` mode. Keep the
     table provisional and additionally produce two tables for design §7.1/§7.2:
     - **Provisional Tech Assumptions** — Area · Provisional Assumption · Reason
       · Must Be Confirmed In Stack Mode? (Yes for any area several viable
       options could satisfy).
     - **Tech-Stack Selection Handoff** — Decision Needed · Architecture
       Constraint · Candidate Options · Risk if Wrong.
     Do **not** lock a final framework / database / cloud / AI-orchestration /
     MCP SDK choice when multiple viable options exist.
   - If `SKIP` (stack already fixed by the blueprint/user): note the stack is
     fixed and by whom, and replace §7.1/§7.2 with that one-line note.

## Output

`intermediate/provisional_tech_stack_decisions.md` (the §7 table plus the
§7.1 Provisional Tech Assumptions and §7.2 Tech-Stack Selection Handoff tables,
or the "stack fixed by …" note).

## Validation / failure policy

- Gate: all tech choices have rationale and alternatives, the table is marked
  provisional, and (unless tech-stack-selection = SKIP) Provisional Tech
  Assumptions + a Tech-Stack Selection Handoff are produced.
- Failure policy: `revise`.
