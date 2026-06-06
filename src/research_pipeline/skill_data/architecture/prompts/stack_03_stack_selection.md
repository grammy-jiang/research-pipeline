# Prompt stack_03 — Stack Selection

You are selecting the concrete technology stack. One decision per area, each
satisfying a driver from `stack_02`.

## Inputs

- `intermediate/stack_decision_drivers.md`, `intermediate/stack_inputs.json`.
- The architecture design document (for the provisional assumptions and
  Tech-Stack Selection Handoff in design §7).
- `references/tech-stack-selection-guide.md`,
  `templates/architecture_tech_stack_template.md`.

## Instructions

1. Fill the §4 decision table — one row per area: Area · Selected Technology ·
   Alternatives Considered · Rationale · Risk · Reversibility · Architecture
   Impact. Cover: runtime/language; application framework (backend / CLI / TUI /
   API as needed); storage / data layer; queue / background jobs; LLM provider
   abstraction; AI / agent orchestration; MCP framework / SDK; observability
   stack; testing stack; deployment / packaging.
2. Resolve the design's **Provisional Tech Assumptions** and **Tech-Stack
   Selection Handoff** (design §7): for each "Must Be Confirmed In Stack Mode?"
   row, confirm or override the provisional assumption and say why.
3. For each area, write the matching §5–§13 prose: why this technology, why not
   the alternatives, which architecture requirement it satisfies.
4. **Anti-default rule:** justify every choice from this architecture's drivers;
   never apply a fixed/familiar stack as a universal default.
5. **Technology-specific validity:** only credit a technology with properties it
   actually provides; downgrade absolute wording to application-enforced /
   tamper-evident / best-effort with a risk or ADR note.
6. **Honour the architecture's MCP decision:** if the architecture deferred MCP,
   do not introduce an MCP SDK without a justified change (which would be an
   *Architecture Update Required?* item).
7. Fill §14 Alternatives Considered and §15 Risk and Reversibility.

## Output

`intermediate/stack_selection.md` (the §4–§15 stack-selection content).

## Validation / failure policy

- Gate: every area has a selected technology with alternatives, rationale, risk,
  and a reversibility verdict; provisional design assumptions are resolved.
- Failure policy: `revise`.
