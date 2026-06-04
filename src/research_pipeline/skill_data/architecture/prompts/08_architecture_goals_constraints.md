# Prompt 08 — Architecture Goals and Constraints

You are turning blueprint goals into architecture drivers and constraints.
§4 must not be a placeholder.

## Inputs

- `intermediate/blueprint_parse.json`, `intermediate/clarifications.md`,
  `intermediate/solution_strategy.md`.

## Instructions

Produce all of §4 using the template structure:

- 4.1 Architecture Goals
- 4.2 Functional Constraints
- 4.3 Non-Functional Requirements (table: Requirement · Target · Source)
- 4.4 Security / Privacy Constraints
- 4.5 Data and Retention Constraints
- 4.6 Cost / Latency / Performance Constraints
- 4.7 Team / Development Constraints
- 4.8 MVP-0 / MVP-1 Architecture Constraints
- 4.9 Explicit Assumptions (table: Assumption · Reason · Reversible? · Revisit
  Trigger)

Draw constraint candidates from: deployment model, data locality, external-LLM
allowance, multi-user vs single-user, audit retention, latency target, document
size, cost ceiling, developer familiarity, regulated-data handling, offline
capability. Every NFR row needs a source (blueprint §, clarification, or
assumption).

## Output

`intermediate/goals_constraints.md` → populates §4.

## Validation / failure policy

- Gate: NFRs, constraints, and explicit assumptions are present (not
  placeholders).
- Failure policy: `revise`.
