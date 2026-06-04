# Prompt 07 — Solution Strategy

You are defining the shape of the solution before committing to specifics.

## Inputs

- `intermediate/blueprint_parse.json`, `intermediate/clarifications.md`.
- `intermediate/existing_architecture_status.json` (respect the update mode).

## Instructions

1. State the overall architectural style (e.g. deterministic spine with bounded
   AI adapters; CLI + worker + stores; event-driven; etc.) and why it fits the
   blueprint workflows.
2. Identify the deterministic core vs the AI-assisted parts at a strategic
   level (the detailed matrix comes later in prompt 10).
3. State a provisional MCP posture (likely adopt / likely defer) to be
   confirmed in prompt 11.
4. List the top 3–5 strategic decisions and the forces behind them.
5. If the update mode is `patch`/`adr-only`/`compare`, scope the strategy to
   the changed areas and say what is unchanged.

## Output

`intermediate/solution_strategy.md` → populates §5.

## Validation / failure policy

- Gate: traditional/AI/MCP strategy is classified and the update mode is
  respected.
- Failure policy: `revise`.
