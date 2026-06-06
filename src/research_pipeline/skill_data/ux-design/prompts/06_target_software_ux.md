# Prompt 06 — Target Software UX

You are defining **Target Software UX** — how the end user (and any AI agents /
MCP clients) interacts with the **actual software being designed**. Keep this
separate from Skill Operator UX (prompt 05).

## Inputs

- `intermediate/architecture_parse.json`, `intermediate/blueprint_parse.json`,
  `intermediate/clarifications.md`.

## Instructions

Define, as the §6 content, the end-to-end product experience grounded in the
architecture's surfaces, state model, and workflows:

- **Job submission** — how the user submits work (per the primary surface).
- **Progress** — how progress is shown (mapped to architecture observability /
  user-visible states).
- **Output review** — how the user reviews output and what they see first.
- **Quality-risk surfacing** — how quality score / review-required status is
  surfaced (technical vs simplified, per clarifications).
- **Human review** — how a human reviewer approves / rejects / edits / requests
  rerun (only if the architecture has a human-review flow; otherwise note n/a).
- **Error recovery** — how the user recovers from failures (retry / edit input /
  switch route / export diagnostics / escalate), aligned with the architecture's
  failure/recovery model (detailed states in prompt 09).
- **Audit inspection** — how the user inspects audit evidence / data-egress
  status (per the architecture's audit + egress model).
- **Agent / MCP interaction** — how an AI agent or MCP client interacts safely
  (only if those surfaces exist): safe high-level operations, read-only-by-default
  resources, refusal/escalation on unsafe requests.

Every behaviour must map to an architecture fact (surface, state, event,
contract). Where the product needs something the architecture lacks, **do not
invent it** — note it for architecture feedback (prompt 11).

## Output

`intermediate/target_software_ux.md` (the §6 content).

## Validation / failure policy

- Gate: submit / progress / review / quality-risk / recovery / audit / agent
  paths are defined (or explicitly n/a) and each maps to an architecture fact.
- Failure policy: `revise`.
