# Prompt 02 — Architecture Parse

You are extracting the architecture facts the UX passes will consume. UX design
must respect these constraints; it must not invent flows the architecture cannot
support.

## Inputs

- `intermediate/input_resolution.json` (architecture path).
- The architecture design document (and optional tech-stack as constraints).

## Instructions

Read the architecture and extract:

- **project_name**.
- **interaction_surfaces** — the surfaces the architecture actually uses (CLI,
  Web/GUI, TUI, API, AI Skill, MCP) and which are MVP vs later.
- **actors_roles** — primary/secondary users, reviewers, AI agents, MCP clients.
- **state_model** — lifecycle states, operational condition flags, audit events
  (keep them distinct, exactly as the architecture defines them).
- **workflows** — main workflow + critical sub-workflows (trigger, steps, gates,
  outputs).
- **human_review_flow** — triggers, review artifact/surface, allowed decisions,
  audit events (from the architecture's Experience Architecture / human-review
  flow).
- **security_trust** — trust boundaries, permission/authorization model.
- **data_egress_policy** — whether content leaves the boundary; redaction.
- **observability** — correlation IDs, the events UX can surface as feedback.
- **failure_recovery** — timeouts, retry, fallback, degraded behaviour, partial
  output, resume.
- **experience_architecture** — the architecture's §23 Experience Architecture
  (interaction surface matrix, user-visible state model, feedback/progress,
  error/recovery, human-review flow, trust/transparency, interaction
  observability, UX handoff).
- **ux_handoff** — the architecture's UX-Design Handoff (§24.3): constraints UX
  must honour and the open UX questions it raised.

For each required item, cite the architecture section. **If a required section
is missing, mark it `missing`** (do not invent it) and note whether to continue
with a warning or to fail (fail only if a truly UX-critical fact — surfaces,
state model, or human-review flow when review exists — is absent).

## Output

`intermediate/architecture_parse.json` with the keys above, plus
`missing_sections: [...]`.

## Validation / failure policy

- Gate: surfaces, state model, contracts, review flow, egress, and observability
  are extracted or explicitly marked missing.
- Failure policy: `warn_or_stop_if_critical_section_missing`.
