# Prompt 08 — Surface-Specific UX

You are defining UX for **only the surfaces the architecture actually uses**. Do
not generate every possible surface section in full detail.

## Inputs

- `intermediate/architecture_parse.json` (interaction_surfaces, MVP vs later),
  `intermediate/user_stories.md`.
- `references/surface-ux-guide.md`.

## Instructions

1. From the architecture's interaction surfaces, include only the relevant §12
   sub-sections; mark the others "not used by this architecture" in one line.
   The candidate sub-sections are: **12.1 CLI**, **12.2 Web / GUI**, **12.3
   TUI**, **12.4 AI Skill**, **12.5 MCP**, **12.6 API / Automation**.
2. For each included surface, follow `references/surface-ux-guide.md`:
   - **CLI** — command groups conceptually, input/output behaviour, progress,
     human-readable vs machine-readable output, exit-code expectations, error
     display, structured-output requirements. (No final exact flags unless the
     architecture specifies them.)
   - **Web / GUI** — screen-map concept, major page types (dashboard / review /
     settings / audit), form behaviour, review interaction, empty/loading/error/
     degraded states. (No visual layout or pixel-level design.)
   - **TUI** — terminal panels, keyboard navigation, status indicators, review
     lists, confirmation flows.
   - **AI Skill** — invocation pattern, required inputs, clarification strategy,
     assumption recording, output behaviour, resume/update, safe refusal/
     escalation.
   - **MCP** — tool/resource discoverability, safe high-level vs low-level tools,
     tool descriptions, schema usability, permission prompts, dangerous-operation
     handling, agent-facing errors, auditability, read-only-by-default resources.
   - **API / Automation** — developer experience, job submission, status polling
     / callbacks, machine-readable errors, idempotency, batch behaviour.
3. **MVP discipline:** flag if too many surfaces are included for MVP, or a
   surface is included that the architecture does not support (the latter is
   architecture feedback).

## Output

`intermediate/surface_specific_ux.md` (the §12 content).

## Validation / failure policy

- Gate: only architecture-supported surfaces are covered; each maps to the user
  stories; no visual/pixel-level design.
- Failure policy: `revise`.
