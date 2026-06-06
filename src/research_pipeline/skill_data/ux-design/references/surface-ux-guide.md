# Surface-Specific UX Guide

Load this for §12. Include **only the surfaces the architecture actually uses**;
mark the rest "not used by this architecture" in one line. Do not define final
exact flags, API schemas, visual layout, or copy — those belong to later stages.

## 12.1 CLI UX

```text
command groups (conceptually, not final flags)
input / output behaviour
progress behaviour (mapped to architecture observability / states)
human-readable vs machine-readable output (and how the user selects)
exit-code expectations
error display requirements
structured-output requirements (e.g. JSON for automation)
```

Do not define final exact CLI flags unless the architecture already specifies
them.

## 12.2 Web / GUI UX

```text
screen-map concept (which screens exist, not layout)
major page types: dashboard / review / settings / audit
form behaviour (validation feedback, not styling)
review interaction
empty / loading / error / degraded states
```

Do not define visual layout or pixel-level design.

## 12.3 TUI UX

```text
terminal panels
keyboard navigation model
status indicators
review lists
confirmation flows
```

## 12.4 AI Skill UX

```text
skill invocation pattern
required inputs
clarification strategy
assumption recording
output document behaviour
resume / update behaviour
safe refusal / escalation behaviour
```

## 12.5 MCP UX

```text
tool / resource discoverability
safe high-level tools vs low-level tools
tool descriptions (agent-facing clarity)
schema usability
permission prompts
dangerous-operation handling
agent-facing errors
auditability
read-only-by-default resources
```

## 12.6 API / Automation UX

```text
developer experience
job submission flow
status polling / callbacks
machine-readable errors
idempotency expectations
batch behaviour
```

## MVP discipline

- Cover only surfaces the architecture supports. Including a surface the
  architecture does not provide is **architecture feedback** (§21), not a UX
  decision.
- WARN if too many surfaces are included for MVP, if CLI-first is chosen for
  non-technical users without mitigation, if MCP is exposed without a clear
  agent user, or if a Web UI is deferred despite frequent human review.
