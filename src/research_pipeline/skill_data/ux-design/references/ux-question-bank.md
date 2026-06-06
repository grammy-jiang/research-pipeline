# UX Question Bank

Load this for the clarification pass (`prompts/04`) and to support input
discovery. Ask **only high-impact** questions, one at a time, each with a
recommended answer, alternatives, and a stated default.

## Question selection rule

Ask a question only if the answer affects one of:

```text
MVP scope
architecture constraints
security / privacy
the state model
human review
failure / recovery
E2E scenarios
implementation-plan readiness
```

**Do not ask** low-value visual/copy questions (they belong to implementation or
visual design):

```text
What colour should this button be?
Should the dashboard use cards or tables?
What exact wording should the warning message use?
What exact CLI flag should be used?
```

## Question format (one at a time)

```markdown
### UX Clarification Question <N>

**Question:** ...
**Why this matters:** ...
**Recommended answer:** ...
**Alternatives:**
- ...
**If no answer is provided:** The UX design will assume ...
```

## The eight categories

### Category 1 — Users and Roles
- Who is the first real user: technical operator, non-technical user, reviewer,
  team, or another AI agent?
- Which users are MVP users, and which are future users?
- Is the human reviewer a frequent user or an exception-path actor?
- Does the product need to support multiple user roles in MVP?

### Category 2 — Workflow
- What is the main happy-path workflow?
- Is the product used one job at a time, in batch, or inside an automation
  pipeline?
- Should the user be guided step-by-step or allowed expert-mode commands?
- Which workflow must be fastest? Which must be safest?

### Category 3 — Interaction Surface
- Is MVP-0 CLI-first, Web-first, API-first, AI-skill-first, MCP-first, or hybrid?
- Should CLI output be human-readable, machine-readable, or both?
- Should a future Web UI be a dashboard, review console, admin panel, or all?
- Should MCP expose low-level operations or safe high-level operations?

### Category 4 — Trust, Control, and Transparency
- What must the user see before trusting the output?
- Should quality scores be technical, simplified, or both?
- Should the user see why a reviewer chain was triggered?
- Should data-egress status be shown every time or only on request?
- Should the user be able to inspect audit evidence directly?

### Category 5 — Failure and Recovery
- When a job fails, should the user retry, edit input, switch route, export
  diagnostics, or escalate?
- Should partial output be available after failure?
- Should degraded output be allowed, blocked, or marked review-required?
- Should failed jobs be resumable?
- Should the user be told which component failed?

### Category 6 — Human Review
- Should human review happen inside the product or via an exported review packet?
- What should a reviewer see first: source, output, quality issues, or
  recommended focus?
- Can a reviewer approve, reject, edit, comment, or request rerun?
- Should reviewer decisions be audited?
- Should reviewer edits become training/evaluation data later?

### Category 7 — AI Skill and MCP
- Should another AI agent call this product through a skill, MCP, CLI, or API?
- Should MCP tools expose raw power-user operations or only safe high-level ones?
- What should happen if an AI agent asks for an unsafe operation?
- Should AI-agent interaction be conversational, tool-call based, or both?
- Should MCP resources be read-only by default?

### Category 8 — E2E Test Orientation
- Which user stories must become end-to-end tests?
- Which failure paths must be tested before release?
- Which user-visible states must be testable?
- Which surfaces must have E2E coverage: CLI, Web, API, MCP, AI skill?

## Input-discovery support

When invoked without an explicit architecture path, search the working
directory, `./docs`, `./design`, `./artifacts`. Rank candidates by: filename
`*-architecture-design.md`; header `# Architecture Design`; presence of
`Experience Architecture`, a state-model section, `Interface Contracts`,
`Recommended Next Stages`; topic-slug match with a nearby blueprint; latest
modified time. If still ambiguous, ASK_USER. If none found, STOP and tell the
user to run `architecture --mode design` first.
