# Adaptive Stage-Gate Routing (§19)

This reference governs §19 of the blueprint — **Recommended Next Stages**. The
blueprint is the first artifact that understands the product shape, so it is the
first **adaptive stage-gate router** after research extraction. §19 recommends
which downstream stages should run next, using controlled decisions with
evidence, confidence, and revisit triggers. It must **not** silently expand the
pipeline — every optional gate is justified, deferred, or skipped on purpose.

## Why routing starts at the blueprint

`research-pipeline` extracts ideas, methods, evidence, and gaps; it does not yet
know the product shape — target product, users, MVP scope, interaction mode,
risk level, AI involvement, human review, data sensitivity, workflow
complexity, or implementation ambiguity. The blueprint makes these explicit, so
it is the right place to decide which specialized design stages are justified.

```text
research-pipeline = evidence extraction
blueprint         = first product-level routing point
```

## Core path + adaptive optional gates

The pipeline is not a rigid waterfall. There is a core path, plus optional gates
recommended only when justified.

| Track | Stages |
|---|---|
| Core path | research-pipeline → blueprint → architecture-design → implementation-plan → implementation |
| Optional gates | tech-stack-selection · ux-design · security-review · test-design · architecture-update · architecture-reconciliation · release-plan |

## Decision vocabulary (controlled values only)

Every stage recommendation uses exactly one of:

| Decision | Meaning | Use when |
|---|---|---|
| **RUN** | Execute this stage before proceeding too far downstream. | The stage reduces major ambiguity, risk, rework, or test failure. |
| **SKIP** | Not useful for this project at the current scope. | The project is simple, low-risk, or the decision is already fixed. |
| **DEFER** | Do not run now; reconsider after another stage. | The blueprint lacks detail, but a later artifact will clarify it. |
| **ASK_USER** | The skill lacks information and the decision materially affects cost, privacy, security, scope, or UX. | An automatic assumption would be risky. |

Do **not** use vague wording — `maybe`, `consider`, `potentially useful`,
`nice to have`. It is not operational. Every RUN/ASK_USER needs evidence; every
SKIP needs a reason; every DEFER needs a revisit trigger. Recommendations are
**overrideable** by the user — present them as defaults, not mandates.

## Pipeline complexity assessment

Score seven dimensions 0–3 (0 = not relevant, 1 = low, 2 = medium, 3 = high):

```text
User-facing complexity
Technical ambiguity
Security / privacy risk
AI / LLM uncertainty
Integration complexity
Human-review complexity
Testing / E2E importance
```

| Total (/21) | Workflow class | Routing posture |
|---|---|---|
| 0–3 | simple | Skip most optional stages unless explicitly requested. |
| 4–7 | lightweight | A few lightweight gates may help; defer uncertain ones. |
| 8–12 | medium | Run the relevant optional gates. |
| 13+ | complex | Recommend the full gated workflow. |

This is a disciplined reasoning aid, not a scientific score.

## Per-stage rules

### architecture-design — default RUN

For almost every serious blueprint, architecture-design RUNs: the blueprint says
what should exist; architecture says how it is structured. SKIP only when the
output is a conceptual note, no implementation is planned, or it is a trivial
script with obvious architecture.

### tech-stack-selection

- **RUN** — multiple serious technology choices exist; storage/database choice
  affects architecture; deployment target unclear; LLM provider / orchestration
  matters; MCP / AI-agent integration affects the stack; performance/cost/
  security constraints drive technology; the team has preferences that should
  guide it.
- **SKIP** — small script; stack already fixed; choice obvious and low-risk;
  architecture can safely include the stack inline.
- **DEFER** — architecture-design must first clarify containers, runtime
  boundaries, or data flows.
- **ASK_USER** — the choice depends on budget, hosting, compliance, licensing,
  or team constraints not present in the blueprint.

### ux-design

- **RUN** — multiple user roles; human review exists; CLI/Web/TUI/MCP/AI-skill
  interaction is non-trivial; failure/recovery UX matters; user stories should
  drive E2E tests; non-technical users; significant trust/control/transparency
  needs.
- **SKIP** — backend-only; a simple one-shot command; blueprint + architecture
  already define enough interaction detail.
- **DEFER** — architecture-design must first define states, contracts,
  permissions, and review artifacts.
- **ASK_USER** — the primary interaction mode is unclear, or there is a major
  CLI-first vs Web-first vs AI-skill-first vs MCP-first trade-off.
- **Blueprint default:** `ux-design = DEFER until after architecture-design`
  unless the product is highly UX-led. Use §9 Product Experience Direction to
  decide.

### security-review

- **RUN** — external LLM data egress; private/sensitive data; authentication /
  authorization matters; MCP tools expose meaningful capabilities; audit /
  compliance requirements; multi-user access; secrets or provider credentials.
- **SKIP** — single-user local toy; no sensitive data; no external providers; no
  durable state; no permissions boundary.
- **DEFER** — architecture-design must first define data flow, trust boundaries,
  and storage.
- **ASK_USER** — data sensitivity, compliance requirement, or deployment
  environment is unclear.

### test-design

- **RUN** — user stories should become E2E tests; workflow correctness matters;
  AI outputs need evaluation; failure/recovery paths matter; a human-review path
  exists; multiple integration surfaces.
- **SKIP** — exploratory only; no implementation planned; a one-off prototype
  with no workflow guarantees.
- **DEFER** — ux-design should first define user stories and E2E scenario seeds.
- **Blueprint default:** often `test-design = DEFER until after ux-design or
  implementation-plan`, or `RUN later`.

### architecture-update — blueprint default DEFER

- **RUN** — tech-stack-selection changes runtime assumptions; ux-design exposes
  missing states/contracts/surfaces/review flows; security-review changes data
  flow or trust boundaries; implementation-plan finds an architecture conflict.
- **SKIP** — no downstream stage changes architecture assumptions.
- **DEFER** — no architecture document exists yet. **At blueprint stage this is
  always DEFER**, revisited after architecture-design and the gates that change
  its assumptions.

### architecture-reconciliation — blueprint default DEFER

A conditional repair / reconciliation mode, never a mandatory stage.

- **RUN** — ux-design conflicts with architecture; test-design finds impossible
  E2E scenarios; security-review invalidates architecture assumptions;
  implementation-plan exposes missing contracts or states.
- **DEFER** — at blueprint stage (no architecture to reconcile yet); revisit
  when a conflict is detected.

## Integrate Product Experience Direction (§9)

§9 makes the routing actionable, not decorative:

| §9 signal | Routing implication |
|---|---|
| Human review exists | ux-design RUN or DEFER; test-design RUN or DEFER; architecture must define review states/artifacts. |
| External data egress exists | security-review RUN; architecture must define the trust boundary; ux-design may need confirmation UX. |
| CLI-first | ux-design may DEFER unless CLI UX is complex; implementation-plan later needs CLI E2E tests. |
| Future MCP | tech-stack-selection RUN or DEFER; architecture defines an MCP extension point but not an MVP container. |

## Output discipline (§19 stays compact)

§19 is **one compact section**: one complexity table, one stage-recommendation
table, one short recommended pipeline, one decision-log table. Use scoring and
tables; avoid essays; do not describe full downstream skill outputs.

## Adaptive Stage-Gate Recommendation Gate

Add these rows to the Appendix A self-check (status `PASS` / `WARNING` / `FAIL`,
with a required action and a blocks-handoff verdict).

| Gate | Checks |
|---|---|
| Recommended Next Stages section exists | §19 is present with a complexity assessment and a stage-recommendation table. |
| Controlled decision values used | Every stage uses only RUN / SKIP / DEFER / ASK_USER. |
| RUN decisions have evidence | Each RUN cites blueprint evidence (a section, risk, or §9 signal). |
| SKIP decisions have reason | Each SKIP states why the stage is not useful now. |
| DEFER decisions have revisit trigger | Each DEFER names the artifact/condition that revisits it. |
| ASK_USER decisions identify missing info | Each ASK_USER names the missing decision input. |
| Product Experience Direction informs recommendations | §9 signals are reflected in the UX/security/test decisions. |

### Fail conditions

`FAIL` if: the Recommended Next Stages section is missing; a decision uses
uncontrolled wording; a RUN lacks evidence; an ASK_USER does not identify the
missing information; architecture-design is skipped without strong
justification; or a high-risk project recommends no optional gates.

### Warning conditions

`WARNING` if: ux-design is skipped despite human review or multiple user roles;
security-review is skipped despite external data egress; test-design is skipped
despite E2E-critical workflows; or tech-stack-selection is skipped despite
multiple serious technology choices.
