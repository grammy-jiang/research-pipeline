# Tech-Stack Selection Guide (`stack` mode)

Load this in `stack` mode. Stack mode selects the **concrete technology stack**
that satisfies an already-defined architecture. It is a *different decision
type* from architecture design — see `references/mode-selection-guide.md`.

## Core discipline

- **Stack mode satisfies the architecture; it does not redesign it.** Consume
  the architecture's NFRs, contracts, state model, security/egress policy, AI
  boundary, and MCP decision as *requirements*, then choose technologies that
  meet them.
- **Never change** the product thesis, MVP scope, UX intent, or the core
  architecture. If you cannot satisfy the architecture without changing it, do
  **not** rewrite it — record the conflict in *Architecture Update Required?*
  and hand it to `update` mode.
- **Anti-default rule:** never apply a fixed stack as a universal default. The
  example stacks in `references/` and `examples/` are illustrations, not
  defaults. Justify each choice from *this* architecture's drivers.

## What stack mode owns

```text
language / runtime
backend / CLI / TUI / API framework
database / storage / artifact store / retrieval
queue / background jobs
deployment target
LLM provider abstraction
AI / agent orchestration
MCP framework / SDK (honour the architecture's MCP decision)
observability stack
testing stack
packaging / distribution
security-relevant stack choices
alternatives considered, risk, reversibility, ADR candidates, architecture impact
```

## What stack mode must NOT own

```text
the product thesis
the MVP scope
the core architecture
the UX intent
new workflows not present in the architecture
```

## Decision discipline

Every technology decision answers six questions:

1. Why this technology?
2. Why not the alternatives?
3. What architecture requirement does it satisfy?
4. What risk does it introduce?
5. How reversible is it (high / medium / low)?
6. Does it require an architecture update?

Use the decision table (§4 of `templates/architecture_tech_stack_template.md`):

```text
| Area | Selected Technology | Alternatives Considered | Rationale | Risk | Reversibility | Architecture Impact |
```

## Technology-specific validity

Only credit a technology with properties it actually provides. Do not describe
one technology's enforcement model in terms of another's. When a guarantee
depends on implementation discipline rather than the technology itself, downgrade
absolute wording ("guaranteed", "immutable", "no grants") to
"application-enforced", "tamper-evident", "best-effort", or "requires
operational control", and add a risk or ADR note. (Example: an append-only audit
log on an embedded store with no role/grant model is *application-enforced +
hash-chain tamper-evident*, not enforced by database grants — say so. This is an
illustration, not a recommendation.)

## Architecture Update Required? — the link to `update` mode

Stack mode ends with an explicit verdict:

```text
| Update Needed? | Affected Architecture Sections | Reason | Priority |
```

Typical triggers:

```text
embedded single-file store selected
  -> update audit / concurrency / deployment assumptions.

server database selected
  -> update deployment, backup, permission, and operational sections.

unifying LLM gateway selected
  -> update provider abstraction, observability, logging, dependency-risk.

MCP SDK selected
  -> update integration surface and security boundary.
```

This links `stack` mode to `update` mode without merging them: stack mode
*declares* the impact; `update` mode (design machinery) *applies* it and handles
ADR supersession and the Update History row.

## Stack quality gate (blocks the architecture update)

```text
Architecture requirements consumed
Alternatives considered
Risk and reversibility stated
Security/privacy implications included
Architecture impact notes produced
Architecture update requirement explicit
```

Fail if technologies are chosen without architecture requirements, no
alternatives are considered, selected technologies contradict architecture
constraints, security/privacy implications are ignored, or architecture impact
is missing.
