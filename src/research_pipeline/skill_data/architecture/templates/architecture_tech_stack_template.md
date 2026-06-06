# Architecture Tech Stack Selection: <Project Name>

> Skeleton for the `architecture` skill **`stack` mode** output. Replace every
> `<…>` placeholder. Keep the section order and headings. Co-locate the file
> with the architecture design as `<topic-slug>-architecture-tech-stack.md`.
>
> Stack mode **selects technologies that satisfy the architecture**. It must not
> change the product thesis, MVP scope, UX intent, or the core architecture. If
> a genuine conflict is found, record it in §18 *Architecture Update Required?*
> rather than rewriting the architecture (that is `update` mode's job).

## Contents

- [1. Generation Metadata](#1-generation-metadata)
- [Update History](#update-history)
- [2. Source Architecture Summary](#2-source-architecture-summary)
- [3. Technology Decision Drivers](#3-technology-decision-drivers)
- [4. Stack Selection Summary](#4-stack-selection-summary)
- [5. Runtime and Language Selection](#5-runtime-and-language-selection)
- [6. Application Framework Selection](#6-application-framework-selection)
- [7. Storage and Data Layer Selection](#7-storage-and-data-layer-selection)
- [8. Queue / Background Job Selection](#8-queue--background-job-selection)
- [9. LLM / AI / Agent Stack Selection](#9-llm--ai--agent-stack-selection)
- [10. MCP / AI-Skill Integration Stack](#10-mcp--ai-skill-integration-stack)
- [11. Observability and Audit Stack](#11-observability-and-audit-stack)
- [12. Testing Stack](#12-testing-stack)
- [13. Deployment and Packaging Stack](#13-deployment-and-packaging-stack)
- [14. Alternatives Considered](#14-alternatives-considered)
- [15. Risk and Reversibility](#15-risk-and-reversibility)
- [16. ADR Candidates](#16-adr-candidates)
- [17. Architecture Impact Notes](#17-architecture-impact-notes)
- [18. Architecture Update Required?](#18-architecture-update-required)
- [19. Tech Stack Quality-Gate Self-Check](#19-tech-stack-quality-gate-self-check)

---

## 1. Generation Metadata

| Field | Value |
|---|---|
| Source blueprint | `<filename or unknown>` |
| Source architecture design | `<filename>` |
| Source architecture version | `<version/hash or unknown>` |
| Architecture skill version | `<from manifest.json version or unknown>` |
| Mode | stack |
| Generated at | `<date>` |
| Operating mode | interactive / automatic / hybrid |
| Decisions made | `<N>` (= §4 rows) |
| Output detail | concise / standard / detailed |
| Target deployment assumption | local / server / cloud / hybrid / unknown |

> Do not invent metadata. Use `unknown` when unavailable.

## Update History

| Date | Source Architecture | Stack Version | Change Type | Affected Areas | Notes |
|---|---|---|---|---|---|
| <YYYY-MM-DD> | `<architecture filename>` | 0.1.0 | initial | all | First tech-stack selection from architecture |

> New documents include the initial row. Updates append a new row. Never delete
> prior rows. Change Type ∈ {initial, regenerate, patch, compare}.

## 2. Source Architecture Summary

<What the architecture requires of the stack: runtime shape, AI boundary, MCP
decision, deterministic spine, data-egress policy, state model, and the
non-functional requirements that constrain technology. Cite architecture
sections. Do not restate the whole architecture — only what drives stack
choices.>

## 3. Technology Decision Drivers

| Driver | Source (architecture §) | Why It Constrains the Stack |
|---|---|---|
| <e.g. external-model data egress policy> | §17 / §3 | <constrains provider abstraction + logging> |
| <e.g. p95 latency target> | §4.3 | <constrains runtime + storage> |
| <e.g. single-user local-first deployment> | §20 | <constrains storage + packaging> |

## 4. Stack Selection Summary

The headline selection (one row per area; full reasoning in §5–§13 and the
decision table below):

| Area | Selected Technology | Alternatives Considered | Rationale | Risk | Reversibility | Architecture Impact |
|---|---|---|---|---|---|---|
| Runtime / language | <choice> | <alts> | <fit to architecture requirement> | <risk> | high/medium/low | <§ affected or none> |
| Application framework | <choice> | <alts> | <rationale> | <risk> | high/medium/low | <impact> |
| Storage / data layer | <choice> | <alts> | <rationale> | <risk> | high/medium/low | <impact> |
| Queue / background jobs | <choice / none> | <alts> | <rationale> | <risk> | high/medium/low | <impact> |
| LLM provider abstraction | <choice> | <alts> | <rationale> | <risk> | high/medium/low | <impact> |
| AI / agent orchestration | <choice / none> | <alts> | <rationale> | <risk> | high/medium/low | <impact> |
| MCP framework / SDK | <choice / defer> | <alts> | <rationale> | <risk> | high/medium/low | <impact> |
| Observability stack | <choice> | <alts> | <rationale> | <risk> | high/medium/low | <impact> |
| Testing stack | <choice> | <alts> | <rationale> | <risk> | high/medium/low | <impact> |
| Deployment / packaging | <choice> | <alts> | <rationale> | <risk> | high/medium/low | <impact> |

> Every choice answers: Why this technology? Why not the alternatives? What
> architecture requirement does it satisfy? What risk does it introduce? How
> reversible is it? Does it require an architecture update?
>
> **Anti-default rule:** never apply a fixed stack (e.g. always
> Python/FastAPI/PostgreSQL) as a universal default. Justify each choice from
> *this* architecture's drivers.
>
> **Technology-specific validity:** only credit a technology with properties it
> actually provides. Downgrade absolute wording ("guaranteed", "immutable", "no
> grants") to "application-enforced", "tamper-evident", "best-effort", or
> "requires operational control", with a risk or ADR note.

## 5. Runtime and Language Selection

<Decision + why; how it satisfies §2/§3 drivers; risk; reversibility.>

## 6. Application Framework Selection

<Backend / CLI / TUI / API framework as the architecture needs. Decision + why
not the alternatives.>

## 7. Storage and Data Layer Selection

<Metadata store, artifact store, search/retrieval. Tie to the §14 state model
and §13 data contracts. Note audit/immutability semantics honestly.>

## 8. Queue / Background Job Selection

<Job execution model, or "none — synchronous" with justification.>

## 9. LLM / AI / Agent Stack Selection

<Provider abstraction, model routing, agent orchestration. Tie to the §6 AI
boundary, §17 data egress, and §16 observability requirements.>

## 10. MCP / AI-Skill Integration Stack

<MCP SDK / framework if MCP is adopted, or DEFER with the reason. Honour the
architecture's MCP decision; do not introduce MCP because it is fashionable.>

## 11. Observability and Audit Stack

<Logging, metrics, tracing, audit storage. Must support the §16 correlation-ID,
audit-trail, and (for external-model systems) redaction requirements.>

## 12. Testing Stack

<Unit / integration / contract / e2e / AI-evaluation / security / failure-mode
tooling that covers the architecture's §19 testing strategy.>

## 13. Deployment and Packaging Stack

<Deployment target, packaging/distribution, configuration/secrets handling.>

## 14. Alternatives Considered

| Area | Chosen | Strongest Alternative | Why Not Chosen | Revisit Trigger |
|---|---|---|---|---|
| <area> | <choice> | <alt> | <reason> | <trigger> |

## 15. Risk and Reversibility

| Risk | Area | Impact | Likelihood | Mitigation | Reversibility |
|---|---|---|---|---|---|
| <risk> | <area> | H/M/L | H/M/L | <mitigation> | high/medium/low |

## 16. ADR Candidates

| Candidate ADR | Decision | Status | Notes |
|---|---|---|---|
| ADR-00NN | <e.g. storage backend selection> | Proposed | <feeds `update` mode / design ADR index> |

> Stack mode proposes ADR candidates; the design `update` mode promotes them
> into the architecture ADR register and handles supersession.

## 17. Architecture Impact Notes

<Per selected technology, the architecture sections it touches and how. This is
the input to `update` mode. Keep it as notes, not an architecture rewrite.>

## 18. Architecture Update Required?

| Update Needed? | Affected Architecture Sections | Reason | Priority |
|---|---|---|---|
| Yes / No | <e.g. §14, §16, §17, §20> | <e.g. selecting an embedded store changes audit/concurrency/deployment assumptions> | high/medium/low |

> Examples: choosing an embedded single-file store → update audit / concurrency
> / deployment assumptions; choosing a server database → update deployment,
> backup, permission, operational sections; choosing a unifying LLM gateway →
> update provider-abstraction, observability, logging, dependency-risk sections;
> adopting an MCP SDK → update the integration surface and security boundary.
> This links `stack` mode to `update` mode without merging them.

## 19. Tech Stack Quality-Gate Self-Check

| Gate | Status | Finding | Required Action | Blocks Architecture Update? |
|---|---|---|---|---|
| Architecture requirements consumed | PASS / WARNING / FAIL | <finding> | <action> | yes/no |
| Alternatives considered | PASS / WARNING / FAIL | <finding> | <action> | yes/no |
| Risk and reversibility stated | PASS / WARNING / FAIL | <finding> | <action> | yes/no |
| Security/privacy implications included | PASS / WARNING / FAIL | <finding> | <action> | yes/no |
| Architecture impact notes produced | PASS / WARNING / FAIL | <finding> | <action> | yes/no |
| Architecture update requirement explicit | PASS / WARNING / FAIL | <finding> | <action> | yes/no |

> Status legend: **PASS** (complete + consistent), **WARNING** ("PASS with
> warning" — acceptable direction, needs cleanup; non-blocking but carries a
> required action), **FAIL** (missing/false/misleading).
>
> **Fail conditions:** technologies chosen without architecture requirements; no
> alternatives considered; selected technologies contradict architecture
> constraints; security/privacy implications ignored; architecture impact
> missing.
