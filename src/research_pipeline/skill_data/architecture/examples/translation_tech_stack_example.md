# Architecture Tech Stack Selection: LLM-Agent Document Translation System

> Worked example of the `architecture` skill **`stack` mode**, produced from
> `translation_blueprint_excerpt.md` + the `translation_architecture_example.md`
> design. It is an **example**, not a universal recommendation; choices are
> justified for this architecture only. Stack mode satisfies the architecture; it
> does not redesign it.

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
| Source blueprint | `translation_blueprint_excerpt.md` |
| Source architecture design | `translation_architecture_example.md` |
| Source architecture version | 0.6.0 |
| Architecture skill version | 0.6.0 |
| Mode | stack |
| Generated at | 2026-06-06 |
| Operating mode | hybrid |
| Decisions made | 10 |
| Output detail | standard |
| Target deployment assumption | server |

## Update History

| Date | Source Architecture | Stack Version | Change Type | Affected Areas | Notes |
|---|---|---|---|---|---|
| 2026-06-06 | `translation_architecture_example.md` | 0.1.0 | initial | all | First tech-stack selection from the architecture |

## 2. Source Architecture Summary

The architecture is a deterministic translation spine with a single bounded LLM
step, conditional reviewer escalation (MVP-1), append-only tamper-evident audit
(§13/§17), correlation-ID observability (§16), and an `external_allowed`
data-egress assumption pending confirmation (§17.9). The stack must satisfy:
strong-consistency state + audit, resumable long jobs, a swappable LLM provider
with egress logging and redaction, and a single-tenant server deployment. The
architecture defers MCP (ADR-0006); the stack honours that.

## 3. Technology Decision Drivers

| Driver | Source (architecture §) | Why It Constrains the Stack |
|---|---|---|
| Strong consistency for state + tamper-evident audit | §13 / §14 / §17 | Constrains the metadata store + audit model |
| Resumable long-running jobs (segment checkpointing) | §18 | Constrains the queue / job-execution model |
| Swappable LLM provider; egress logging + redaction | §17.9 / §16 | Constrains the provider abstraction + logging stack |
| Single-tenant server, MVP-0 simplicity → MVP scale path | §20 / §4.8 | Constrains storage + deployment/packaging |
| MCP deferred | §11 / ADR-0006 | No MCP SDK is introduced |

## 4. Stack Selection Summary

| Area | Selected Technology | Alternatives Considered | Rationale | Risk | Reversibility | Architecture Impact |
|---|---|---|---|---|---|---|
| Runtime / language | Python 3.12 | Go, TypeScript | LLM ecosystem + team familiarity; not CPU-bound | GIL (not the bottleneck) | high | none |
| Application framework | Typer CLI + thin FastAPI HTTP | Full web framework | Batch jobs, not interactive UI | — | high | none |
| Storage / data layer | PostgreSQL | SQLite, document store | Concurrency + durable audit on a multi-tenant path | Ops overhead vs SQLite | medium | §14/§20 |
| Queue / background jobs | PostgreSQL-backed durable queue | Redis, RabbitMQ | One store; transactional enqueue with state | Throughput ceiling | medium | §9/§18 |
| LLM provider abstraction | Provider adapter interface (in-house) | Direct SDK, LiteLLM | Isolate provider; enforce egress logging + redaction | Adapter drift | high | §16/§17 |
| AI / agent orchestration | None — deterministic worker | Agent framework | One bounded AI call needs no agent runtime | — | high | none |
| MCP framework / SDK | Defer (honour ADR-0006) | Adopt an MCP SDK | No reusable multi-client surface yet | Re-add later | high | none |
| Observability stack | OpenTelemetry + structured logs | Vendor APM | Correlation IDs + audit assertions; provider-neutral | Setup cost | high | §16 |
| Testing stack | pytest + golden fixtures + injection corpus | unittest | Deterministic golden + AI-eval + security tests | — | high | §19 |
| Deployment / packaging | Container image, single-tenant server | Serverless | Long-running stateful jobs | — | medium | §20 |

> Choices are justified from *this* architecture's drivers (§3), not as defaults.
> The audit store is **application-enforced + hash-chain tamper-evident**, not
> enforced by database grants — selecting PostgreSQL does not change that wording.

## 5. Runtime and Language Selection

Python 3.12 — confirms the design's §7.1 provisional assumption. The workload is
I/O- and LLM-bound, so the GIL is not a constraint; the LLM/tooling ecosystem and
team familiarity dominate. Reversible (the deterministic spine is language-agnostic).

## 6. Application Framework Selection

Typer CLI for batch submission + a thin FastAPI HTTP surface for status/poll and
the reviewer API (MVP-1). A full web framework is rejected — there is no
interactive UI at MVP. Satisfies §12 API contracts and the §23 interaction
surfaces.

## 7. Storage and Data Layer Selection

PostgreSQL resolves the §7.2 metadata-store handoff. SQLite would serve MVP-0 but
the architecture's multi-tenant path (§4.8) and concurrent reviewer access favour
PostgreSQL now to avoid a later migration. The append-only audit remains
application-enforced + hash-chain tamper-evident (§13/§17) — Postgres grants are
not credited with that guarantee.

## 8. Queue / Background Job Selection

A PostgreSQL-backed durable queue (transactional enqueue alongside job state)
resolves the §7.2 queue handoff. Redis/RabbitMQ are stronger at high throughput
but add a second stateful system; the single-tenant volume does not justify it
yet (revisit trigger: sustained queue depth). Supports §18 resumable jobs.

## 9. LLM / AI / Agent Stack Selection

An in-house provider adapter interface (not a framework) keeps the single bounded
call swappable and lets the §17.9 egress logging + redaction be enforced at the
boundary. No agent orchestration runtime — the architecture has exactly one
bounded AI step. LiteLLM was considered but adds a dependency whose value is low
for one call.

## 10. MCP / AI-Skill Integration Stack

DEFER — honours ADR-0006. No MCP SDK is introduced; there is no reusable
multi-client tool surface. Revisit if a second consumer needs the translation
tools.

## 11. Observability and Audit Stack

OpenTelemetry traces/metrics + structured JSON logs, correlation-ID propagated
per §16. Audit records persist in PostgreSQL with a hash chain (tamper-evident).
Redaction wraps the provider adapter so raw source content never reaches logs
(§17.12).

## 12. Testing Stack

pytest with: golden translation fixtures, a quality-gate calibration suite, an
LLM-output evaluation harness, a prompt-injection corpus, and audit/correlation-ID
assertions — covering the §19 testing architecture.

## 13. Deployment and Packaging Stack

A container image deployed to a single-tenant server; configuration + secrets via
environment/secret store (no secrets in the image). Matches §20.

## 14. Alternatives Considered

| Area | Chosen | Strongest Alternative | Why Not Chosen | Revisit Trigger |
|---|---|---|---|---|
| Storage | PostgreSQL | SQLite | Avoids a later migration on the multi-tenant path | If product stays strictly single-user local |
| Queue | PostgreSQL-backed | Redis | Avoids a second stateful system at MVP volume | Sustained queue depth / latency SLA |
| LLM | In-house adapter | LiteLLM | One bounded call; low value vs added dependency | Multiple providers/routing needed |

## 15. Risk and Reversibility

| Risk | Area | Impact | Likelihood | Mitigation | Reversibility |
|---|---|---|---|---|---|
| Postgres ops overhead at MVP-0 | Storage | M | M | Start single-node; managed Postgres option | medium |
| Queue throughput ceiling | Queue | M | L | Monitor depth; broker swap path documented | medium |
| Provider adapter drift | LLM | M | M | Contract tests against the adapter interface | high |

## 16. ADR Candidates

| Candidate ADR | Decision | Status | Notes |
|---|---|---|---|
| ADR-0002a | Storage = PostgreSQL (supersedes the design's provisional SQLite→Postgres note) | Proposed | `update` mode promotes + handles supersession |
| ADR-0007 | Queue = PostgreSQL-backed durable queue | Proposed | Revisit if throughput grows |

## 17. Architecture Impact Notes

- **PostgreSQL** touches §14 (state durability), §20 (deployment now needs a
  managed/standalone DB), and operational sections (backup/permissions).
- **PostgreSQL-backed queue** touches §9 (container/runtime view gains no broker)
  and §18 (recovery semantics tied to the DB transaction).
- **Provider adapter + redaction** touches §16/§17 (egress logging is the
  adapter's responsibility).
- No change to the product thesis, MVP scope, UX intent, or the AI boundary.

## 18. Architecture Update Required?

| Update Needed? | Affected Architecture Sections | Reason | Priority |
|---|---|---|---|
| Yes | §14, §20, §21 (ADR index) | Selecting PostgreSQL (over the provisional SQLite→Postgres path) and a DB-backed queue firms up state durability, deployment, backup/permissions, and supersedes the provisional storage ADR | medium |

> Run `architecture --mode update` to apply these into the design and supersede
> ADR-0002's provisional storage note. Stack mode declares this impact; it does
> not rewrite the architecture here.

## 19. Tech Stack Quality-Gate Self-Check

| Gate | Status | Finding | Required Action | Blocks Architecture Update? |
|---|---|---|---|---|
| Architecture requirements consumed | PASS | §3 drivers trace to architecture §13/§14/§17/§18/§20 | — | no |
| Alternatives considered | PASS | §14 + per-area rationale (SQLite, Redis, LiteLLM) | — | no |
| Risk and reversibility stated | PASS | §15 covers each high-impact choice | — | no |
| Security/privacy implications included | PASS | Egress logging + redaction at the provider adapter (§11/§16) | — | no |
| Architecture impact notes produced | PASS | §17 per technology | — | no |
| Architecture update requirement explicit | PASS | §18 = Yes (§14/§20/§21) | Run `update` mode | no |

> Status legend: PASS / WARNING / FAIL. Stack mode stays in scope — it declares
> the architecture update rather than rewriting the architecture or changing the
> product thesis / MVP / UX intent.
