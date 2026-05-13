# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the
`research-pipeline` project.

ADRs capture the rationale behind significant technical decisions — including
alternatives considered and the consequences of the choice made.

## Format

Each ADR uses this structure:

```
# ADR-NNN: Title

## Status
Accepted | Deprecated | Superseded by ADR-XXX

## Date
YYYY-MM (approximate)

## Context
What forces, constraints, or problems led to this decision?

## Decision
What was decided, and how is it implemented?

## Consequences
What are the positive and negative outcomes?
```

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](ADR-001-uv-build-tool.md) | Uv as the Build and Dependency Manager | Accepted |
| [ADR-002](ADR-002-pydantic-domain-models.md) | Pydantic v2 for All Domain Models | Accepted |
| [ADR-003](ADR-003-stage-based-pipeline.md) | Stage-Based Idempotent Pipeline Architecture | Accepted |
| [ADR-004](ADR-004-conversion-backend-registry.md) | Backend Registry Pattern for PDF Conversion | Accepted |
| [ADR-005](ADR-005-sqlite-persistent-state.md) | SQLite for All Persistent State | Accepted |
| [ADR-006](ADR-006-fastmcp-server.md) | FastMCP for the MCP Server | Accepted |
| [ADR-007](ADR-007-multi-tier-memory.md) | Multi-Tier Memory Architecture | Accepted |
| [ADR-008](ADR-008-briefing-vs-academic-pipeline.md) | Separation of Academic Pipeline from Daily Briefing Pipeline | Accepted |

## Adding a New ADR

1. Copy the format above
2. Number sequentially from the highest existing number + 1
3. File name: `ADR-NNN-kebab-case-title.md`
4. Add a row to the index above
5. If the ADR supersedes an existing one, update the old ADR's Status field
