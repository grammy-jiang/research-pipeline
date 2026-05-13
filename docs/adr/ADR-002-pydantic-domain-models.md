# ADR-002: Pydantic v2 for All Domain Models

## Status
Accepted

## Date
2024

## Context

The pipeline processes paper metadata, search results, download records,
conversion manifests, quality scores, and synthesis reports — all of which
must be:
- Serialisable to/from JSON (workspace artefacts persist to disk)
- Validated at ingestion boundaries (external API responses can be malformed)
- Type-safe for IDE support and mypy strict mode

Options considered:
1. Plain `dataclasses` — no validation, no serialisation support
2. `attrs` — validation but no built-in JSON round-trip
3. `pydantic` v1 — widely used, good ecosystem, but slower
4. `pydantic` v2 — same API, significantly faster, full mypy support

## Decision

Use **Pydantic v2 `BaseModel`** for all domain objects in
`src/research_pipeline/models/`. Every model must use type hints on all fields.
Optional fields use `default=None` or `default_factory` for backward
compatibility with existing workspace artefacts.

## Consequences

**Positive:**
- JSON serialisation/deserialisation via `.model_dump()` / `.model_validate()`
- Automatic validation of all external data
- mypy strict mode support
- Backward-compatible schema evolution (new optional fields don't break old artefacts)

**Negative:**
- Pydantic v2 API differs from v1 (migration cost if external code used v1 patterns)
- Pydantic is a runtime dependency for all consumers of the package

**Rules enforced in CI:**
- `mypy` strict mode with `pydantic` plugin
- Ruff rule `RUF012` is ignored (mutable class defaults — Pydantic uses them
  intentionally via `Field(default_factory=...)`)
