# Verification Policy

Verification must be deterministic.

Prefer these layers:

1. Unit tests.
2. Offline integration tests.
3. Offline end-to-end tests.
4. Report validator.
5. Artifact layout checks.
6. Linting/type checks.
7. Manual review only after deterministic checks pass.

Do not rely only on snapshot tests.

Tests must assert:

- schema fields
- stable IDs
- link counts
- item counts
- required report sections
- ranking scores
- tie-breaker ordering
- validation errors
- telemetry records where relevant
