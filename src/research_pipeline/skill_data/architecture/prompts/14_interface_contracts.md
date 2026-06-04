# Prompt 14 — Interface Contracts

You are defining the contracts at every major module boundary.

## Inputs

- `intermediate/c4_views.md` (the containers/components and their edges).
- `templates/interface_contract_template.md`.

## Instructions

For every major boundary, define: owner; input contract; output contract; error
contract; validation rules; versioning expectation; observability fields.

Produce these subsections in §12:

- 12.1 API Contracts
- 12.2 Event Contracts
- 12.3 Internal Module Contracts
- 12.4 Agent Input/Output Contracts
- 12.5 Tool Schemas
- 12.6 MCP Resources and Tools (only if MCP adopted; else state "n/a — MCP
  deferred")
- 12.7 Error Model (categories, retryable?, what surfaces to the caller)
- 12.8 Versioning Rules

Keep interfaces small and deep. Make agent/tool I/O structured so it is
contract-testable (§19).

## Output

`intermediate/interface_contracts.md` → populates §12.

## Validation / failure policy

- Gate: every major boundary has a contract including an error contract and a
  versioning expectation.
- Failure policy: `revise`.
