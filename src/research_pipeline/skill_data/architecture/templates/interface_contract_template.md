# Interface Contract Template

Every module boundary must have: owner, input contract, output contract, error
contract, validation rules, versioning expectation, and observability fields.

## Per-interface skeleton

```markdown
### <Interface name>

- **Owner:** <module/team>
- **Kind:** API / event / internal module / agent I/O / tool schema / MCP
- **Input contract:** <fields, types, required/optional, constraints>
- **Output contract:** <fields, types, success shape>
- **Error contract:** <error codes/categories, retryable?, error body>
- **Validation rules:** <deterministic checks applied at the boundary>
- **Versioning:** <semver / additive-only / deprecation policy>
- **Observability fields:** <correlation IDs emitted, audit events>
```

## Required interface sections in §12

```markdown
## 12. Interface Contracts

### 12.1 API Contracts
### 12.2 Event Contracts
### 12.3 Internal Module Contracts
### 12.4 Agent Input/Output Contracts
### 12.5 Tool Schemas
### 12.6 MCP Resources and Tools
### 12.7 Error Model
### 12.8 Versioning Rules
```

## Error Model skeleton

| Category | Example | Retryable? | Surface to caller |
|---|---|---|---|
| Validation | bad input shape | no | yes (400-class) |
| Transient | provider timeout | yes (backoff) | maybe |
| Permission | tool call denied | no | yes |
| Internal | unexpected | no | sanitized only |

## Versioning rules

- Additive changes are backward-compatible; breaking changes require a version
  bump and a deprecation window.
- Agent and tool I/O should be **structured** wherever possible so contracts
  are testable (contract tests in §19).
- Keep interfaces **small and deep**: a narrow surface hiding substantial
  behavior, not a wide surface leaking internals.
