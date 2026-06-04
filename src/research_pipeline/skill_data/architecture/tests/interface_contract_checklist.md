# Checklist: Interface Contracts

For every major module boundary:

- [ ] Owner named.
- [ ] Input contract defined.
- [ ] Output contract defined.
- [ ] Error contract defined (not just the happy path).
- [ ] Validation rules defined.
- [ ] Versioning expectation defined.
- [ ] Observability fields named.

Section coverage in §12:

- [ ] API Contracts
- [ ] Event Contracts
- [ ] Internal Module Contracts
- [ ] Agent Input/Output Contracts
- [ ] Tool Schemas
- [ ] MCP Resources and Tools (or "n/a — MCP deferred")
- [ ] Error Model
- [ ] Versioning Rules
- [ ] Interfaces are small and deep; agent/tool I/O is structured and
      contract-testable.
