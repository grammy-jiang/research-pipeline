# Rule Pack: Boundary Rules

Apply as a review gate (prompt 21), not as essay content in the output.

- [ ] Domain/application logic is not coupled to framework details.
- [ ] AI output cannot mutate durable state directly (a deterministic
      validation gate sits between any AI proposal and any state write).
- [ ] External providers are behind adapters (no direct provider calls leaking
      through the application core).
- [ ] Security boundaries are explicit (every cross-zone edge names a control).
- [ ] Conceptual blueprint components are not each promoted to a separate
      service without rationale.
- [ ] Each container/module has a single clear responsibility and owner.

**Gate:** if any box is unchecked, mark the relevant section FAIL in the
self-check and revise (max 3 attempts), then surface the failing rule.
