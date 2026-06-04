# Rule Pack: Data Rules

Apply as a review gate (prompt 21).

- [ ] Every stored object has a clear storage owner.
- [ ] Retention is defined for every store and artifact.
- [ ] Schema evolution strategy is considered (additive-first, migration path).
- [ ] Audit record immutability is defined (append-only semantics).
- [ ] Large artifacts are not forced into the metadata DB (metadata and blobs
      are separated).
- [ ] Sensitive data is classified, and classification drives storage and
      retention choices.

**Gate:** unchecked items block §13/§14 and the data-related ADR.
