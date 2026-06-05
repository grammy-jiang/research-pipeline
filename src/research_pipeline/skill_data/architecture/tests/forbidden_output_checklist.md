# Checklist: Forbidden Output

The architecture skill DOES choose a tech stack — that is its job. But it must
not cross into implementation. The output FAILS if it contains:

- [ ] Source code / function bodies (beyond tiny illustrative signatures).
- [ ] Concrete database migrations or DDL (`CREATE TABLE …`, `ALTER TABLE …`).
- [ ] A detailed implementation task breakdown or coding tickets (that is the
      implementation-plan skill's job).
- [ ] Deployment scripts or CI/CD pipeline definitions.
- [ ] An MCP server introduced without passing the adoption gate.
- [ ] An AI component that mutates durable state without a deterministic
      validation gate.
- [ ] Every conceptual blueprint component promoted to a separate service
      without rationale.
- [ ] Secrets, API keys, or credentials of any kind.
- [ ] Hidden/unstated unresolved questions (they must appear in §23).
- [ ] File-by-file implementation steps, or package/module names presented as
      final file paths rather than labelled "proposed module namespaces".
- [ ] Enforcement/security claims a chosen technology does not actually provide
      (downgrade absolute wording to application-enforced / tamper-evident /
      best-effort and add a risk or ADR note).

Allowed (and expected): tech-stack names with rationale, schema-level data
objects (not migrations), interface contracts, C4 diagrams, ADRs, and
illustrative pseudo-signatures.
