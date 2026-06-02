# Prompt 03 — Resolve Ideas

You are resolving research-derived ideas into product-design decisions.

For each major idea, choose exactly one:

- **ADOPT** — central, evidence-backed, aligned with the product.
- **ADAPT** — use, but modify for the target product context.
- **MERGE** — combine multiple related ideas into one capability.
- **DEFER** — valuable but not MVP-critical (or MEDIUM/LOW confidence).
- **REJECT** — weak, redundant, unsafe, too speculative, or out of scope.
- **DEFER / VALIDATE** — an unresolved `ACADEMIC` gap that may be
  valuable but is not ready to become a product requirement.

## Criteria

- evidence confidence grade (HIGH / MEDIUM / LOW)
- relevance to the product thesis
- MVP necessity
- complexity
- risk
- dependency on unresolved research (`ACADEMIC` gaps)
- safety or governance impact
- Round History signal (reaching the configured gap-closure maximum with
  many remaining gaps → apply conservative DEFER/REJECT pressure)
- Readiness Assessment verdict (`HAS_GAPS` → flag affected capabilities
  as requiring validation before release)

## Decision discipline

1. HIGH-confidence, product-critical ideas → usually ADOPT or MERGE.
2. MEDIUM-confidence ideas → usually ADAPT or DEFER.
3. LOW-confidence ideas → usually DEFER unless cheap and low-risk.
4. `ACADEMIC` gaps must not become MVP requirements unless the product's
   purpose is research validation.
5. Security-critical gaps must become risk controls or release gates.
6. Ideas needing model retraining, infra-heavy work, or exotic capability
   → usually DEFER.
7. Ideas that make the MVP too large → DEFER unless required for safety or
   correctness.

## Output

Return a decision table:

| Source Idea | Research Citation | Decision | Product Translation | Rationale | MVP? | Related Risks |
|---|---|---|---|---|---|---|

Be conservative. It is better to ship a small, correct MVP and defer
research-interesting extensions than to bloat the product with weakly
evidenced ideas.
