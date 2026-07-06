# Prompt 02 — Translate to Product Primitives

You are translating research items into product primitives.

## Confidence and gap biases

- HIGH-confidence items → bias toward ADOPT or MERGE.
- MEDIUM-confidence items → bias toward ADAPT or DEFER.
- LOW-confidence items → bias toward DEFER unless cheap and safe.
- `ACADEMIC` gap items → translate to **validation requirements / open
  questions**, never product requirements (unless the product's purpose
  is to answer the research question).
- `ENGINEERING` gap items → translate to **product requirements**, carrying
  their severity as a priority signal. Final MVP inclusion is decided only in
  Prompt 03.
- `OUT_OF_SCOPE` gap items → note as a non-goal; do not translate.

See `references/gap-type-mapping.md` for the full mapping.

## Translation rules

| Research item | Product translation |
|---|---|
| Taxonomy | Conceptual domain model |
| Mechanism | Product capability or decision policy |
| Algorithm | Decision rule, scoring policy, or workflow step |
| Workflow pattern | User/system workflow |
| Benchmark | Evaluation strategy |
| Security method | Governance or safety control |
| Data structure | Conceptual information object |
| Empirical result | Priority signal (HIGH → ADOPT; LOW → DEFER) |
| Assumption | Validation requirement |
| Contradiction | Explicit design decision |
| Academic gap | Research risk / future validation — NOT a requirement |
| Engineering gap | Product requirement or delivery risk |
| Risk | Mitigation and governance rule |
| Operational implication | Runtime/product constraint |
| Architecture hint | Logical component or integration surface |

## For each research item, produce one or more primitives

Allowed primitive types: capability, workflow, policy, conceptual
component, information object, evaluation requirement, governance rule,
risk control, user interaction, lifecycle state, integration surface.

Each primitive must capture:

- name
- type
- derived_from (the source research item id(s) **with citation**)
- rationale
- priority signal: high / medium / low / validation_only / out_of_scope
  (non-binding; final MVP inclusion is decided only in Prompt 03)

## Merge

After producing primitives, merge overlapping ones into a single coherent
capability. Record which source items each merged primitive draws from so
traceability survives the merge.

## Constraints

Stay implementation-neutral. Do **not** name programming languages,
frameworks, databases, vector databases, cloud services, UI libraries, or
deployment models. Use neutral conceptual component names (e.g.
"Admission Controller", "Retrieval Orchestrator"). When uncertain, apply
the borderline-case decision rule in `references/borderline-cases.md`: if
the constraint survives replacing the specific term with its purpose, keep
the conceptual version; otherwise it belongs to the technical-design skill.
