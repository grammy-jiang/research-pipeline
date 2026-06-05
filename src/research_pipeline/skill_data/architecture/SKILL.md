---
name: architecture
description: >
  Convert a product blueprint into a concrete technical architecture and
  tech-stack design: C4 views, a Traditional-Software-vs-AI responsibility
  matrix, a justified tech stack, skill/MCP boundaries, interface and data
  contracts, security/trust boundaries, observability/audit, ADRs, and
  implementation-planning handoff notes. The third stage of
  research-pipeline → blueprint → architecture → implementation-plan. Use
  when the user has a product blueprint and asks to "convert this blueprint
  to architecture", "design the technical architecture", "choose the tech
  stack", "create C4 diagrams", or "define module/interface/AI/MCP
  boundaries". Aliases: blueprint-to-architecture, technical-architecture,
  architecture-design, system-architecture, tech-stack-design. Do NOT use
  for research (use `research-pipeline`), blueprint generation (use
  `blueprint`), implementation task breakdown (use implementation-plan), or
  writing code.
license: MIT
---

# Blueprint-to-Architecture Design

This skill is the third stage of a research-driven development workflow:

```text
research-pipeline → blueprint → architecture → implementation-plan → implementation
```

The `blueprint` skill answers: **what product or workflow should exist?**

This `architecture` skill answers: **how should this product be technically
structured, implemented, integrated, observed, secured, and governed?**

The later `implementation-plan` skill answers: **what exact tasks should the
coding agent build, in what order?**

It consumes an implementation-neutral product blueprint and produces a
concrete technical architecture. It integrates four method sources as
workflow mechanics — not as text to paste into the output:

1. **C4 Model** — architecture visualization structure (context → container →
   component → dynamic → deployment).
2. **grill-me-style interaction** — one-question-at-a-time clarification for
   high-impact, irreversible, or risk-bearing decisions.
3. **agent-rules-books-style rule packs** — compact engineering quality gates
   applied as review checks (`rule-packs/`).
4. **Stage-gated workflow discipline** — explicit artifacts, assumptions,
   downstream impact, and review gates.

## When To Trigger

- "Convert this blueprint to architecture" / "architecture for this blueprint"
- "Design the technical architecture" / "generate architecture design"
- "Choose the tech stack" / "turn product blueprint into system architecture"
- "Create C4 diagrams" / "define module boundaries" / "define interfaces
  between modules" / "define data contracts"
- "Define AI boundaries" / "define MCP architecture" / "define observability
  and telemetry" / "define ADRs for this blueprint"
- The aliases `blueprint-to-architecture`, `technical-architecture`,
  `architecture-design`, `system-architecture`, `tech-stack-design`
- Accepting the `blueprint` skill's technical-design handoff (its
  §17 Handoff Notes for Technical Design).

## Do Not Use For

- Raw idea intake → idea/product workflow skill
- Research synthesis, paper review → `research-pipeline`
- Product blueprint generation → `blueprint`
- Detailed implementation task breakdown → the implementation-plan skill
- Code generation, repository modification, deployment scripting, or database
  migration generation → the coding agent

## Inputs

**Required:**

- One product blueprint Markdown document, normally
  `<topic-slug>-product-blueprint.md` (produced by the `blueprint` skill).

**Optional:**

- An existing `<topic-slug>-architecture-design.md` to update
- `adr/ADR-*.md` from a prior run
- `project_name`, `target_deployment` (local / server / cloud / hybrid),
  `constraints`, `excluded_scopes`
- `operating_mode`: interactive / automatic / hybrid (default: hybrid)
- `output_detail`: concise / standard / detailed (default: standard)

If no blueprint path is supplied, discover it automatically (see
`references/input-discovery.md`): prefer the most recently discussed blueprint
in context, then files matching `*-product-blueprint.md` / `*blueprint*.md` in
the working directory and common design directories, scored by the table in
that reference. If no candidate is found, STOP and ask for the path.

## Output

Primary output, co-located with the source blueprint unless told otherwise:

```text
<topic-slug>-architecture-design.md
```

Optional ADR files under `adr/` (one per high-impact decision):

```text
adr/ADR-0001-runtime-architecture.md
adr/ADR-0002-tech-stack.md
adr/ADR-0003-ai-boundary.md
adr/ADR-0004-storage-and-data-lifecycle.md
adr/ADR-0005-observability-and-audit.md
adr/ADR-0006-mcp-adoption.md
```

Derive `<topic-slug>` deterministically (see
`references/input-discovery.md` "Topic-Slug Derivation"): strip
`-product-blueprint.md` or `-blueprint.md` from the source filename; otherwise
slugify the product name in the blueprint title; if still ambiguous, use
`architecture-design.md` and record the naming assumption. If files cannot be
written, emit the full Markdown inline and state the recommended filename.

The architecture document must contain these 25 sections, in order, and begin
with a `## Contents` section and an `## Update History` near the top
(see `templates/architecture_design_template.md`):

1. Executive Architecture Summary
2. Source Blueprint Interpretation
3. Clarification Summary
4. Architecture Goals and Constraints
5. Solution Strategy
6. Traditional Software vs AI-Agent Boundary
7. Recommended Tech Stack
8. System Context View
9. Container / Runtime View
10. Component View
11. AI / Skill / MCP Architecture
12. Interface Contracts
13. Data Contracts and Schemas
14. State, Storage, and Data Lifecycle
15. Workflow / Sequence Views
16. Observability, Logging, Telemetry, and Audit
17. Security and Trust Boundaries
18. Failure Handling and Recovery
19. Testing and Evaluation Architecture
20. Deployment Architecture
21. Architecture Decision Records
22. Technical Risks and Trade-offs
23. Open Questions
24. Architecture Quality-Gate Self-Check
25. Handoff Notes for Implementation Planning

Formatting requirements:

- `## Contents` with internal links to every numbered section.
- `## Update History` table (Date · Source Blueprint · Architecture Version ·
  Change Type · Affected Sections · Notes). New documents add an initial row;
  updates append a row; never delete prior rows.
- Generation metadata block (§1.x) — copy, never invent; use `unknown` when
  unavailable.
- **Mermaid** C4 diagrams for the System Context view, the Container/Runtime
  view, and at least one Dynamic (sequence) view of the main workflow.
  Component and Deployment views are conditional (see
  `references/c4_model_summary.md`).
- A Traditional-vs-AI responsibility matrix (table).
- A tech-stack decision table (Decision · Recommendation · Alternatives ·
  Rationale · Risks · Reversible?).
- ADRs for every high-impact decision (`templates/adr_template.md`).

## Responsibilities

The skill must: accept or discover a blueprint; parse blueprint sections;
build a blueprint-to-architecture traceability map; ask or infer
architecture-relevant decisions; choose a tech stack with rationale; generate
C4 views; define traditional-vs-AI responsibility boundaries; decide skill/MCP
boundaries; define interface and data contracts; define security and trust
boundaries; define observability, logs, metrics, traces, and audit events;
define failure handling and recovery; define the testing/evaluation
architecture; generate ADRs; run rule-pack quality gates; and produce handoff
notes for implementation planning.

## Anti-Patterns (do NOT)

- Turn every conceptual blueprint component into a separate service.
- Select a technology without rationale, alternatives, and reversibility.
- Introduce MCP because it is fashionable (see `references/mcp_adoption_guide.md`).
- Place durable state changes under direct AI control.
- Allow AI components to bypass deterministic validation.
- Ignore the blueprint's MVP-0 / MVP-1 staging.
- Generate implementation tasks too early, or write code.
- Hide unresolved architecture questions.
- Use domain-biased default stacks (e.g. always Python/FastAPI/PostgreSQL);
  example stacks in `references/` and `examples/` are examples, not defaults.

## Method

Follow the staged method. The 24 prompt files under `prompts/` carry the
detailed instructions for each pass; `manifest.json` records the task graph,
dependencies, validation gates, and failure policy. The manifest is a
task-graph specification: it is valid as documentation for a prompt-driven
agent and ready for a future runtime runner (see §"Manifest" in
`tests/manifest_coverage_checklist.md`). If no runner exists, follow the
manifest order manually.

1. **Resolve input blueprint** — explicit argument → context → working-dir
   search → candidate scoring (`prompts/01`, `references/input-discovery.md`).
2. **Detect existing architecture** — none / existing same-topic document /
   existing ADRs; select an update mode (`prompts/02`). This output is
   consumed by the context, solution-strategy, ADR, draft, and final-document
   passes.
3. **Prepare blueprint context** — pass small blueprints through unchanged;
   create an architecture-focused extract for large ones, preserving MVP,
   risks, interfaces, security, observability, and handoff notes (`prompts/03`).
4. **Parse blueprint** — thesis, MVP-0/MVP-1, actors, capabilities, workflows,
   logical architecture, conceptual objects, decision policies, risks,
   evaluation, handoff notes, decision register (`prompts/04`).
5. **Build the blueprint-to-architecture traceability map** (`prompts/05`,
   `templates/blueprint_to_architecture_map_template.md`).
6. **Run clarification** — interactive / automatic / hybrid; ask only
   architecture-impacting questions, one at a time, each with a recommended
   answer; record assumptions (`prompts/06`).
7. **Generate the solution strategy** (`prompts/07`).
8. **Define architecture goals and constraints** — goals, functional and
   non-functional requirements, security/privacy, data/retention, cost/latency,
   team, MVP-0/MVP-1, and explicit assumptions (`prompts/08`).
9. **Select a provisional tech stack** with rationale and alternatives,
   marked provisional (`prompts/09`, `templates/tech_stack_decision_table.md`).
10. **Generate the Traditional-vs-AI responsibility matrix** (`prompts/10`,
    `templates/ai_responsibility_matrix_template.md`).
11. **Decide skill / MCP / server boundaries** — MCP only if justified
    (`prompts/11`, `references/mcp_adoption_guide.md`).
12. **Run tech-stack / AI-boundary coherence review** and revise the stack so
    it is consistent with the AI boundary and MCP strategy (`prompts/12`).
13. **Generate C4 views** — system context, container/runtime, dynamic; plus
    component (most complex container only) and deployment (only when topology
    affects security, privacy, scaling, availability, or operations)
    (`prompts/13`, `references/c4_model_summary.md`).
14. **Define interface contracts** — owner, input/output/error contracts,
    validation, versioning, observability fields (`prompts/14`,
    `templates/interface_contract_template.md`).
15. **Define data contracts and storage lifecycle** — storage owner,
    retention, schema evolution, audit immutability (`prompts/15`).
16. **Define security and trust boundaries** — trust zones, identity/access,
    AI/LLM boundary, prompt-injection controls, secrets, security events and
    gates (`prompts/16`, `templates/security_trust_boundary_template.md`,
    `references/security_trust_model_guide.md`).
17. **Define observability, logging, telemetry, and audit** — correlation IDs,
    logs, metrics, traces, audit trail (`prompts/17`,
    `templates/observability_plan_template.md`,
    `references/observability_event_catalogue.md`).
18. **Define failure handling and recovery** (`prompts/18`).
19. **Define the testing and evaluation architecture** (`prompts/19`).
20. **Generate or update ADRs** — supersede, never silently overwrite
    (`prompts/20`, `templates/adr_template.md`, `references/adr_guidance.md`).
21. **Apply rule-pack quality gates** as review checks (`prompts/21`,
    `rule-packs/`).
22. **Generate the architecture draft**, respecting the selected update mode
    (`prompts/22`, `templates/architecture_design_template.md`).
23. **Run the quality-gate self-check** (`prompts/23`).
24. **Produce the final architecture document** with Contents, metadata,
    Update History, self-check, and handoff notes; apply update-mode rules if
    an architecture document already exists (`prompts/24`).

## Operating Modes

- **Interactive** — high ambiguity, or a decision affecting cost, privacy,
  security, legal exposure, UX, deployment, or data retention. Ask one
  question at a time with a recommended answer; never ask what the blueprint
  or codebase already answers.
- **Automatic** — autonomous processing of conventional, low-risk, reversible
  decisions. Make the decision, record the assumption, state reversibility,
  define a revisit trigger.
- **Hybrid (default)** — proceed automatically for low-risk decisions; ask
  only high-impact unresolved questions (3–7); record all assumptions; flag
  critical decisions for review.

## Quality Gates

The output fails (revise; max 3 attempts on gate passes, then surface failing
gates and stop) if any hold — full text in `prompts/23_quality_gate_self_check.md`
and `tests/expected_sections_checklist.md`:

1. **Skill contract** — SKILL.md trigger/use contract and the 24-task manifest
   cover the major workflow passes.
2. **Traceability** — a blueprint-to-architecture map exists and every major
   architecture decision traces to a blueprint section, a user clarification,
   a rule-pack decision, or an explicit recorded assumption.
3. **Document completeness** — Contents, generation metadata, and Update
   History are present; all 25 sections exist.
4. **Tech stack** — every major choice has rationale, alternatives, and a
   reversibility verdict; no domain-biased default applied without
   justification.
5. **AI boundary** — a Traditional-vs-AI responsibility matrix exists; AI
   components cannot mutate durable state without deterministic validation;
   MCP is introduced only with clear clients, resources, permissions, audit,
   error model, versioning, and a considered non-MCP alternative.
6. **C4 coverage** — System Context, Container/Runtime, and a main-workflow
   Dynamic view are present; container/module ownership is explicit; the
   architecture does not convert every conceptual component into a separate
   service without rationale.
7. **Contracts, data, security, observability, failure** — interface
   contracts, data contracts with storage owner/retention/schema-evolution, a
   security/trust-boundary model, an observability/audit plan, and failure
   handling for every critical workflow are all present.
8. **MVP & ADRs** — MVP-0/MVP-1 from the blueprint is respected; ADRs exist
   for every high-impact decision with supersession handled; handoff notes for
   implementation planning are present.
9. **Update behavior** — when an architecture document already exists, the
   update mode (regenerate / patch / compare / adr-only / resume) is explicit
   and Update History is appended.
10. **Metadata consistency** — §1 metadata counts match the document
    (`Clarification count` = §3 rows; `Assumptions made` = §4.9 rows) and every
    `A-N`, ADR, Contents, and section reference resolves; no metadata is
    invented.
11. **Hybrid decision review** — in hybrid mode every major §3 decision carries
    a Source and a Review Requirement, and high-impact inferred/assumed
    decisions are review-flagged (hybrid must not silently act as automatic).
12. **Technology-specific validity** — the architecture never credits a chosen
    technology with enforcement/security properties it does not provide;
    absolute wording is downgraded to application-enforced / tamper-evident /
    best-effort with a risk or ADR note.
13. **Probe/evaluator availability** — every model-backed evaluator or gating
    probe has an availability policy (required level, behavior if unavailable,
    whether auto-accept is still allowed, audit event), or the gate is n/a.
14. **Architecture-vs-implementation boundary** — module boundaries and
    *proposed* namespaces are allowed; task tickets, code, migrations, and
    file-by-file steps are not.

## References

| File | Load when |
|------|-----------|
| `references/input-discovery.md` | Discovering/scoring a blueprint; deriving the topic slug |
| `references/c4_model_summary.md` | Choosing and drawing C4 views; deciding component/deployment triggers |
| `references/adr_guidance.md` | Writing/superseding ADRs |
| `references/security_trust_model_guide.md` | Trust zones, AI boundary rules, prompt-injection controls |
| `references/observability_event_catalogue.md` | Correlation IDs, log/metric/trace/audit catalogue |
| `references/mcp_adoption_guide.md` | Deciding skill vs MCP; the MCP adoption gate |
| `templates/architecture_design_template.md` | The 25-section output skeleton |
| `templates/*` | Per-section skeletons (ADR, interfaces, observability, security, AI matrix, tech-stack, traceability, metadata, update history, contents) |
| `rule-packs/*` | Boundary, data, interface, reliability, AI-boundary, observability, security review gates |
| `examples/*` | A worked blueprint excerpt and the architecture it produces |

## Final Reminder

This skill bridges product blueprint and implementation planning. Its core
rule:

> Normal software owns deterministic control, state, storage, security, audit,
> interfaces, telemetry, and workflow transitions. AI / LLM / agents own
> language-heavy, judgment-heavy, and reasoning-heavy tasks. MCP is justified
> only when reusable external tool/data access is needed.

It answers *"how should this product be technically structured?"* — it does
**not** produce implementation tasks or code; that is the next skill's job.
