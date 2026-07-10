# Changelog

All notable changes to research-pipeline.

## [v0.32.0] — 2026-07-10

Continued the bundled `blueprint` design-chain skill review: closed a second
5-issue batch (#92–#96) hardening the coherence guard, the self-check, and the
template/prompt discipline, each fix shipped with tests. The skill manifest moved
0.9.0 → 0.10.0.

### blueprint skill

- Recompute the quality-gate self-check against the final body: a deterministic
  `vendor_leak` scan (named products, vendor CLIs, wire-level config flags) now
  re-derives the implementation-neutrality verdict on every revision, so a leak
  reintroduced by a later edit cannot survive behind a stale `PASS` (#92).
- Strengthen the coherence guard — anchor coverage is enforced (MVP-staging prose
  with no anchors, or an Open Questions section with no `stage=open` anchor, now
  FAILs), a `consumes=` signal edge catches signal inversions, and
  `orphan_reference` catches a consumed signal whose producer is never registered
  (#93).
- Add an altitude ceiling for agent/tool-surface sections: WARNING-tier
  `mechanism_altitude` (dedup keys, compaction, single-PEP, retry-bounding,
  transport outside the §18 handoff) and `tool_identifier_altitude` (a named tool
  in a policy/eval/MVP row that should bind to a READ/ACT/AUTH class) (#94).
- Enforce actor-channel completeness: §3 tags every actor's channel class and §8
  defines every escalation/authorization path per channel, with a new Gate 10
  that FAILs an escalation assuming an inline human on a headless channel (#95).
- Template discipline: one authoritative home per decision (Appendix B, keyed by
  a stable ID) and a conservative MVP-0 default that keeps an unvalidated,
  non-research-derived capability out of the smallest core behind a cheap gating
  test (#96).

## [v0.31.0] — 2026-07-06

Hardened the bundled `blueprint` design-chain skill: closed a 5-issue review
(#81–#85) of its quality gate, test oracle, and internal duplication, each fix
shipped with tests. The skill manifest moved 0.8.0 → 0.9.0.

### blueprint skill

- Add a deterministic cross-phase coherence guard
  (`scripts/check_blueprint_coherence.py`) wired between `compose-blueprint` and
  `quality-gate`: it materializes a phase-inversion def→use graph from stable
  template anchors and fails loudly on an MVP-N node whose required servicer is
  staged later, or an MVP control gated on a non-blocking open question (#81).
- Strengthen the quality gate — citation fidelity for load-bearing claims, an
  agent-mode READ/ACT authorization-boundary requirement with a matching risk
  row, and an amend-without-a-new-report update playbook (#85).
- Make the golden fixture a trustworthy oracle: regenerate it phase-coherent,
  add labelled servicer-reachability / precondition-currency regression pairs, a
  mutation-derived negative set, and a weak-input output fixture (#83).
- Derive the required-section list from the template's own `## N.` headings with
  an exhaustive check, ending the stale 18-vs-20 section-count drift (#82).
- Single-source the skill's duplicated structure — assert the frontmatter
  triggers against `When To Trigger`, reconcile the manifest orthogonality seams
  (one owner for intake / input-quality / extraction / MVP membership), and have
  the quality gate reference the generation prompt instead of restating it (#84).

## [v0.30.0] — 2026-07-06

Closed two review umbrellas: the MCP-server conformance/security review
(#14, 13 issues) and the end-to-end pipeline session findings (#34, 18 issues).
31 fixes/features, each shipped with tests; the unit suite grew to 4593 passing.

### Security

- Sandbox custom Jinja2 report templates with `ImmutableSandboxedEnvironment`,
  closing an SSTI → RCE via `tool_report(custom_template=…)` (#35).
- Confine resource paths: a `_safe_join` containment helper + traversal
  validators on `run_id`/`paper_id`/`date`, plus a `run_id` schema validator (#40).
- Wire the zero-trust `McpGuard` (registration + schema-hash integrity +
  rate-limit + audit) into MCP tool dispatch (#45).
- Scrub absolute filesystem paths from tool error messages (#44).

### MCP server

- Tools emit `isError` on failure and return `ToolResult`, so FastMCP now
  populates `outputSchema` and `structuredContent` (#38).
- Fix 5 broken prompts (invalid `role:"system"`) (#36); make elicitation gates
  real and fail-closed on cancel (#37); declare the `logging` capability and
  honour `setLevel` (#41).
- Resource reads raise on missing artifacts instead of returning
  success-shaped error blobs (#42); size-cap resource reads + search progress (#44).
- Flip `readOnlyHint` on 7 writer tools (#39); unify the `workspace` default (#43).
- Capability-domain toolsets via `RESEARCH_PIPELINE_MCP_TOOLSETS` to cut the
  64-tool schema tax (#46); add an in-memory protocol test harness (#47).

### Pipeline & CLI

- New `verify` command so the skill's manifest validation gates work (#18).
- Field-scope plain-language arXiv query variants (#16); per-source search
  summary + zero-yield warning + `--strict-sources` (#20).
- `quality`/`summarize`/`export-bibtex` parse shortlists leniently, with `llm`
  coercion (#25, #28); `CandidateRecord` defaults/coerces null category fields (#24).
- `expand` merges instead of overwriting (`--replace` for the old behaviour) (#27);
  convert-rough/-fine rebuild the unified manifest (#30); download surfaces the
  `max_per_run` cap truncation (#29).
- Config loader raises on an explicit missing `--config` (#21); statistic range
  sanity checks with a `stat_warnings.json` sidecar (#33).

### Skill & setup

- `setup --check` diagnoses dangling-symlink/stub installs; warn on fragile
  `--symlink` (#19).
- Skill runner validates MCP-kind tasks and captures the plan `run_id` (#17);
  runner `--status` exit code locked (#22); `check_completion` finds
  `validation_result.json` (#32); `report` accepts `--config` (#31); sub-agent
  model guidance de-pinned + missing-agent fallback (#23); fix null-externalIds
  warning spam in the citation graph (#26).

## [v0.29.1] — 2026-06-30

### Fixed (CI drift)

CI had drifted red since master was last green (2026-06-07) — tooling and advisories moved while the
code did not. Restored all gates to green:

- **Lint:** excluded `docs/` (doc-build utility scripts) from ruff — a newer ruff flagged 34 latent
  E501/style issues there.
- **Security:** bumped CVE-flagged deps (python-multipart 0.0.32, starlette 1.3.1, vcrpy 8.2.1,
  cryptography 49.0.0, msgpack 1.2.1, pydantic-settings 2.14.2); ignored torch CVE-2025-3000 in
  pip-audit (no fix available, transitive) in CI + Makefile.
- **Type check:** added a mypy `follow_imports = "skip"` override for `fitz` / `pymupdf` /
  `pymupdf4llm` (they now ship partial/no types) and removed 7 stale `# type: ignore` comments that
  a newer mypy flagged as unused.

## [v0.29.0] — 2026-06-29

### Changed

- **Extracted the `architecture` and `ux-design` skills to the separate `design-pipeline` repo.**
  research-pipeline now ships the `research-pipeline`, `blueprint`, and `daily-ai-intelligence`
  skills; the downstream product-design stages (blueprint → architecture → ux-design) live in
  their own repo. The chain is unchanged: skills install side-by-side into `~/.claude/skills/`
  and hand off via artifact files; the artifact-contract in `blueprint/references/` is the interface.

### Removed

- `skill_data/architecture/`, `skill_data/ux-design/` and their `tests/unit/test_skill_*.py`.
- Architecture-skill design/improvement notes under `docs/`.
- `tests/unit/test_artifact_contract.py` trimmed to the blueprint (producer) side; downstream
  template/prompt conformance now lives in design-pipeline.

## [v0.28.0] — 2026-06-07

### Added

- **`architecture` skill — Cross-Section Consistency Gate (skill manifest 0.9.0 → 1.0.0).**
  Prevents downstream stages from discovering architecture gaps too late by adding
  a Cross-Section Consistency Pass inside `architecture --mode design`.
  - **`prompts/22_architecture_draft.md`** — adds step 2c: Cross-Section
    Consistency Pass verifying (i) every MVP operation in §23 exists in §12,
    (ii) every user-visible state in §23.3/§23.4 maps onto §14, (iii) every
    human-review action in §23.6 has a §12 contract + §14 transition + §16 audit
    event + §18 failure behaviour, (iv) every progress item in §23.4 has a §16
    observability event, (v) §24 handoffs reference only formalized or deferred
    operations. Gaps logged in §25 and §27.
  - **`prompts/23_quality_gate_self_check.md`** — adds five v0.9.0 cross-section
    consistency gates (20–24); updates Instructions to include them in the §26
    table.
  - **`templates/architecture_design_template.md`** — adds
    `### Cross-Section Consistency Gate` sub-table to §26.
  - **`SKILL.md`** — adds quality gate 29.
  - **`manifest.json`** — bumped to 1.0.0.

- **`ux-design` skill — MVP phase and testability metadata (skill manifest 0.2.0 → 0.3.0).**
  Enables `implementation-plan` to convert E2E seeds into concrete test tasks
  without guessing and to scope MVP-0 correctly.
  - **`prompts/07_user_stories.md`** — requires phase/release-gate header on
    every story (Phase, Primary Surface, Release Gate, Depends On); updates
    validation gate.
  - **`prompts/10_e2e_scenario_seeds.md`** — requires testability metadata block
    on every seed and a Testability Summary Table in §20; updates validation gate.
  - **`templates/e2e-scenario-template.md`** — adds testability metadata fields
    to skeleton and worked example.
  - **`templates/user-story-template.md`** — adds phase + release-gate fields;
    updates Rules.
  - **`prompts/13_quality_gate_self_check.md`** — adds six gate rows, five fail
    conditions, six warning conditions.
  - **`SKILL.md`** — adds quality gate 11.
  - **`manifest.json`** — bumped to 0.3.0.

## [v0.27.0] — 2026-06-07

### Added

- **`architecture` skill — `materialize` mode (skill manifest 0.8.0 → 0.9.0).**
  Adds `architecture --mode materialize` to consolidate the base architecture
  design plus all accepted update notes into one canonical implementation-ready
  source of truth before `implementation-plan`.
  - **`prompts/materialize_02_discover_and_order.md`** — discovers the base
    architecture by topic slug, discovers and validates accepted update notes
    (acceptance criteria: `Artifact Type = architecture_update`, quality-gate
    PASS, no unresolved blocking conflicts), sorts by version/update history,
    builds a section-level patch plan, flags preliminary conflicts.
  - **`prompts/materialize_03_apply_patches.md`** — checks six BLOCKING conflict
    classes (SECTION_CONFLICT, PATCH_TARGET_MISSING, TEXT_ANCHOR_GONE,
    CONTRACT_CONFLICT, SUPERSESSION_CONFLICT, BREAKING_CHANGE_UNRESOLVED); on
    any conflict writes a materialization-blocked report and stops; otherwise
    applies patches in order, removes resolved provisional wording, updates ADRs
    / open questions / handoffs.
  - **`prompts/materialize_04_final_document.md`** — produces the 30-section
    canonical architecture (27 original + Applied Updates + Superseded Patch
    Notes + Implementation-Plan Readiness), the artifact registry, and the
    open-question ledger; runs the Materialization Quality-Gate Self-Check.
  - **`templates/architecture_canonical_template.md`** — 30-section canonical
    architecture skeleton with Applied Updates, Superseded Patch Notes, and
    Implementation-Plan Readiness sections; `Artifact Type = architecture_canonical`.
  - **`templates/artifact_registry_template.md`** — artifact registry skeleton
    (Current Canonical Artifacts + Superseded / Historical Artifacts + Pipeline
    Stage Summary) telling downstream agents which files to read and which are
    audit history.
  - **`templates/open_question_ledger_template.md`** — open-question ledger
    skeleton centralizing all open questions with source, owner stage, blocking
    status, and required resolution action.
  - **`templates/materialization_blocked_template.md`** — blocked report skeleton
    emitted when a BLOCKING conflict prevents safe materialization.
  - **`references/materialization-guide.md`** — full guide: when to use, what it
    owns / must not own, accepted-update discovery rules, conflict classes, output
    files, quality-gate fail/warn conditions.
  - **`references/patch-manifest-guide.md`** — Patch Manifest YAML format guide
    and update-type taxonomy (NONE / NOTE_ONLY / ADR_ONLY / CONTRACT_PATCH /
    SECURITY_PATCH / OBSERVABILITY_PATCH / STRUCTURAL_PATCH / BREAKING_CHANGE).
  - **`references/artifact-registry-guide.md`** — artifact registry purpose,
    controlled status values, artifact role vocabulary, and rules.
  - **`references/open-question-ledger-guide.md`** — ledger purpose, ID prefix
    convention, status and owner-stage values, blocking-status values.
  - Mode resolver (`prompts/00_mode_resolver.md`) updated: adds `materialize`
    to the mode table and routing JSON; adds explicit materialize triggers and
    negative triggers; updates preconditions for six modes.
  - Mode-selection guide (`references/mode-selection-guide.md`) updated: adds
    `materialize` mode description, selection logic, negative triggers, and
    downstream flow showing full pipeline through materialize to
    `implementation-plan`.
- **`architecture --mode update` metadata improvements (skill manifest 0.8.0 → 0.9.0).**
  Update mode output grows from 11 to 13 sections:
  - **§5 Patch Manifest** — machine-readable YAML consumed by `materialize`;
    every accepted decision maps to a patch entry with target section, operation,
    patch type, and implementation-blocking flag.
  - **§12 Feedback Closure Matrix** — tracks which downstream feedback items
    (ux-design Architecture Feedback, security-review findings) are resolved by
    which patches; `NOT_APPLICABLE` when no downstream feedback was applied.
  - **Patch-Type Taxonomy** added to Generation Metadata §1; multi-type allowed.
  - **Skill version metadata** — `UNKNOWN — resolver could not determine this
    value` replaces bare `unknown`; quality gate warns on any UNKNOWN field.
  - `prompts/update_02_apply_decisions.md` and `update_03_final_document.md`
    updated to instruct producing §5, §12, and classified patch types.
  - `references/architecture-update-guide.md` updated with taxonomy table,
    Feedback Closure Matrix guidance, and 13-section quality gate.
  - `SKILL.md` updated: six-mode table, references table, `materialize` mode
    description, method section with `materialize_tasks` task graph.
  - `manifest.json` version 0.8.0 → 0.9.0; `materialize_tasks` task graph and
    `materialize_mandatory_gates` added; mode resolver routing updated.

## [v0.26.0] — 2026-06-07

### Added

- **Cross-Skill Artifact Contract** across the `blueprint`, `architecture`, and
  `ux-design` skills (blueprint skill 0.7.0 → 0.8.0, architecture 0.7.0 → 0.8.0,
  ux-design 0.1.0 → 0.2.0). A shared standard so every generated document is both
  a human report **and** a machine-readable handoff artifact, preventing pipeline
  drift / artifact mismatch / unstable topic slugs as the skill graph grows.
  - **`references/artifact-contract.md`** — the canonical contract (artifact-type
    registry, stable-topic-slug rules, filename rules, required Generation
    Metadata, Source/Resolved artifacts, decision register, assumptions, open
    questions, recommended next stage, quality-gate, controlled vocabulary,
    per-skill requirements). Skills install independently (each is symlinked to
    its own dir), so the file ships **identically inside each of the three
    skills**; a guard test asserts the copies stay byte-identical.
  - **Templates aligned (7):** the blueprint, the five architecture mode
    templates (design / tech-stack / update / reconciliation / review), and the
    ux-design template now carry `Artifact Type` + stable `Topic Slug` metadata,
    an unnumbered **`## Cross-Skill Artifact Contract`** block (Source Artifacts
    Consumed, Resolved Input Artifacts where discovery is used, and a Contract
    Field Map that points at each skill's existing decision/assumption/
    open-question/next-stage sections — alignment, not duplication), and a
    **Cross-Skill Artifact Contract Gate** in every self-check. No existing
    section numbering changed.
  - **Prompts:** the seven main generation / final-document prompts now instruct
    contract compliance with the controlled vocabulary; each SKILL.md References
    table lists `artifact-contract.md`.
  - **Guard tests:** `tests/unit/test_artifact_contract.py` enforces the contract
    structurally (templates include Generation Metadata / Topic Slug / Source
    Artifacts Consumed / Recommended Next Stage / Quality-Gate Self-Check; the
    five architecture mode templates include Resolved Input Artifacts; every
    self-check includes the contract gate; every skill references the contract;
    the contract file carries the registry + controlled vocabulary; the three
    copies are identical).
  - Worked examples are dated pre-contract snapshots and adopt the contract on
    next regeneration; `implementation-plan` / `security-review` / `test-design`
    will comply from day one.

## [v0.25.0] — 2026-06-07

### Added

- **`architecture` skill — `review` / `update` / `reconcile` modes (skill
  manifest 0.6.0 → 0.7.0).** The three remaining modes are now fully implemented
  (previously recognized-but-deferred), completing the architecture lifecycle:
  `design → stack → update → ux-design → reconcile → review → implementation-plan`.
  - **Shared artifact resolver** (`prompts/resolve_artifacts.md`,
    `references/artifact-discovery-guide.md`). When the user passes no filenames,
    all three modes auto-discover the most relevant prior skill/mode outputs from
    the working dir / `docs` / `design` / `artifacts` by topic slug + candidate
    scoring (filename suffix, section markers, recency), require the architecture
    design (STOP with a clear message if missing), ASK_USER on ambiguity, and
    embed a **Resolved Input Artifacts** table in every output.
  - **`review`** — evaluates architecture quality **without changing it**: a
    10-point score breakdown with justifications, blocking/warning/polish issue
    classification, readiness assessments, and recommended next actions →
    `<topic-slug>-architecture-review.md` (19 sections). It is the safe default
    when bare `architecture` is invoked and an architecture already exists.
  - **`update`** — applies **already-accepted** decisions (priority source: a
    tech-stack with *Architecture Update Required? = Yes*, then accepted
    reconciliation, security-review, newer blueprint, explicit user decision;
    **never** ux-design directly) into `<topic-slug>-architecture-update.md` (11
    sections) — an update note that **does not overwrite** the design document by
    default.
  - **`reconcile`** — turns downstream feedback (primarily a ux-design
    *Architecture Feedback* section) into findings, a missing-architecture-support
    map, a minimal patch plan, and an explicit **Architecture Update Required?**
    verdict + handoff to `update` → `<topic-slug>-architecture-reconciliation.md`
    (11 sections). It **does not patch** the architecture by default and does not
    blindly accept a downstream artifact that contradicts the blueprint.
  - **No-silent-mutation rule** across all three modes; three new task graphs
    (`review_tasks` / `update_tasks` / `reconcile_tasks`) + mandatory gates; the
    mode resolver and mode-selection guide updated (bare `architecture` +
    existing architecture → `review`, never `update`); three templates, four
    references, three worked translation-system examples that chain off the
    existing stack (`Architecture Update Required? = Yes`) and ux-design
    (Architecture Feedback) examples; and Phase-5 guard tests.

### Changed

- AGENTS.md: the `architecture` skill is now described as **five implemented
  modes**.

## [v0.24.0] — 2026-06-07

### Added

- **`ux-design` skill (new; skill manifest 0.1.0).** The fifth bundled skill and
  the fourth stage of the design chain: `research-pipeline → blueprint →
  architecture → ux-design → implementation-plan`. It consumes an architecture
  design document (`architecture --mode design` output; the matching blueprint
  and tech-stack are optional) and emits `<topic-slug>-ux-design.md`. Pure
  prompt-driven (no CLI/MCP backend); auto-discovered by `setup`.
  - **User-story-driven, not screen-drawing.** It separates **Skill Operator UX**
    (how the user drives the skill workflow) from **Target Software UX** (how end
    users and agents interact with the product), and produces structured user
    stories (preconditions, main / alternative / failure-recovery flows,
    user-visible states resolving to the architecture state model, acceptance
    criteria), core journeys, surface-specific UX (only for
    architecture-supported surfaces), human-in-the-loop UX, error / empty /
    loading / degraded / recovery states, acceptance criteria, and Gherkin-style
    **E2E scenario seeds** (seeds, not executable tests).
  - **Mandatory architecture feedback.** §21 records UX-exposed architecture gaps
    (missing states / events / review schema / permissions) with severity and
    recommends `architecture --mode reconcile` when needed — the UX→architecture
    feedback loop.
  - **Input discovery + fail-fast.** Auto-discovers `*-architecture-design.md` in
    the working dir / `docs` / `design` / `artifacts`, ranks candidates, and
    STOPs with a clear message if none is found (it does not run from a blueprint
    alone). Hybrid is the default operating mode.
  - 22-section + Appendix-A output template, 13-task manifest, 13 prompts, 3
    templates, 4 references, a worked translation-system example, and structural
    tests (`tests/unit/test_skill_ux_design.py`).

### Changed

- **`architecture` skill references** updated to reflect that `ux-design` now
  exists (the mode-selection and Experience-Architecture guides no longer call it
  a "future skill not yet built"; no behavior change, architecture manifest stays
  0.6.0).
- AGENTS.md: "Four skills" → "Five skills" with a `ux-design` row and description.

## [v0.23.0] — 2026-06-06

### Changed

- **`architecture` skill — `design`/`stack` mode split (skill manifest 0.5.0 →
  0.6.0).** Phase 1 of the architecture-skill mode-split improvement plan
  (`docs/architecture-skill-mode-split-improvement-plan.md`). The skill becomes
  **one skill with internal modes** instead of mixing architecture design and
  tech-stack selection in one pass. Tech/domain-neutral; the existing
  prompts/template/example are extended, not rewritten.
  - **Mode resolver.** A new `prompts/00_mode_resolver.md` + a `## Modes` section
    in SKILL.md + `references/mode-selection-guide.md` select the mode (`design`
    / `stack` / `update` / `review` / `reconcile`) from an explicit `mode`
    argument or automatic detection. `design` and `stack` are fully specified;
    `update` reuses design mode's existing regenerate/patch/compare/adr-only/
    resume machinery; `review`/`reconcile` are recognized but deferred (never
    silently mutate an architecture).
  - **`design` mode consumes blueprint §9 and §19.** It now reads the blueprint's
    §9 Product Experience Direction into a new **§23 Experience Architecture**
    section (interaction surfaces, user-visible states mapped onto the §14 state
    model, feedback/progress, error/recovery, human-review flow, trust/
    transparency, UX handoff — architecture-level UX support, not UX design), and
    reflects the §19 Recommended Next Stages routing in a new **§24 Recommended
    Next Stages and Downstream Handoffs** section (tech-stack / ux-design /
    security-review / test-design handoffs + update/reconciliation triggers). The
    design output grows from **25 to 27 sections**.
  - **Provisional-tech discipline.** §7 keeps the tech stack **provisional** with
    new §7.1 Provisional Tech Assumptions and §7.2 Tech-Stack Selection Handoff
    sub-sections whenever §19 routes `tech-stack-selection = RUN`/`DEFER`; final
    selection is owned by `stack` mode. Five new §26 self-check gates enforce
    this (Product Experience Direction consumed, Experience Architecture
    produced, Recommended Next Stages consumed, tech-stack-provisional, and
    downstream-handoffs-present).
  - **`stack` mode.** A separate 6-task graph (`prompts/stack_01`–`stack_06`,
    `templates/architecture_tech_stack_template.md`,
    `references/tech-stack-selection-guide.md`) selects the concrete technology
    stack against the architecture — one decision per area with alternatives,
    rationale, risk, reversibility, and architecture impact — and ends with an
    explicit **Architecture Update Required?** verdict. It satisfies the
    architecture and never redesigns it; on a genuine conflict it declares the
    update instead of rewriting. Output: `<topic-slug>-architecture-tech-stack.md`.
  - A worked `stack` example (`examples/translation_tech_stack_example.md`) is
    added and the `design` example is updated to 27 sections.

## [v0.22.0] — 2026-06-06

### Changed

- **`blueprint` skill — adaptive next-stage routing clarity (skill manifest
  0.6.0 → 0.7.0).** A focused follow-up to the §19 router and §9 Product
  Experience Direction, acting on an external review of a generated blueprint.
  No redesign — the changes are columns, a couple of small tables, and a short
  rationale, all tech/domain-neutral and folded into the existing
  prompts/templates/reference/example:
  - **§19.2 `Depends On` column.** The Stage Recommendations table now states
    each stage's prerequisite explicitly (e.g. security-review and ux-design
    depend on architecture-design), distinct from the revisit trigger, so the
    pipeline no longer looks more rigid — or more parallel — than intended.
  - **§19.4 Recommended Pipeline split.** The single flat pipeline list is now a
    **Recommended Linear Path** (core ordered sequence) plus a **Conditional
    Follow-up Gates** table (`Gate | Run When | Typical Input | Output`), so
    deferred conditional stages — chiefly `architecture-update` /
    `architecture-reconciliation` — are never silently dropped from the routing
    picture.
  - **§19.3 ASK_USER Decision Rationale.** When no stage is `ASK_USER`, §19 now
    justifies why (each high-impact unknown is already answered by the source
    report / §9, deferred to architecture-design, or delegated to
    security-review, with a named owner) instead of looking over-confident;
    `FAIL` if `ASK_USER` is absent despite an unresolved high-impact unknown with
    no downstream owner.
  - **§19.1 complexity score labelled a routing heuristic.** The output now
    states that the `/ 21` score is a routing heuristic, not a formal project
    estimate, to be revisited after architecture-design — preventing false
    precision.
  - **§9.4 / §9.5 interaction-mode `Classification`.** Every interaction mode now
    carries a controlled classification — *primary surface* / *secondary
    surface* / *wrapper / integration surface* / *future surface* — with an
    explicit rule that "AI Skill" must be disambiguated (usually a
    wrapper/integration surface around the CLI/core, not a separate runtime) and
    never conflated with MCP (a tool surface for external AI agents).
  - **Quality gates.** The Product Experience Gate gains an "Interaction modes
    classified" row; the Adaptive Stage-Gate Recommendation Gate gains
    "Stage table has Depends On", "Linear-path vs conditional-gates split",
    "ASK_USER absence explained", and "Complexity score labelled heuristic" rows,
    plus matching WARNING conditions. The §19 output-discipline budget was
    updated so the additions (columns + small tables) do not contradict the
    "keep §19 compact" rule.

## [v0.21.0] — 2026-06-06

### Added

- **`blueprint` skill — Recommended Next Stages (adaptive stage-gate routing;
  skill manifest 0.5.0 → 0.6.0).** The blueprint is now the **first adaptive
  stage-gate router** after research extraction: instead of leaving every
  optional downstream stage to manual choice, it evaluates the project and
  recommends which stages should run next — with evidence, confidence, and
  revisit triggers — without silently expanding the pipeline. Folded into the
  existing prompts/templates/example (no new manifest tasks),
  tech/domain-neutral:
  - **New §19 Recommended Next Stages** (the output template grows from 19 to 20
    sections; §19 sits after the technical-design handoff and before the
    Traceability Appendix, which renumbers to §20). It contains a **Pipeline
    Complexity Assessment** (seven 0–3 dimensions — user-facing complexity,
    technical ambiguity, security/privacy risk, AI/LLM uncertainty, integration
    complexity, human-review complexity, testing/E2E importance — with a total
    `/ 21` and a simple/lightweight/medium/complex workflow class), a **Stage
    Recommendations** table covering architecture-design, tech-stack-selection,
    ux-design, security-review, test-design, architecture-update, and
    architecture-reconciliation, a short **Recommended Pipeline**, and a
    **Stage-Gate Decision Log**.
  - **Controlled decision vocabulary** — every stage uses exactly one of
    **RUN / SKIP / DEFER / ASK_USER** (no vague "maybe / consider / nice to
    have"). Every RUN/ASK_USER needs evidence, every SKIP a reason, every DEFER
    a revisit trigger; recommendations are overrideable defaults.
    `architecture-design` is normally RUN; `architecture-update` and
    `architecture-reconciliation` default to DEFER at blueprint stage (no
    architecture document exists yet).
  - **Adaptive Stage-Gate Recommendation Gate** added to the quality-gate prompt
    (Gate 9) and the Appendix A self-check (seven rows: section exists,
    controlled decisions, RUN evidence, SKIP reason, DEFER revisit trigger,
    ASK_USER missing-info, and Product Experience Direction informs the
    recommendations), with explicit FAIL and WARNING conditions.
  - **Product Experience Direction integration** — §9 signals drive the routing
    (human review → ux-design / test-design; external egress → security-review;
    CLI-first → ux-design DEFER; future MCP → tech-stack-selection RUN/DEFER),
    making §9 actionable rather than decorative.
  - **New reference** `references/adaptive-stage-gate-routing.md` — the decision
    vocabulary, per-stage RUN/SKIP/DEFER/ASK_USER rules, complexity scoring, the
    PED-integration table, the output-discipline budget, and the gate
    definition. SKILL.md, prompts 04/05, the template, the example, and both
    checklists were updated; the shipped example models a realistic 16/21
    "complex" routing recommendation.

### Changed

- `blueprint` SKILL.md now documents 20 sections, adds adaptive-routing trigger
  phrases ("what should we build next?", "which design stages should we run
  next?"), and a ninth quality gate.

## [v0.20.0] — 2026-06-06

### Added

- **`blueprint` skill — Product Experience Direction (lightweight UX-intent
  support; skill manifest 0.4.0 → 0.5.0).** The blueprint now captures enough
  product-experience direction to keep downstream architecture and
  implementation from inventing UX assumptions, without turning blueprint into a
  full UX-design stage. The boundary is explicit: **blueprint defines UX intent
  · architecture defines the UX-enabling technical structure · a later UX-design
  skill defines the detailed experience · implementation-plan turns it into
  build tasks.** Folded into the existing prompts/templates/example (no new
  manifest tasks), tech/domain-neutral:
  - **New §9 Product Experience Direction** (the output template grows from 18
    to 19 sections; §9 sits after Workflow Model and before Logical
    Architecture, so UX intent exists before the architecture handoff). It
    captures, compactly (1–2 pages, tables): Primary Experience Thesis · Primary
    User / Operator · Primary Job-to-Be-Done · Primary Interaction Mode
    (CLI/Web/GUI/TUI/API/AI-Skill/MCP/Hybrid, with MVP stage + rationale) ·
    Secondary / Future Interaction Modes · Critical Trust, Control, and
    Transparency Requirements · Human-in-the-Loop Experience · Failure and
    Recovery Expectations · UX Assumptions for Architecture · Product Experience
    Handoff to Architecture.
  - **Product Experience Gate** added to the quality-gate prompt and the
    Appendix A self-check (eight rows: primary user, job-to-be-done, experience
    thesis, interaction mode, trust/control/transparency, human-in-the-loop,
    failure/recovery, UX handoff), with explicit FAIL and WARNING conditions.
  - **UX-intent boundary enforced** — §9 must not contain screen layout,
    wireframes, CSS/visual design, full user journeys, exact CLI syntax/flags,
    exact MCP/API schemas, copywriting, mobile navigation, full accessibility
    checklists, or implementation tasks (UX over-reach is a gate FAIL). SKILL.md
    gains UX trigger phrases and negative triggers (detailed UI/UX → a later
    UX-design skill); the forbidden-content checklist gains a detailed-UX
    section.
  - **New reference** `references/product-experience-direction.md` — the section
    template, the boundary rule, the grill-me-style clarification-question
    format (good vs. bad questions), the Product Experience Gate, and the output
    budget.
  - **Chain handoff** — the `architecture` skill's blueprint-parse prompt now
    extracts `product_experience` (interaction mode, trust/control/transparency,
    human-in-the-loop, failure/recovery, UX assumptions) so it preserves UX
    intent instead of inventing it.
  - Updates the manifest, SKILL.md, prompt 04 (generate), prompt 05
    (quality-gate), the output template, both test checklists, and the worked
    example (now modelling §9 + a PASS-with-warning PE-gate row), with new guard
    tests including a Copilot description-length check.

## [v0.19.5] — 2026-06-05

### Changed

- **`architecture` skill — provenance & output-discipline hardening pass (skill
  manifest 0.4.0 → 0.5.0).** Fourth review-driven pass. The reviewed document
  was confirmed generated by v0.19.4 (metadata reports skill 0.4.0) and the
  v0.19.4 fixes were verified to have landed (data-egress + verification-table
  §17.12 present, zero unchecked checkboxes). Five targeted additions, folded
  into the existing prompts + self-check (no new manifest tasks), tech-neutral:
  - **Decision evidence / provenance** — high-impact §3 decisions carry a
    Decision Evidence value (confirmed_in_interactive_answer /
    …_from_supplied_configuration / …_from_previous_architecture_document /
    …_from_blueprint / architecture_assumption / unknown_requires_review) and
    are never labelled `user-confirmed` without an explicit source; unclear
    provenance downgrades to `architecture_assumption` + review.
  - **Raw source content forbidden in logs by default** — for external-model
    systems the logging answer is "No" unless explicitly opted into; provider
    SDKs that may log prompts require redaction + a log-snapshot test + a
    provider-wrapper redaction test, as a release-blocking §17.12 gate (operator
    configuration alone is insufficient). (The reviewed doc had "Logs may
    contain source content: Yes".)
  - **Warning surfacing** — every §24 WARNING / PASS-with-warning row is echoed
    into §1 (Executive Summary, new §1.2) and §25 (Handoff) with its required
    action and blocking status, so an implementation-plan agent reading the
    handoff cannot miss them.
  - **Stricter standard-output budget** — the budget pass now carries concrete
    targets (≤ 1-page summary, ≤ ~15 tech decisions, core contracts/entities
    only, ADR summary table in body, heavy material to appendices).
  - **Build-sequencing cap** — §25 may include at most five high-level
    sequencing constraints; file-by-file order, task tickets, PR sequences, and
    class/migration ordering are rejected (implementation-plan skill's job).
  Most of the review's lower-priority items (PASS-with-warning semantics,
  data-egress classification, state-semantics, verification-table gates) were
  already shipped in v0.19.3/v0.19.4 and were **not** re-implemented. The worked
  example now models a realistic PASS-with-warning (data-egress assumption) and
  surfaces it. Updates prompts 06/16/22/23/24, both templates, the example,
  SKILL.md, and the security checklist, with new guard tests.

## [v0.19.4] — 2026-06-05

### Changed

- **`architecture` skill — quality-control hardening pass (skill manifest 0.3.0
  → 0.4.0).** Third review-driven pass. The reviewed document was confirmed to
  be generated by v0.19.3 (its metadata reports skill 0.3.0), so the review was
  evaluated against the latest skill, not a stale output. Most of the review
  duplicated checks already shipped in v0.19.3 (PASS-with-warning semantics,
  standard-vs-detailed budget, state-semantics, data-egress decision) — those
  were **not** re-implemented. The two genuinely new, verified items were:
  - **Security quality gates as a verification table (§17.12).** Security gates
    must be a table — Security Gate · Required Implementation Evidence ·
    Verification Method · Blocks Release? — never ambiguous unchecked `- [ ]`
    checkboxes (which are also where a residual technology-inconsistent claim
    had hidden). Added a `Security gate verification format` self-check gate.
  - **Data Egress / External Model Use table (§17.9).** The v0.19.3 single §3
    data-egress decision is expanded into a dedicated table
    (content-leaves-boundary, which providers, redaction, logs-may-contain-source,
    domain-plugin-override) whenever external models are used;
    `unknown_requires_user_review` blocks implementation planning.
  - **Residual invalid-claim scan strengthened** to explicitly cover §17
    security-gate rows and §24 checklist rows (not just §7 prose) — the verified
    "no UPDATE/DELETE granted to the application DB user" line lived inside a
    checkbox security gate, so converting those gates to a verification table
    structurally removes that class of leftover claim.
  Updates prompts 16/23, the security and architecture templates, the worked
  example (now models a §17.9 egress table and a §17.12 verification table with
  honest append-only wording, and contains no unchecked checkboxes), SKILL.md,
  and the security checklist; adds guard tests. Kept tech/domain-neutral.

## [v0.19.3] — 2026-06-05

### Changed

- **`architecture` skill — quality-control hardening pass (skill manifest 0.2.0
  → 0.3.0).** Driven by a second review of a real generated architecture
  document; each claim was verified against the actual output before
  implementing, and the checks were folded into the existing
  `quality_gate_self_check` + pass prompts (no new manifest tasks):
  - **Data egress / external-model-use decision** — when the architecture sends
    content to an external model, §3 must carry a *distinct* data-egress
    decision (external_allowed / external_allowed_with_redaction / local_only /
    hybrid_by_domain / unknown_requires_user_review), separate from the
    provider-abstraction choice and review-flagged when assumed. (The reviewed
    output used external models with no such decision.)
  - **Residual invalid-claim scan** — the validity check now scans the whole
    document (security gates, data, ADRs, checklist rows), not just the
    tech-stack section, for technology-inconsistent wording. (A residual
    "no UPDATE/DELETE granted to the application DB user" line had survived the
    earlier per-section fix.)
  - **Self-check skepticism** — the §24 status model is standardized to
    PASS / WARNING / FAIL (WARNING ≡ "PASS with warning", non-blocking but with
    a required action); a section with a known contradiction or residual invalid
    claim must never be a clean PASS.
  - **State-semantics consistency** — §14 is now a canonical state model that
    keeps lifecycle states, operational condition flags, and audit events
    distinct; every state/status/condition term used elsewhere must resolve to
    it.
  - **Output detail budget** — `standard` output stays a concise main body +
    appendices (heavy schemas / full ADR bodies / long matrices move to
    appendices) rather than a full dossier; no required section or major
    decision is dropped.
  Kept tech/domain-neutral (SQLite/LiteLLM/translation specifics are
  illustrations, not hard-coded rules). Updates prompts 06/15/16/22/23, the
  template (§3 data-egress note, §14 canonical state model, §24 gate rows +
  status legend), the worked example, SKILL.md, and the security/expected
  checklists, with new guard tests in `tests/unit/test_skill_architecture.py`.

## [v0.19.2] — 2026-06-05

### Changed

- **`architecture` skill — quality-control pass (skill manifest 0.1.0 →
  0.2.0).** Driven by a review of a real generated architecture document. Five
  final-gate checks were added to `quality_gate_self_check` (and the relevant
  pass prompts), folded into the existing 24-task manifest rather than adding
  new tasks — no workflow redesign:
  - **Metadata consistency gate** — `Clarification count` must equal the §3
    row count, `Assumptions made` must equal the §4.9 row count, and every
    `A-N`, ADR, Contents, and section reference must resolve. (The reviewed
    output had count/table and `A-N` mismatches.)
  - **Hybrid-mode decision-review classification** — §3 now carries **Source**
    and **Review Requirement** columns; high-impact inferred/assumed decisions
    (external LLM use, data privacy, deployment, storage, auth, retention, cost
    routing, MCP, human approval) must be review-flagged so hybrid mode does not
    silently behave like automatic mode.
  - **Technology-specific validity** — the architecture may not credit a chosen
    technology with enforcement/security properties it does not provide;
    absolute wording is downgraded to application-enforced / tamper-evident /
    best-effort with a risk or ADR note. (Kept tech-neutral — illustrated, not
    hard-coded.)
  - **Probe/evaluator availability policy** — every model-backed evaluator or
    gating probe needs a required-level + behavior-if-unavailable +
    auto-accept-allowed + audit-event policy; required probes disable
    auto-accept when unavailable. (Generalized; n/a when there are no
    model-backed evaluators.)
  - **Architecture-vs-implementation boundary** — module names are labelled
    "proposed module namespaces," and task tickets / code / migrations /
    file-by-file steps are rejected (they belong to the implementation-plan
    skill).
  Implemented inside the existing prompts/template/example plus new guard tests
  in `tests/unit/test_skill_architecture.py`; the skill remains a pure
  prompt-driven transformation.

## [v0.19.1] — 2026-06-05

### Fixed

- **`architecture` skill — shortened the `SKILL.md` description to fit GitHub
  Copilot's 1024-character limit.** The v0.19.0 description was 1376 characters,
  which prevented the GitHub Copilot CLI from loading the skill (Claude Code and
  Codex CLI were unaffected). Rewrote it to 925 characters while preserving the
  purpose statement, the pipeline-chain position, the key trigger phrases, the
  aliases, and the do-not-use routing — the dropped detail already lives in the
  SKILL.md body. Added `test_skill_md_description_fits_copilot_limit` to
  `tests/unit/test_skill_architecture.py` so the description cannot creep back
  over the limit.

## [v0.19.0] — 2026-06-05

### Added

- **New `architecture` skill (skill manifest 0.1.0).** Converts a product
  blueprint into a concrete technical architecture and tech-stack design,
  continuing the chain `research-pipeline → blueprint → architecture →
  implementation-plan`. Like `blueprint`, it is a pure prompt-driven
  transformation (no CLI/MCP backend) and is auto-discovered and installed by
  `research-pipeline setup` alongside the other bundled skills.
  - **Input discovery & resume** — discovers/parses a `*-product-blueprint.md`
    (with candidate scoring), derives the topic slug deterministically, detects
    an existing `<topic-slug>-architecture-design.md`, and supports
    regenerate / patch / compare / adr-only / resume update modes with an
    `## Update History`.
  - **Tech-stack / AI-boundary co-design** — provisional tech stack →
    Traditional-vs-AI responsibility matrix → skill/MCP decision → coherence
    review, so the stack stays consistent with the AI boundary and MCP
    strategy. MCP is adopted only when it passes an explicit adoption gate.
  - **25-section output** — Contents, generation metadata, Update History, a
    blueprint-to-architecture traceability map, C4 views (system context,
    container/runtime, dynamic; optional component/deployment), interface and
    data contracts, security/trust boundaries, observability/audit, failure
    handling, a testing/evaluation architecture, ADRs, a quality-gate
    self-check, and implementation-planning handoff notes.
  - **Determinism guarantee** — durable control, state, storage, audit, and
    workflow transitions stay in traditional software; AI output is validated
    before any state change. The skill selects a tech stack but never writes
    code or implementation tasks.
  - Ships 24 prompt files, 11 templates, 6 references, 7 rule packs, a worked
    translation example, and 10 quality-gate checklists under
    `src/research_pipeline/skill_data/architecture/`, with
    `tests/unit/test_skill_architecture.py` validating the contract.

## [v0.18.5] — 2026-06-04

### Changed

- **`blueprint` skill — finalization-quality pass (skill manifest 0.3.0 →
  0.4.0).** Driven by a review of a third real generated blueprint (which
  scored 9.4/10). Refinements, not a redesign:
  - **Contents/appendix consistency** — the Contents now lists every
    appendix actually present (the Appendix A self-check was sometimes
    omitted); added to the template, prompts, gate, and example.
  - **Self-repair pass** — the skill now applies safe wording rewrites the
    self-check identifies *before* delivery (detect → repair → re-check),
    so only judgement-needing warnings remain and the self-check reflects
    the post-repair document.
  - **Stricter `standard` length control** — large tables (full §5
    translation map, §18 traceability) move to appendices; the main body
    stays scannable in one pass.
  - **Optional Appendix B — Design Decision Register** — an opt-in,
    handoff-only register (reversibility + revisit trigger) that does not
    duplicate §6.
  - **MVP-0/MVP-1 split now mandatory** when a product has more than 4
    major capabilities.
  - Thesis-emphasis, gap-citation-fallback, scope-control, and
    metadata-integrity gates from 0.2.0/0.3.0 are preserved.
  Implemented entirely inside the existing prompts/template/references — no
  new prompt files; the skill remains a pure prompt-driven transformation.

## [v0.18.4] — 2026-06-04

### Changed

- **`blueprint` skill — precision & buildability pass (skill manifest
  0.2.0 → 0.3.0).** Driven by a review of a second real generated blueprint.
  The quality gate now also enforces: **thesis emphasis** (lead with the
  primary research-backed architecture, not a conditional/secondary
  mechanism); **MVP-0 / MVP-1 split** (§14 splits the core path into a
  smallest-demonstrable MVP-0 and a first-usable MVP-1, with safety and
  evaluation baselines kept separate); **release-gate confidence
  consistency** (a MEDIUM/LOW-confidence mechanism becomes a release gate
  only with HIGH risk impact + no cheaper control + an explicit why-now);
  and a **gap-citation fallback** (gap-derived rows cite
  `[Source Report: Research Gaps — <name>]` instead of leaving the citation
  blank). The Appendix A self-check is now **actionable** (gate · status ·
  finding · required action · blocks-technical-design). Implemented entirely
  inside the existing prompts/template/references — no new prompt files; the
  skill remains a pure prompt-driven transformation.

## [v0.18.3] — 2026-06-04

### Changed

- **`blueprint` skill — post-generation quality-control hardening (skill
  manifest version 0.1.0 → 0.2.0).** Driven by a review of a real generated
  blueprint. The quality gate now also enforces: **metadata integrity**
  (skill version copied from `manifest.json` or `unknown`; **pipeline runs
  integrated** and **gap-closure rounds** kept as separate fields, never
  conflated); **scope control** (primary actors/domains must match the
  thesis; high-stakes domains seen only as evidence stay Secondary/Future);
  **source fidelity** (claims classified research-backed / extrapolation /
  design-decision / speculative / unsupported, with relocation or removal);
  an **implementation-neutrality warning tier** (runtime-leaning wording is
  flagged, not silently accepted, while only named tech/vendor choices hard
  fail); and **`standard` length budgets**. §14 MVP now separates a minimal
  Core Value Path from Safety and Evaluation baselines, and every blueprint
  ends with a compact `Appendix A: Blueprint Quality-Gate Self-Check`
  (PASS/WARNING/FAIL). The skill remains a pure prompt-driven transformation
  (no new prompt files or executing runner).

### Fixed

- **`setup` test** updated to assert the always-overwrite default-mode
  upgrade semantics introduced in v0.18.2 (`test_setup.py`), restoring a
  green test job.

## [v0.18.2] — 2026-06-03

### Fixed

- **`research-pipeline setup` now overwrites stale skill copies in default
  multi-target mode.** Previously, running `setup` after `pipx upgrade` would
  silently skip any already-installed skill directory (e.g.
  `~/.agents/skills/daily-ai-intelligence/`, `~/.claude/skills/…`) with only a
  WARNING, leaving stale copies behind. Root cause: `_install_directory` was
  called with `skip_existing=True` in default mode. Fix: pass
  `force or default_multi_target` so default mode always overwrites (upgrade
  semantics) while single-target mode still requires explicit `--force`.
  Same fix applied to `_install_agent_files` for agent files.

## [v0.18.1] — 2026-06-03

### Fixed

- **Skill description truncated to fit the 1024-character Copilot CLI limit.**
  Both `daily-ai-intelligence` and `blueprint` SKILL.md `description:` fields
  exceeded 1024 characters, causing the skills to fail to load in GitHub
  Copilot CLI. Descriptions trimmed to ≤926 chars while preserving all
  trigger phrases and `Do NOT use` guards.

## [v0.18.0] — 2026-06-02

### Added

- **New bundled skill: `blueprint` (Research-to-Product Blueprint).**
  Converts a `research-pipeline` synthesis report into an
  implementation-neutral product blueprint (`<topic-slug>-product-blueprint.md`).
  It classifies input quality (strong/usable/weak/insufficient), maps
  `ACADEMIC` gaps to validation requirements and `ENGINEERING` gaps to
  product requirements, resolves each idea as ADOPT/ADAPT/MERGE/DEFER/REJECT,
  and emits an 18-section blueprint (thesis, users, workflows with Mermaid,
  logical architecture, conceptual information model, decision policies,
  risk model, evaluation strategy, MVP boundary, roadmap, open questions,
  technical-design handoff, traceability appendix). The skill is a pure
  prompt-driven transformation with no CLI/MCP backend, and never selects a
  tech stack.
- The skill ships under `src/research_pipeline/skill_data/blueprint/`
  (SKILL.md, manifest.json, five prompts, five templates, four references,
  and example inputs/outputs) and is auto-discovered and installed by
  `research-pipeline setup` alongside `research-pipeline` and
  `daily-ai-intelligence`.

## [v0.17.34] — 2026-05-17

### Fixed

- **Bug A (LOW — research-pipeline/references/output-templates.md): Contents link list had
  `[Executive Summary]` before `[Round History]`, contradicting the template body order**
  The `## Contents` template (the ToC model at the top of the report template) listed
  `[Executive Summary](#executive-summary)` on the first link line and
  `[Round History](#round-history)` on the second. However, the actual template body
  defines `## Round History` first (line 113) and `## Executive Summary` second (line 126).
  The `final-report-contract.yaml` required_sections list correctly orders Round History
  before Executive Summary. An agent following the Contents template verbatim would generate
  a table of contents whose first two links are in the wrong order relative to the document
  body, causing a broken reading experience. Fixed by swapping the two lines so Round History
  precedes Executive Summary in the Contents link list.

- **Bug B (LOW — research-pipeline/manifest.json): `validate-report` task label still read
  "14-section check" after the required section count was reduced to 12 in v0.17.32**
  `manifest.json` task `validate-report` had `"label": "Validate final report completeness
  (14-section check)"`. The required section count was reduced from 14 to 12 in v0.17.32
  (final-report-contract.yaml) and v0.17.31 (final_report.schema.json), but the manifest
  label was never updated. This created a confusing discrepancy: every other document (schema,
  contract, templates) said 12, while the task label still said 14. Fixed: "14-section" →
  "12-section" in the label.

- **Bug C (LOW — daily-ai-intelligence/runners/subagent_contracts/rank_reviewer.yaml):
  `forbidden_actions` third bullet said "only write verdict.json" but the actual output
  file is `rank_review.json`**
  Line 34 of `rank_reviewer.yaml` read `Do NOT mark the task accepted yourself — only write
  verdict.json`. The actual output artifact is `rank_review.json` (declared in the same
  file's `outputs` block on line 22 and `completion_criteria` on line 68). "verdict.json"
  does not exist anywhere in the DAI workflow. A sub-agent reading the forbidden_actions
  list would be directed to write a nonexistent file name. Fixed: "verdict.json" →
  "rank_review.json".

- **Bug D (CHANGELOG — CHANGELOG.md): `## [v0.17.32]` and `## [v0.17.31]` version headers
  were missing; their bug entries were present but unattributed to a version**
  Lines 61–135 (v0.17.32 bugs 1–6) and lines 138–219 (v0.17.31 bugs 1–7 + changed) had
  their `### Fixed` section headers but lacked the preceding `## [v0.17.x] — date` version
  entry, making the changelog structurally invalid. Readers could not tell which version
  introduced each fix. Added `## [v0.17.32] — 2026-05-15` before the Bug 1 that starts
  the v0.17.32 block, and `## [v0.17.31] — 2026-05-15` before the Bug 1 that starts the
  v0.17.31 block.

## [v0.17.33] — 2026-05-16

### Fixed

- **Bug 1 (BREAKING — daily-ai-intelligence/manifest.json): `generate-daily` did not depend
  on `review-ranked`, so the reviewer gate had no effect on brief generation**
  `generate-daily.depends_on` was `["rank"]`. When `review-ranked` was delegated (LLM reviewer
  working), the runner paused after delegation (returned 0). On the next runner invocation,
  `review-ranked` status was `"delegated"` — which the loop skips with `continue` — and
  `generate-daily` found its only listed dependency `rank` already `accepted`, so it was
  immediately auto-accepted and the brief was generated regardless of the reviewer's verdict.
  The reviewer gate introduced by v0.17.30 Bug 1 was therefore completely ineffective. Fixed
  by adding `"review-ranked"` to `generate-daily.depends_on`. Because `skipped_by_policy ∈
  READY_STATUSES`, runs without `--reviewer` are unaffected (`review-ranked` is
  `skipped_by_policy` and `generate-daily` proceeds normally). With `--reviewer`, the brief
  now correctly waits for the reviewer to accept before proceeding.

- **Bug 2 (LOW — research-pipeline/references/final-report-contract.yaml): `purpose` still
  said "14-section output template" after v0.17.32 Bug 5 reduced the required sections to 12**
  v0.17.32 Bug 5 updated `required_sections` and `completion_criteria` to use 12 sections, but
  the `purpose` free-text field on line 14 still read "the 14-section output template". An
  agent reading the purpose as a header description would see an incorrect section count.
  Fixed: "14-section" → "12-section".

- **Bug 3 (LOW — research-pipeline/references/final-report-contract.yaml): `evidence_requirements`
  second bullet referenced "Key Findings" instead of "Confidence-Graded Findings"**
  v0.17.32 Bug 6 explicitly stated it "updated the `evidence_requirements` reference from
  'Key Findings' to 'Confidence-Graded Findings'", but the change was not actually applied;
  line 96 still read "The Evidence Table must list every paper used in Key Findings." The
  section "Key Findings" does not exist in the current report template — the correct section
  is "Confidence-Graded Findings" (see `output-templates.md` and `final_report.schema.json`).
  Fixed: "Key Findings" → "Confidence-Graded Findings" in the evidence requirement.

- **Bug 4 (LOW — research-pipeline/references/final-report-contract.yaml): `round_state.json`
  fallback description still referenced the stale `run_ids_all` field removed in v0.17.32**
  v0.17.32 Bug 1 removed `run_ids_all` from `round_state_template.json` because the runner
  never writes it, and noted that `final-report-contract.yaml` had referenced it. However, the
  contract's `round_state.json` description still contained "treat as round: 1, run_ids_all:
  [<current_run_id>]" — instructing a report writer to expect a field that doesn't exist at
  runtime. Fixed: replaced `run_ids_all: [<current_run_id>]` with `open_gaps: []` to match
  the actual fields written by `_write_round_state()` (see `round_state_template.json`).

- **Bug 5 (MEDIUM — research-pipeline/references/output-templates.md): Contents section
  template was missing `Confidence-Graded Findings` and `Evidence Map` entries**
  Both `confidence_graded_findings` and `evidence_map` are required sections in
  `final_report.schema.json` and appear in the full report template body of `output-templates.md`,
  but the `## Contents` list template (the table of contents model at the top) was missing both
  entries. An agent following the Contents template would produce a table of contents that omits
  two required sections, causing the report validator to flag a mismatch. Fixed by adding
  `[Confidence-Graded Findings](#confidence-graded-findings)` and
  `[Evidence Map](#evidence-map)` to the Contents template list, in the correct position
  between `Research Landscape` and `Research Gaps`, and between `Practical Recommendations`
  and `References` respectively.


## [v0.17.32] — 2026-05-15

### Fixed

- **Bug 1 (BREAKING — references/round_state_template.json): Template required fields
  did not match what runner.py `_write_round_state()` actually writes**
  The template declared `required: ["round", "run_id", "topic_slug", "convergence_reason",
  "gaps_addressed", "gaps_remaining", "run_ids_all", "updated_at"]`, but the runner never
  writes `convergence_reason`, `gaps_addressed`, `gaps_remaining`, or `run_ids_all`.
  Conversely, fields the runner always writes — `workflow_id`, `topic`, `status`, `profile`,
  `open_gaps` — were entirely absent from the template. An agent or report writer consulting
  the template would build wrong expectations about `round_state.json` content (e.g.
  `final-report-contract.yaml` referenced `run_ids_all` from the template). Fixed by
  rewriting the template `required` list and all property definitions to match the actual
  runtime output of `_write_round_state()`. The old convergence/accumulation fields
  (`convergence_reason`, `gaps_addressed`, `gaps_remaining`, `run_ids_all`) were removed
  because the runner never writes them; the template description was updated to clarify
  that this is a running-state snapshot for hooks (stop-check.sh, resume-inject.sh)
  and report writers.

- **Bug 2 (BREAKING — references/gap-classification-contract.yaml): Contract contradicted
  the normative `gap_classification.schema.json` in multiple fields**
  The `references/gap-classification-contract.yaml` document (read by agents as guidance)
  used different field names and required-fields than the normative JSON schema that the
  runner validates `gaps.json` against. Specific conflicts:
  (a) Top-level `required` included `report_path` and `date` (not in normative schema) and
  omitted `round` (required by normative schema).
  (b) Gap description field was `text` (schema requires `description`).
  (c) Convergence field was `stop_reason` (schema requires `reason`).
  (d) `stop_reason` enum used stale values (`converged_no_gaps`, `cap_reached`,
  `no_new_papers`, `user_out_of_scope`, `continue`) instead of the normative values
  (`open_gaps_remain`, `no_open_gaps`, `round_cap_reached`, `no_new_papers`,
  `user_marked_out_of_scope`).
  (e) Convergence `required` listed `open_academic` and `open_engineering` (not required by
  schema). An agent following this contract would produce a `gaps.json` that fails
  schema validation and whose gaps the runner could not filter correctly. Fixed by aligning
  all field names and required lists with `gap_classification.schema.json`; supplementary
  fields (`rationale`, `section`, `open_academic`, `open_engineering`) are now documented
  as optional extras consistent with `additionalProperties: true` in the schema. Updated
  `completion_criteria` to require the canonical `reason` enum values.

- **Bug 3 (LOW — runners/subagent_contracts/synthesis_reviewer.yaml): completion_criteria
  omitted `accepted_with_issues` as a valid status**
  The `completion_criteria` block stated `"status is 'accepted' or 'rejected'"`, omitting
  the third valid status `accepted_with_issues` defined in `reviewer_result.schema.json`.
  A synthesis reviewer agent reading this criterion might incorrectly reject or re-write
  a verdict with status `accepted_with_issues`. Fixed: criterion now reads
  `"status is 'accepted', 'rejected', or 'accepted_with_issues'"`.

- **Bug 4 (LOW — runners/subagent_contracts/rank_reviewer.yaml, daily-ai-intelligence):
  Same `accepted_with_issues` omission in DAI rank reviewer**
  The `rank_reviewer.yaml` completion_criteria had the identical defect as Bug 3.
  Fixed with the same correction.

- **Bug 5 (BREAKING — references/final-report-contract.yaml): required_sections listed
  14 stale section headings that no longer match `output-templates.md` or `final_report.schema.json`**
  The contract listed sections from an earlier report structure: `## Background`,
  `## Key Findings`, `## Evidence Table`, `## Comparative Analysis`, `## Assumption Map`,
  `## Risk Register`, `## Recommendations`, `## Conclusion`. These were all replaced or
  removed in the `output-templates.md` and `final_report.schema.json` overhaul
  (v0.17.31). The current required sections — `## Research Question`, `## Papers Reviewed`,
  `## Research Landscape`, `## Confidence-Graded Findings`, `## Practical Recommendations`,
  `## Evidence Map` — were entirely absent from the contract's list. An agent using this
  contract as the authoritative section checklist would produce a report that fails
  `research-pipeline validate`. Fixed: replaced `required_sections` with the 12 sections
  declared in `final_report.schema.json`; updated completion_criteria from "All 14 required
  sections" to "All 12 required sections".

- **Bug 6 (MEDIUM — references/final-report-contract.yaml): Input path referenced
  `screen/shortlist.json` (non-existent file) instead of `screen/screened.jsonl`**
  The `required_inputs` block listed `"runs/<run_id>/screen/shortlist.json"`. The screen
  stage writes `screened.jsonl` (JSONL, not JSON, and named `screened` not `shortlist`).
  This same error was corrected in `command-reference.md` in v0.17.31 but not in
  `final-report-contract.yaml`. Fixed path to `"runs/<run_id>/screen/screened.jsonl"`.
  Also updated the `evidence_requirements` reference from "Key Findings" to
  "Confidence-Graded Findings" to match the current section name.


## [v0.17.31] — 2026-05-15

### Fixed

- **Bug 1 (MEDIUM — research-pipeline/runners/subagent_contracts/paper_screener.yaml):
  Header comment `# Task: screen-candidates` did not match the manifest task ID `paper-screener`**
  The first two comment lines read `# Task: screen-candidates (paper_screener)`. The actual
  `task_id` field (line 6) was already `paper-screener` (correct), but the comment contradicted
  it. A sub-agent reading the contract for self-identification would see a task name
  (`screen-candidates`) that does not exist anywhere in `workflow_state.json`, causing confusion
  when interpreting delegation messages. Fixed by correcting the header comment to
  `# Task: paper-screener (paper_screener)`.

- **Bug 2 (MEDIUM — research-pipeline/runners/subagent_contracts/paper_analyzer.yaml):
  Header comment `# Task: paper-analysis` did not match the manifest task ID `paper-analyzer`**
  Same class of bug as Bug 1 above. The comment said `# Task: paper-analysis (paper_analyzer)`,
  while the `task_id` field was already `paper-analyzer` (correct). Fixed by correcting the
  header comment to `# Task: paper-analyzer (paper_analyzer)`.

- **Bug 3 (LOW — research-pipeline/runners/subagent_contracts/paper_synthesizer.yaml):
  Header comment `# Task: synthesis` did not match the manifest task ID `paper-synthesizer`**
  The comment said `# Task: synthesis (paper_synthesizer)`; the actual `task_id` field was
  already `paper-synthesizer` (correct). Fixed by updating the header comment to
  `# Task: paper-synthesizer (paper_synthesizer)`.

- **Bug 4 (MEDIUM — research-pipeline/runners/runner.py + hooks/resume-inject.sh):
  `_write_round_state()` wrote `current_round` but `round_state_template.json` and the
  fixed `iterative-synthesis.md` (v0.17.26) both use the field name `round`**
  `_write_round_state()` in `runners/runner.py` constructed the `round_state.json` dictionary
  with key `current_round`, but `round_state_template.json` (the authoritative schema) declares
  the field as `round`, and `iterative-synthesis.md` was corrected to use `round` in v0.17.26.
  In a multi-round session the LLM (following the fixed guide) writes `round_state.json` with the
  `round` field; the runner had already written `current_round` in the same file for the prior
  round. If the LLM replaced the whole file (not just patched it), `resume-inject.sh`'s lookup of
  `current_round` would find nothing and display `"?"` for the round number, breaking the
  context-injection hook. This was a latent regression introduced when v0.17.26 fixed
  `iterative-synthesis.md` but left the runner and hook unchanged.
  Fixed by:
  (a) Changing the key in `_write_round_state()` from `"current_round"` to `"round"` so the
  runner-written `round_state.json` matches the schema and the LLM-written version;
  (b) Changing `resume-inject.sh` to read `d.get("round", "?")` instead of
  `d.get("current_round", "?")` and updating the display label from `current_round:` to
  `round:`.

- **Bug 5 (MEDIUM — research-pipeline/runners/subagent_contracts/paper_analyzer.yaml):
  Input/output paths used `{cwd}` instead of `{run_dir}`, pointing outside the pipeline run directory**
  The `paper_analyzer.yaml` contract specified input paths as `{cwd}/convert/markdown/` and
  `{cwd}/screened.jsonl`, and output as `{cwd}/analysis/`. The runner context provides
  `run_dir` = `<cwd>/runs/<run_id>`, so all pipeline stage artifacts live under `{run_dir}`,
  not `{cwd}`. Using `{cwd}` would direct the sub-agent to look one level too high (the skill
  working directory rather than the specific pipeline run). Fixed by updating all paths to
  use `{run_dir}`: `{run_dir}/convert/markdown/`, `{run_dir}/screen/screened.jsonl`,
  `{run_dir}/analysis/`, and the completion criterion reference.

- **Bug 6 (LOW — research-pipeline/references/command-reference.md):
  Profile membership table and screen stage output filename were stale**
  The `standard` profile was described as adding `paper-screener, expand, enrich,
  analyze-claims, score-claims` to `quick`; the actual `standard` set in `manifest.json` is
  `expand, convert-fine, analyze-claims, score-claims, classify-gaps`. The `deep` profile
  description likewise listed tasks that belong to `standard`. Additionally the Screen stage
  output was listed as `screen/shortlist.json` instead of the correct `screen/screened.jsonl`.
  Fixed profile membership descriptions to match `manifest.json` and corrected the output
  filename.

- **Bug 7 (MEDIUM — research-pipeline/schemas/final_report.schema.json):
  Required section keys were stale and did not match `references/output-templates.md` headings**
  The schema required old section names (`background`, `key_findings`, `evidence_table`,
  `comparative_analysis`, `assumption_map`, `risk_register`, `conclusion`) that no longer exist
  in the current report template. The validator would accept reports with those legacy headings
  while rejecting reports written against the current `output-templates.md`. Fixed by aligning
  the required section keys with `output-templates.md`: `research_question`, `papers_reviewed`,
  `research_landscape`, `confidence_graded_findings`, `practical_recommendations`, `evidence_map`
  are now required; optional sections (`methodology_comparison`, `trade_off_analysis`,
  `points_of_agreement`, `points_of_contradiction`, `reproducibility_notes`, `future_directions`,
  `readiness_assessment`, `appendix_run_metadata`) are now listed as known boolean properties.

### Changed

- **research-pipeline/config.toml**: Added `analysis_model = "claude-opus-4.6"` under
  `[summarization]` so the model used by all LLM worker sub-agents (paper-screener,
  paper-analyzer, paper-synthesizer, gap-classifier, synthesis-reviewer) is documented and
  configurable in one place. Also expanded the default `[sources].enabled` list to include
  `semantic_scholar`, `openalex`, `dblp`, and `huggingface` alongside `arxiv`.

## [v0.17.30] — 2026-05-14

### Fixed

- **Bug 1 (BREAKING — daily-ai-intelligence/manifest.json + runners/runner.py): `review-ranked`
  had `optional: true` but no `trigger_condition`, causing it to always delegate as an
  `llm_reviewer` and block daily brief generation**
  The DAI runner skips optional tasks only when they have a non-empty `trigger_condition` AND
  the condition is not met (`if trigger and not _optional_trigger_met(...)`). `review-ranked`
  had `optional: true` but lacked `trigger_condition` entirely, so the guard was never entered.
  As an `llm_reviewer` task, the runner always delegated it and returned 0 immediately after
  `rank` completed — before `generate-daily` ever ran. The `failure_policy.note` correctly
  described it as "non-blocking for daily runs" but the code made it blocking in practice.
  Fixed by:
  (a) Adding `trigger_condition` to `review-ranked` in `manifest.json`;
  (b) Adding `"review-ranked": ["reviewer_requested"]` to `_optional_trigger_met` in
  `runners/runner.py`;
  (c) Adding a `--reviewer` CLI flag that sets `reviewer_requested` in the workflow context;
  (d) Adding `"reviewer_requested": ""` to `workflow_state_template.json` context;
  (e) Updating `references/workflow-steps.md` `[review-ranked]` section to document that the
  task only runs when `--reviewer` is passed, and that it is skipped automatically otherwise;
  (f) Updating `SKILL.md` Launch section to list `review-ranked` among optional tasks and
  document the `--reviewer` flag.

- **Bug 2 (MEDIUM — research-pipeline/references/sub-agents.md): Mermaid flowchart node C
  incorrectly listed `download, convert, extract, summarize` as running before
  `paper-screener` delegation**
  Node C read `"Deterministic stages run automatically (plan, search, screen, download,
  convert, extract, summarize)"`, implying all these stages run before the `paper-screener`
  LLM delegation. But per `manifest.json`, `download` depends on `paper-screener`, so
  `download`, `convert-rough`, `convert-fine`, `extract`, and `summarize` all run *after*
  `paper-screener` is accepted. An agent following the incorrect diagram might misunderstand
  the pipeline order and attempt to run download/convert before delegating the screener.
  Fixed by:
  (a) Changing node C to list only the pre-screener stages: `"(plan, verify-plan, search,
  screen)"`;
  (b) Adding new node G2 between G ("re-run runner.py") and H ("delegate paper-analyzer") to
  show the post-screener deterministic stages: `"Deterministic stages continue (expand,
  download, convert-rough, extract, summarize)"`.

### Fixed

- **Bug 1 (BREAKING — daily-ai-intelligence/runners/subagent_contracts/rank_reviewer.yaml):
  `verdict_schema` fields did not match `reviewer_result.schema.json`**
  The `verdict_schema` section described a JSON structure that failed schema validation:
  `reviewer_id` (wrong) → `reviewer_task_id` (required by schema); `task_id` (extra, non-normative)
  removed; `verdict: "accept | reject"` (wrong field name + wrong enum) →
  `status: "accepted | rejected | accepted_with_issues"`; `findings: list[str]` (wrong field name)
  → `issues: list[str]`; `target_artifact` (required) was absent — added as
  `"{workspace}/{date}/clusters/ranked.jsonl"`. This is the same class of bug as v0.17.27 Bug A
  (which fixed `synthesis_reviewer.yaml`) but was never applied to `rank_reviewer.yaml`. An agent
  following the old contract would write a verdict file that failed schema validation, breaking the
  optional reviewer gate for the DAI skill. Also updated `completion_criteria`
  ("verdict is 'accept' or 'reject'" → "status is 'accepted' or 'rejected'").

- **Bug 2 (MEDIUM — daily-ai-intelligence/SKILL.md Rule 3): Rule 3 still referenced stale
  `verdict: reject` field and value**
  `daily-ai-intelligence/SKILL.md` Rule 3 said "If the optional rank_reviewer returns
  `verdict: reject`… Do not override a `reject` verdict." `reviewer_result.schema.json`
  uses `status` (not `verdict`) and the rejection value is `"rejected"` (not `"reject"`).
  The `synthesis_reviewer.yaml` contract and `research-pipeline/SKILL.md` Rule 4 were both
  corrected in v0.17.27–v0.17.28, but the equivalent DAI `SKILL.md` Rule 3 was not updated.
  Fixed by updating Rule 3 to reference `status: "rejected"` and `rejected` throughout.

- **Bug 3 (MEDIUM — daily-ai-intelligence/references/workflow-steps.md): `[review-ranked]`
  section still referenced stale `verdict: reject`**
  The `[review-ranked]` task description said "On `verdict: reject`, manually reset `[rank]` to
  `pending`…". Same stale field as Bug 2. Fixed to "On `status: \"rejected\"`".

## [v0.17.28] — 2026-05-14

### Fixed

- **Bug 1 (BREAKING — runners/runner.py): `_write_round_state` filtered gaps by wrong field `gap_type` instead of `classification`**
  `research-pipeline/runners/runner.py` `_write_round_state` built the `open_gaps`
  list for `round_state.json` using `g.get("gap_type") != "OUT_OF_SCOPE"`.
  `gap_classification.schema.json` defines the field as `classification` (enum:
  ACADEMIC, ENGINEERING, OUT_OF_SCOPE) — `gap_type` was the stale field name
  corrected in the `gap_classifier.yaml` contract by v0.17.25, but the runner itself
  was never updated. Because `g.get("gap_type")` always returns `None` and
  `None != "OUT_OF_SCOPE"` is always `True`, every gap including OUT_OF_SCOPE ones
  was included in `open_gaps`. This caused `round_state.json` to misreport the open
  gap count, misleading the `resume-inject.sh` context injection into reporting
  inflated gap counts to the agent on every new prompt. Fixed by changing
  `g.get("gap_type")` to `g.get("classification")`.

- **Bug 2 (MEDIUM — SKILL.md Rule 4): Rule 4 still referenced stale `verdict: reject` field and value**
  `research-pipeline/SKILL.md` Rule 4 said "If a reviewer sub-agent returns
  `verdict: reject`… Do not override a `reject` verdict." `reviewer_result.schema.json`
  uses `status` (not `verdict`) and the rejection value is `"rejected"` (not `"reject"`).
  The `synthesis_reviewer.yaml` contract was corrected to use `status: rejected` in
  v0.17.27, but `SKILL.md` was not updated at the same time. An orchestrating agent
  reading Rule 4 would look for the wrong field (`verdict`) and wrong value (`reject`),
  potentially never triggering the rejection handler. Fixed by updating Rule 4 to
  reference `status: "rejected"` and `rejected` throughout.

## [v0.17.27] — 2026-05-14

### Fixed

- **Bug A (BREAKING — synthesis_reviewer.yaml): `verdict_schema` fields did not match `reviewer_result.schema.json`**
  The `verdict_schema` section described a JSON structure that failed schema validation:
  `reviewer_id` (wrong) → `reviewer_task_id` (required by schema); `task_id` (extra, non-normative) removed;
  `verdict: "accept | reject"` (wrong field name + wrong enum) → `status: "accepted | rejected | accepted_with_issues"`;
  `findings: list[str]` (wrong field name) → `issues: list[str]`; `target_artifact` (required) was absent.
  An agent following the old contract would write a file that failed schema validation against
  `reviewer_result.schema.json`. Also updated `completion_criteria` ("verdict is 'accept' or 'reject'"
  → "status is 'accepted' or 'rejected'") and the `note` at the bottom, which incorrectly stated
  "the orchestrator reads the verdict and decides" — in reality the agent must manually update
  `workflow_state.json` and re-run runner.py.
  Additionally, `review_dimensions.*.verdict_field` names had a naming inconsistency with the
  `verdict_schema.scores` keys: suffixes `_score` and `_ok` were removed for uniform naming
  (`faithfulness_score` → `faithfulness`, `coherence_score` → `coherence`,
  `gap_completeness_score` → `gap_completeness`, `citation_integrity_ok` → `citation_integrity`),
  and `rejection_triggers` were updated accordingly to reference `scores.faithfulness` and
  `scores.citation_integrity`.

- **Bug B (MEDIUM — references/sub-agents.md): paper-analyzer `Writes` line said "returned in agent
  output; optionally written to `analysis/`"**
  The `paper_analyzer.yaml` contract requires writing to `{run_dir}/analysis/` (mandatory
  `completion_criteria` and `status_update` step 1: run `tool_analyze_papers --collect` to validate
  files on disk). The sub-agents.md entry incorrectly described these writes as optional, potentially
  misleading an orchestrating agent into thinking the paper-analyzer sub-agent need not write files to
  disk. Fixed to clearly state the output path and that it is required. (Same class as v0.17.21 fix for
  paper-synthesizer `Writes` line.)

- **Bug C (MEDIUM — runners/subagent_contracts/paper_synthesizer.yaml): `evidence_requirements` used
  field name `gap_type` instead of `classification`**
  `synthesis_report.schema.json` defines gaps with a `classification` field (enum: ACADEMIC,
  ENGINEERING, OUT_OF_SCOPE). The contract said "Open gaps must each have a `gap_type`" — an agent
  following the contract would use the wrong field name, leaving `classification` absent. Fixed to
  reference `classification` with all three valid values.

- **Bug D (LOW — runners/subagent_contracts/paper_synthesizer.yaml): `gap_classification` table
  missing `OUT_OF_SCOPE` entry**
  `synthesis_report.schema.json` and `gap_classifier.yaml` (downstream) both support
  `OUT_OF_SCOPE` as a valid gap classification, but the `paper_synthesizer` contract's
  `gap_classification` table only listed ENGINEERING and ACADEMIC, preventing the synthesizer
  from emitting `OUT_OF_SCOPE` gaps. Fixed by adding `OUT_OF_SCOPE` to the table.
  (Bug C and D fixed together in one edit.)

## [v0.17.26] — 2026-05-14

### Fixed

- **Bug 1 (BREAKING — manifest.json): `paper-screener` depended on `["search"]` instead of `["screen"]`**
  The runner processes tasks in manifest order. With `depends_on: ["search"]`, both `paper-screener`
  and `screen` were ready at the same time. The runner delegated `paper-screener` first, paused, and
  when re-run `screen` executed and **overwrote** `screened.jsonl` with the BM25 result — discarding
  the LLM screener's improved shortlist entirely. Fixed by setting `depends_on: ["screen"]` so the LLM
  screener always runs after BM25 screening and only refines its output. Also set `phase: "screen"` and
  added `output.path: "runs/{run_id}/screen/screened.jsonl"` to the manifest entry, and replaced the
  invalid `failure_policy.fallback` key (not enforced by the runner) with a `note` field.

- **Bug 2 (BREAKING — manifest.json): `expand`, `quality`, `enrich`, `download` depended on `["screen"]`
  instead of `["paper-screener"]`**
  Even with Bug 1 fixed, if `paper-screener` was `delegated` (paused for LLM), downstream tasks still
  saw `screen` as accepted and ran immediately — processing the BM25 shortlist before the LLM screener
  could improve it. Fixed by setting all four tasks to `depends_on: ["paper-screener"]`. Since
  `skipped_by_policy ∈ READY_STATUSES`, this is transparent in `quick`/`standard` profiles where
  `paper-screener` is not included (it is immediately skipped, and downstream tasks proceed normally).

- **Bug 3 (MEDIUM — paper_screener.yaml): contract primary input was `search/candidates.jsonl`**
  The sub-agent contract listed `search/candidates.jsonl` as primary input, but `cmd_screen.py` writes
  BM25 scores to `screen/cheap_scores.jsonl` (a richer file containing scores + explanation fields),
  and `references/sub-agents.md` already documented `cheap_scores.jsonl` as the correct input. Fixed
  by updating the contract to read `screen/cheap_scores.jsonl` as primary, with `search/candidates.jsonl`
  as a fallback if BM25 scores are unavailable.

- **Bug 4 (MEDIUM — sub-agents.md): paper-screener `Writes` line said "returned in agent output"**
  The documentation stated that the LLM screener's output was "returned in agent output", contradicting
  the `paper_screener.yaml` contract which requires writing to `{run_dir}/screen/screened.jsonl` and
  uses `artifact_exists` as a completion criterion. Fixed by correcting both the `Reads` line
  (to `screen/cheap_scores.jsonl`) and the `Writes` line (to `{run_dir}/screen/screened.jsonl`).

- **Bug 5 (MEDIUM — iterative-synthesis.md): wrong field name, missing state-reset instruction,
  and undefined template variable in runner invocation (step 3)**
  Three errors in the per-round invocation instructions:
  1. "set `current_round = <N+1>`" — `workflow_state.json` uses `"round"`, not `"current_round"`.
     (`current_round` belongs to the separate `round_state.json` written by hooks.) This would cause
     the runner to start round 2 with `round: 1` still in state, corrupting round tracking.
  2. No instruction to reset task statuses to `pending`. The runner reads `workflow_state.json` and
     sees all tasks as `accepted` from round 1, making no progress on the new round. The correct
     preparation is to copy `workflow_state_template.json` (which has all tasks as `pending`) and
     then update `run_id`, `round`, `context.prior_paper_ids`, and `context.prior_gaps`.
  3. `--config CFG` — `CFG` is an undefined bare word. All other template variables in the docs use
     `{variable}` syntax. Fixed to `--config {config}`.
  Replaced the single "Before invoking…" sentence with a numbered 5-step preparation checklist that
  covers all necessary `workflow_state.json` fields before the runner is invoked.

## [v0.17.25] — 2026-05-14

### Fixed

- **Bug (sub-agents.md): Mermaid flowchart node still used `--topic` flag after v0.17.24 partial fix**
  `research-pipeline/references/sub-agents.md` Mermaid flowchart node A showed
  `python3 runner.py --topic TOPIC --profile deep`. `runner.py` declares `topic`
  as a **positional** argument (`parser.add_argument("topic", nargs="?", …)`), so
  `--topic` is unrecognized by argparse and raises "unrecognized arguments" at
  runtime. v0.17.23 fixed `python` → `python3` in this exact node; v0.17.24 then
  fixed the identical `--topic` issue in `iterative-synthesis.md` and stated "This
  pattern was already correct everywhere else" — but the `sub-agents.md` Mermaid
  node was still wrong. Fixed by removing `--topic` and using the positional form:
  `python3 runner.py TOPIC --profile deep`.

- **Bug (gap_classifier.yaml): `output_schema` field names contradicted `gap_classification.schema.json` — BREAKING**
  `research-pipeline/runners/subagent_contracts/gap_classifier.yaml` `output_schema`
  section described the `gaps.json` structure using field names that did not match the
  normative `schemas/gap_classification.schema.json`. An agent following the YAML
  contract would produce a `gaps.json` that fails schema validation because the
  required fields `id` and `classification` were absent. The specific mismatches:
  - `gap_id` → must be `id` (schema `required`)
  - `gap_type: ENGINEERING | ACADEMIC` → must be `classification: ACADEMIC | ENGINEERING | OUT_OF_SCOPE` (schema `required`; enum extended to include `OUT_OF_SCOPE`)
  - `search_queries: list[str]` → must be `suggested_search_query: str` (different name
    and type: list vs. single string)
  - `engineering_refs: list[str]` → not in schema; replaced with `resolution_notes: str`
    which is the schema-sanctioned field for per-gap notes
  - `priority` enum values `high | medium | low` → must be `HIGH | MEDIUM | LOW`
  - Top-level required fields `run_id`, `round`, and `convergence` were entirely absent
    from the `output_schema` illustration. The runner reads `convergence.should_continue`
    to decide whether to continue iterating; an agent omitting `convergence` causes the
    runner to silently treat convergence as `false` and stop after one round even when
    open ACADEMIC gaps remain.

  Additionally fixed in the same contract:
  - `description`: mentioned only ENGINEERING/ACADEMIC; added OUT_OF_SCOPE.
  - `classification_criteria`: was missing an OUT_OF_SCOPE entry; added it.
  - `completion_criteria`: used `search_query` (inconsistent with both the old
    `search_queries` and the correct `suggested_search_query`) and `engineering_ref`
    (not in schema); both corrected to match schema field names.
  - `instructions`: added step 4 explaining how to populate the `convergence` object,
    since this is a runner-observable required field that the agent must produce.


### Fixed

- **Bug (iterative-synthesis.md): `--topic` flag does not exist in runner.py — BREAKING**
  `research-pipeline/references/iterative-synthesis.md` showed the per-round
  gap-closure `runner.py` invocation using `--topic "<gap-specific topic>"`.
  However `runner.py` declares `topic` as a **positional** argument
  (`parser.add_argument("topic", nargs="?", …)`), so passing `--topic` would cause
  argparse to raise "unrecognized arguments". Fixed by removing the `--topic` flag
  and using the bare positional form: `runner.py "<gap-specific topic>" --profile …`.
  This pattern was already correct everywhere else in the skill documentation.

- **Bug (rank_reviewer.yaml): `forbidden_actions` referenced wrong filename `ranked_events.json`**
  `daily-ai-intelligence/runners/subagent_contracts/rank_reviewer.yaml` line 32
  said `Do NOT edit ranked_events.json`. The actual output file written by
  `brief_rank_events` is `ranked.jsonl` (manifest path:
  `{workspace}/{date}/clusters/ranked.jsonl`). The same stale name was fixed in
  `workflow-steps.md` in v0.17.21 but the contract file was overlooked. Fixed to
  `Do NOT edit ranked.jsonl`.

- **Bug (synthesis_reviewer.yaml): `forbidden_actions` referenced wrong filename `synthesis_report.json`**
  `research-pipeline/runners/subagent_contracts/synthesis_reviewer.yaml` line 34
  said `Do NOT edit synthesis_report.json or synthesis.md`. `synthesis_report.json`
  lives in `{run_dir}/summarize/` and is not one of the reviewer's input files;
  the reviewer's actual input is `{run_dir}/analysis/synthesis.json` (written by
  `paper-synthesizer`). The identical class of wrong filename was fixed in
  `paper_synthesizer.yaml` in v0.17.21 but this contract was missed. Fixed to
  `Do NOT edit synthesis.json or synthesis.md`.

- **Bug (manifest.json): `review-synthesis` missing `paper-synthesizer` dependency — race condition**
  `research-pipeline/manifest.json` task `review-synthesis` declared
  `"depends_on": ["report"]`. The synthesis reviewer reads
  `{run_dir}/analysis/synthesis.json` which is written by `paper-synthesizer`, not
  by `report`. Because `report` and `paper-synthesizer` are on parallel paths in
  the task DAG (both descend from `convert-rough` independently), the runner could
  delegate `review-synthesis` while `paper-synthesizer` was still in `delegated`
  state — causing the sub-agent to try to read a file that does not yet exist.
  Fixed by changing `"depends_on"` to `["paper-synthesizer", "report"]`, so
  `review-synthesis` only becomes ready after `paper-synthesizer` is accepted.
  In non-deep profiles (quick, standard) where neither `paper-synthesizer` nor
  `review-synthesis` is in scope, both receive `skipped_by_policy` immediately, so
  the extra dependency is safe.

## [v0.17.23] — 2026-05-14

### Fixed

- **Bug (DAI runner): `run_deterministic` lacked timeout — could hang forever**
  `daily-ai-intelligence/runners/runner.py` called `subprocess.run(…)` with no
  `timeout` argument and no `subprocess.TimeoutExpired` handler. If
  `validate-registry.sh` or `check_completion.sh` ever hung, the DAI runner would
  block indefinitely. The RP runner already had `timeout=300` with a proper
  `TimeoutExpired` catch. Fixed by mirroring the RP runner pattern: pass
  `timeout=300` to `subprocess.run` and wrap in `try/except subprocess.TimeoutExpired`
  returning `False, "command timed out after 300s: ..."`.

- **Bug (DAI manifest): `export-obsidian` had `on_failure: block` but is `optional: true`**
  `daily-ai-intelligence/manifest.json` task `export-obsidian` declared
  `"optional": true` but `"failure_policy": {"on_failure": "block", ...}`. While
  the DAI runner never actually evaluates the failure policy for MCP tool tasks
  (they are always auto-accepted), the policy was misleading and violated the
  invariant that optional tasks must use `on_failure: skip`. The identical pattern
  was fixed for the RP skill's `paper-synthesizer` in v0.17.21 but the DAI
  `export-obsidian` was missed. Fixed by changing `"on_failure"` from `"block"` to
  `"skip"` and renaming the explanatory key from `"message"` to `"note"` to match
  all other optional tasks.

- **Bug (iterative-synthesis.md + sub-agents.md): remaining bare `python` invocations**
  `v0.17.22` fixed SKILL.md and `workflow-steps.md` for both skills but missed two
  further locations where runner.py was invoked with bare `python`:
  - `research-pipeline/references/iterative-synthesis.md` — bash code block in the
    per-round procedure (step 3) used `python {skill_dir}/runners/runner.py`
  - `research-pipeline/references/sub-agents.md` — Mermaid flowchart node
    `A["You: python runner.py …"]`
  Both fixed to `python3`.

- **Chore (runner.py + check_completion.py docstrings): remaining bare `python` in Usage blocks**
  The module-level docstrings in both `research-pipeline/runners/runner.py` and
  `daily-ai-intelligence/runners/runner.py`, and `scripts/check_completion.py`,
  still showed bare `python` in their `Usage:` examples. These are read by agents
  consulting the source for CLI invocation hints. Fixed all to `python3`.

## [v0.17.22] — 2026-05-14

### Fixed

- **Bug (DAI runner): `print_llm_delegation` ignored manifest-declared contract path**
  `daily-ai-intelligence/runners/runner.py` always derived the contract filename from
  the task ID (`task_id.replace("-","_") + ".yaml"`), so for the `review-ranked` task
  it looked for `review_ranked.yaml` instead of the actual file `rank_reviewer.yaml`
  declared in `executor.contract`. Fixed by mirroring the RP runner: first check
  `task["executor"].get("contract", "")` and resolve relative to `SKILL_DIR`; fall
  back to the derived name only when no manifest path is declared.

- **Bug (DAI runner): `print_llm_delegation` never substituted context variables**
  The function took no `ctx` parameter and printed raw contract text, so the reviewer
  sub-agent saw literal `{workspace}/{date}/...` template placeholders instead of real
  paths. Fixed by adding a `ctx` parameter and iterating over its key/value pairs to
  replace placeholders in the contract text — matching the pattern already used in
  the RP runner. Updated the call site from `print_llm_delegation(task)` to
  `print_llm_delegation(task, ctx)`.

- **Bug (DAI stop-check.sh): hook blocked agent when no brief was run today**
  `daily-ai-intelligence/hooks/stop-check.sh` called `check_completion.sh` without
  `--workspace` / `--date`, so `check_completion.sh` exited 1 when
  `./workspace/briefing/<today>/` did not exist, causing the hook to exit 2 and block
  the agent in every project that had the hook installed globally — even when no brief
  had been started. The header comment said "The hook is a no-op when no brief workspace
  is found for today" but the code contradicted this. Fixed by adding a sentinel check
  after locating `$CHECK_SCRIPT`: if `./workspace/briefing/$(date -u +%F)` does not
  exist, exit 0 immediately — mirroring the RP `stop-check.sh` pattern.

- **Bug (SKILL.md + workflow-steps.md): bare `python` invocations of runner.py**
  Four locations still used bare `python` instead of `python3` for runner.py launch
  examples. `v0.17.21` fixed `manifest.json` and the `check_completion.py` debug line
  in `references/workflow-steps.md`, but missed the actual Launch blocks and the
  Profiles section. Fixed:
  - `research-pipeline/SKILL.md` Launch block
  - `daily-ai-intelligence/SKILL.md` Launch block
  - `research-pipeline/references/workflow-steps.md` Profiles section (two lines)
  - `daily-ai-intelligence/references/workflow-steps.md` core-pipeline section

## [v0.17.21] — 2026-05-14

### Fixed

- **Bug (workflow-steps.md RP): resume_context.json field names were wrong in debug block**
  `references/workflow-steps.md` resume-check section (prose description and the
  Python debug snippet) used stale key names that never matched what `resume-check.sh`
  actually writes. Updated four keys:
  - `ctx["resuming"]` → `ctx["resume"]`
  - `ctx["prior_report_path"]` → `ctx["snapshot"]`
  - `ctx["prior_arxiv_ids"]` → `ctx["prior_paper_ids"]`
  - `ctx["prior_gaps"]` → `ctx["open_gaps_raw"]`
  Also updated the prose description on line 27 ("with `prior_arxiv_ids` and
  `prior_gaps`" → "with `prior_paper_ids` and `open_gaps_raw`") to match.
  The ground-truth keys are defined in `scripts/resume-check.sh` (unchanged).

- **Bug (gap_classifier contract): completion_criteria referenced wrong primary source**
  `runners/subagent_contracts/gap_classifier.yaml` `completion_criteria` said
  "Every gap in `{run_dir}/analysis/synthesis.json`…" but that path is the
  *optional* deep-profile output from `paper-synthesizer`; the *primary* source
  is `{run_dir}/summarize/synthesis_report.json`. The `inputs` section already
  correctly named both files with the right primary/optional distinction — only
  `completion_criteria` was inconsistent. Updated the criterion to reference the
  primary source first and the alternative second, matching the `instructions`
  section logic.

- **Bug (workflow-steps.md DAI): incorrect claim that runner auto-re-queues [rank] on reviewer reject**
  `daily-ai-intelligence/references/workflow-steps.md` task `[review-ranked]`
  section said "On `verdict: reject`, the runner re-queues `[rank]`." The runner
  does NOT automatically reset any task status — that must be done manually by
  the agent. Updated to: "On `verdict: reject`, manually reset `[rank]` to
  `pending` in `workflow_state.json` and re-run the runner."

- **Bug (workflow-steps.md DAI): wrong artifact filename for ranked clusters**
  The `[poll]+[rank]+[generate-daily]` section listed
  `<WS>/<DATE>/clusters/clusters.jsonl` as the ranked-clusters artifact, but the
  DAI manifest `rank` task output path is `{workspace}/{date}/clusters/ranked.jsonl`.
  Updated to `ranked.jsonl`.

- **Bug (sub-agents.md): paper-synthesizer Outputs section described CLI tool outputs, not sub-agent outputs**
  The `## paper-synthesizer` section listed `synthesis_report.md`,
  `synthesis_report.json`, and `synthesis_traceability.json` as outputs — these
  are files written by the deterministic CLI `summarize` stage, not by the
  `paper-synthesizer` sub-agent. The sub-agent writes to
  `{run_dir}/analysis/synthesis.md` and `{run_dir}/analysis/synthesis.json`.
  Updated the `Outputs` field, the prompt template write instruction (was
  `/runs/<run_id>/summarize/` → `/runs/<run_id>/analysis/`), and the `Writes`
  line to reflect the correct paths.

- **Bug (paper_synthesizer contract): status_update referenced wrong output filename**
  `runners/subagent_contracts/paper_synthesizer.yaml` `status_update` said
  "Verify synthesis_report.json validates against the schema" but the actual
  output is `synthesis.json` (in `{run_dir}/analysis/`). Updated to
  "Verify synthesis.json validates against the schema".

- **Bug (DAI manifest): dossier task had wrong `type` field**
  `daily-ai-intelligence/manifest.json` `dossier` task declared
  `"type": "llm_worker_task"` but its executor is
  `"kind": "deterministic_mcp_tool"`. The `type` field is metadata — the runner
  uses `executor.kind` — but the inconsistency was misleading. Changed `type`
  to `"pipeline_stage"` to match all other optional MCP tasks (`feedback`,
  `export-obsidian`, etc.).



### Fixed

- **Bug (DAI runner): `task_ready` blocked on `skipped_by_policy` dependencies**
  `daily-ai-intelligence/runners/runner.py`: `task_ready()` only accepted `"accepted"`
  as a satisfying dependency status, causing `preferences` (which depends on optional
  `feedback`) to stay permanently `pending` when `feedback` was skipped. Added
  `READY_STATUSES = {"accepted", "skipped_by_policy"}` constant — mirroring the
  research-pipeline runner — and updated `task_ready()` to use `not in READY_STATUSES`.

- **Bug (research-pipeline manifest): `check-completion` used bare `python` not `python3`**
  `manifest.json` check-completion executor command used `python` which fails on
  Python-3-only systems. Changed to `python3` for consistency with all other skill
  scripts (`resume-check.sh`, `stop-check.sh`, `resume-inject.sh`, `validate-registry.sh`).
  Also fixed the debug invocation example in `references/workflow-steps.md`.

- **Bug (gap_classifier contract): `status_update` referenced wrong output filename**
  `runners/subagent_contracts/gap_classifier.yaml` `status_update` said
  `gap_classifications.json` but the actual output (per manifest and runner code) is
  `gaps.json`. This was a leftover from the partial fix in `0a3aaeb`. Updated to `gaps.json`.

- **Bug (workflow-steps.md): gap-closure section referenced wrong filename**
  `references/workflow-steps.md` gap-closure section said `read gap_classifications.json`
  but the file is `gaps.json`. Updated to `gaps.json`.

- **Bug (research-pipeline manifest): `paper-synthesizer` had conflicting `failure_policy`**
  `paper-synthesizer` is declared `optional: true` but had `"on_failure": "block"`.
  A task that is semantically optional must not block the workflow on failure. Changed
  `"on_failure": "block"` → `"on_failure": "skip"` with explanatory note.

- **Bug (research-pipeline manifest + contract): missing `paper_analysis.schema.json`**
  `manifest.json` output section for `paper-analyzer` and the `paper_analyzer.yaml`
  sub-agent contract both reference `schemas/paper_analysis.schema.json` which did not
  exist. Created the schema with all required fields from the analysis template:
  `paper_id`, `title`, `research_question`, `methodology`, `key_findings`
  (with `evidence_type` enum), `limitations`, `reproducibility`, `confidence_scores`,
  `raw_claims`.

- **Bug (sub-agents.md): stale `_analysis` filename convention**
  `references/sub-agents.md` still used the old underscore convention
  (`{arxiv_id}_analysis.json`) after `0a3aaeb` changed to dot-separator
  (`{arxiv_id}.analysis.json`). Updated both occurrences.


## [v0.17.14] — 2026-05-13

### Added
- **Final Daily AI Intelligence completeness audit** (Phases A–G):
  - All 63 tickets (Phases A–G) confirmed `audit_pass` via MCP `run_implementation_check`
    against `daily-ai-intelligence-implementation-plan.md` (260 satisfied, 0 violated)
  - `docs/daily-ai-intelligence/final-traceability-matrix.md` — 63-row feature-to-implementation map
  - `docs/daily-ai-intelligence/final-gap-register.md` — empty gap register (no gaps found)
  - `docs/daily-ai-intelligence/final-completeness-audit-report.md` — complete audit report
  - `phase-status.yaml` updated: `final_audit.status: complete`, `verdict: no_gaps_found`
- **Architecture compliance artifacts** in `.agent/artifacts/`:
  - `compliance_report.md` — overall partially_compliant (101 satisfied, 0 violated, 1 unknown)
  - `impl_check_daily_ai.json` — 260 satisfied, 0 violated from daily-AI-intelligence plan
  - `impl_check_implplan.json` — implementation-plan.md fully compliant (118 satisfied, 0 violated)


## [v0.17.13] — 2026-05-12

### Added
- **Multi-agent reliability structs** (Research Report Rec. 4):
  - `AgentDiversityConfig` — enforces ≥N model families across sub-agent pool
    with `validate_diversity()` returning warnings on insufficient diversity
  - `SubAgentBudget` — per-role token limits (max + target) with `SUB_AGENT_BUDGETS`
    defaults for paper-analyzer, synthesizer, screener, and report-generator
  - `AgentsConfig` — wrapper for diversity + budgets, exposed via `PipelineConfig.agents`
  - `PreCommitmentPolicy` enum (PARALLEL / SEQUENTIAL_BLIND / SEQUENTIAL_INFORMED)
    for controlling sub-agent dispatch isolation
- **Minority finding tracking** in synthesis:
  - `MinorityFinding` model with `finding`, `supporting_sources`, `contradicting_sources`,
    `evidence_quality`, `evaluation`, and `suppression_risk` fields
  - `SynthesisReport.minority_findings` and `SynthesisReport.consensus_confidence`
    populated by `_build_template_synthesis()` from `_detect_dissent()` output
- **Memory lifecycle hooks** on `MemoryManager`:
  - `consolidate()` — promotes episodic memories to semantic store via `EpisodeStore`
  - `between_stages(new_stage, consolidation_threshold=20)` — resets working memory
    and auto-triggers consolidation when episodic count exceeds threshold
  - Orchestrator now calls `memory.between_stages()` at all 7 stage boundaries

## [v0.17.12] — 2026-05-13

### Added
- **deep profile orchestration**: Wire `expand`, `quality`, `analyze_claims`, and
  `score_claims` stages into the `run` orchestrator pipeline for the deep profile.
  Previously these stages were no-ops (log-only TODO stubs) even when selected by
  `--profile deep`.
  - `expand`: auto-expands citation graph from shortlisted paper IDs via Semantic Scholar
  - `quality`: computes composite quality scores (citation impact, venue reputation, recency)
  - `analyze_claims`: decomposes paper summaries into atomic claims with evidence classification
  - `score_claims`: scores confidence for decomposed claims using multi-signal aggregation
- Stage verifiers for all 4 new deep-profile stages (`_verify_expand`, `_verify_quality`,
  `_verify_analyze_claims`, `_verify_score_claims`) added to `STAGE_VERIFIERS`
- Full resume support for all 4 new stages

## [v0.17.10] — 2026-05-13

### Added
- **GAP-005**: `--retry-failed` flag on `download` command (spec §6.2 dead-letter tracking)
  Reads existing `download_manifest.jsonl`, filters entries with `status='failed'`,
  re-attempts only those, and merges results back with successful entries.

## [v0.17.9] — 2026-05-13

### Fixed
- **GAP-002**: Align composite quality DEFAULT_WEIGHTS with spec §5
  (citation: 0.35, venue: 0.25, author: 0.25, recency: 0.15, reproducibility: 0.0)
- Add `reproducibility_weight` to `QualityConfig` and propagate to all callers
  (`cmd_quality.py`, `mcp_server/tools.py`)

### Added
- **GAP-003**: New MCP tool `get_venue_tier` — look up CORE venue tier and score
- **GAP-004**: New MCP tool `compute_semantic_scores` — SPECTER2 semantic similarity
  scores for screened candidates (spec §3.5)
- New `GetVenueTierInput` and `ComputeSemanticScoresInput` schemas in `mcp_server/schemas.py`

## [v0.17.8] — 2026-05-12

### Fixed

- Corrected stale environment variable name in `docs/architecture.md`:
  `ARXIV_PAPER_PIPELINE_CONFIG` → `RESEARCH_PIPELINE_CONFIG` (the
  implementation has used `RESEARCH_PIPELINE_CONFIG` since v0.3.0; the
  documentation lagged behind).

### Changed

- Updated `astral-sh/setup-uv` from `@v7` to `@v8` in all GitHub Actions
  workflows (`ci.yml`, `docs.yml`, `daily-brief.yml`, `publish.yml`).

## [v0.17.0] — 2026-04-28

### Added

- Added a parallel daily AI intelligence briefing pipeline under
  `research_pipeline.briefing` with governed source registry loading, stable
  event and cluster IDs, JSONL artifacts, telemetry, exact deduplication,
  deterministic ranking, report validation, and replayable workflow state.
- Added `research-pipeline brief ...` commands for polling sources, ranking
  events, generating and validating daily briefs, running the full workflow,
  recording feedback, reviewing topic aliases, exporting Obsidian notes,
  generating dossiers, computing preference adjustments, resuming runs,
  comparing source-expanded runs, and weekly synthesis.
- Added Phase A-G briefing surfaces: GitHub releases, RSS/Atom, manual,
  Hacker News, Hugging Face papers, and arXiv-style source adapters; topic
  memory/fatigue; explicit feedback and reversible preferences; Obsidian
  daily/topic/source notes; hot-topic dossiers; MCP `brief_*` tools and
  `briefings://` resources.
- Added a bundled `daily-ai-intelligence` skill with command, source-policy,
  report-template, feedback-loop, troubleshooting, and evaluation references.
- Added offline briefing fixtures and unit tests for the briefing workflow,
  source expansion gates, feedback conflicts, alias review, dossier
  generation, replay, MCP resources, and report quality.

### Changed

- `research-pipeline setup` now discovers and installs all bundled skills
  instead of only the academic `research-pipeline` skill.
- Daily briefing reports now suppress filler items, select the strongest
  evidence from duplicate clusters, label factual evidence, expose dynamic
  novelty/confidence/action fields, and mark reports as validated after
  deterministic validation passes.

## [v0.16.2] — 2026-04-27

### Changed

- Moved the MCP server into the packaged source tree at
  `src/research_pipeline/mcp_server/` so it is included in installed wheels.
- Added the packaged MCP CLI surface:
  `research-pipeline mcp serve` and `research-pipeline mcp config`.
- Made packaged skill and agent data the canonical setup source, removing the
  duplicate `.github/skills/` and `.github/agents/` copies that could drift.
- Extended `research-pipeline setup` to install a reusable MCP config snippet
  at `~/.config/research-pipeline/mcp.json` alongside skills and agents.
- Updated human and AI-agent documentation for the packaged MCP, skill, and
  agent layout.

## [v0.16.1] — 2026-04-22

### Changed

- **Skill refined per Anthropic's Skill-Building Guide** (bumped skill
  metadata version 1.9.0 → 1.10.0):
  - Stronger YAML `description` with explicit positive and negative
    trigger phrases (e.g., "resume a prior research report", "fill
    research gaps"; explicit redirects for general web search,
    one-off PDF conversion, and requirements analysis).
  - Added `license: MIT` and `compatibility` frontmatter fields.
  - Added **`## When To Trigger`** section listing trigger phrases
    and out-of-scope requests.
  - Added **`## Examples`** section with four concrete user-prompt →
    action pairs (fresh review, resume prior report, system-building,
    out-of-scope redirect).
  - Reformatted `## Critical Rules` into numbered, scannable items.
- README, AGENTS.md, and `.github/copilot-instructions.md` updated to
  document resume-on-top, 4-round gap-closure, and required report
  formatting (Contents, Round History, Mermaid, LaTeX).

## [v0.16.0] — 2026-04-20

### Added

- **Unified Horizon Metric (UHM)** — closes gap A3-5 from the Deep Research
  Report. Combines difficulty-weighted normalized score, horizon efficiency,
  stability (UltraHorizon token-entropy trend), and Pass[k] reliability into
  a single scalar in `[0, 1]` via geometric mean + reliability gate.
  See `src/research_pipeline/evaluation/horizon.py`.
  - New CLI: `research-pipeline horizon --score ... --achieved ... --target ...`
  - New MCP tool: `tool_horizon_metric`.
- **Recall / Reasoning / Presentation (RRP) diagnostic** — operationalizes
  the DeepResearch Bench II finding (Theme 16) that Information Recall is
  the dominant bottleneck (<50% of expert rubrics satisfied) while
  Presentation is usually near-saturated. Decomposes a synthesis report
  into three axes and identifies the bottleneck.
  See `src/research_pipeline/evaluation/recall_diagnostic.py`.
  - New CLI: `research-pipeline rrp --report <md> --shortlist <json>`
  - New MCP tool: `tool_rrp_diagnostic`.
- 30 new unit and CLI integration tests for the metrics above.

### Changed

- MCP server now registers 53 tools (was 51).

## [v0.14.4] — 2026-04-19

### Added

- Step 1 structured per-paper extraction records with typed statements,
  evidence snippets, confidence labels, uncertainty notes, and quality scores.
- Step 2 design-neutral cross-paper synthesis with taxonomy, evidence matrix,
  recurring patterns, assumption map, contradiction map, evidence-strength map,
  operational implications, risk register, and traceability appendix.
- `research-pipeline summarize --step extraction|synthesis|all` for running
  the structured stages independently.
- `structured_synthesis` report template and validation support for the new
  synthesis report shape.

### Changed

- `summarize` now writes rich artifacts under `summarize/extractions/` and
  `summarize/synthesis_report.*` while preserving legacy `*.summary.json` and
  `synthesis.json` projections.
- Bundled AI skill and human docs now describe the structured extraction and
  synthesis workflow.

## [v0.14.3] — 2026-04-18

### Added

- 64 new unit tests covering 16 previously untested CLI handlers and scholar source
- CI coverage threshold raised from 80% to 85%

### Fixed

- Remove deprecated PEP 639 `License :: OSI Approved :: MIT License` classifier
  (`license = "MIT"` SPDX identifier is sufficient)

### Changed

- Coverage: 80% → 86% (3430 tests, 15816 statements, only `__main__.py` at 0%)

## [v0.14.2] — 2026-04-18

### Added

- 98 new unit tests across 7 files covering 20+ previously untested modules
- CI coverage threshold raised from 75% to 80%
- CI coverage report XML artifact upload (Python 3.12)
- README badges: CI status, mypy, ruff
- GitHub repo description and 10 topics
- PyPI classifiers: Scientific/Engineering, MIT License

### Changed

- Development Status classifier upgraded: Alpha → Beta
- Coverage: 77% → 80% (3362 tests, 15816 statements)

## [v0.14.1] — 2026-04-18

### Fixed

- Fix all 241 mypy errors across 59 source files (now 0 errors in 211 files)
- Fix real bugs found via type checking: non-existent `config.runs_dir`,
  `resolve_workspace`, `LLMProvider.complete()`, wrong `get_stage_dir` args
- Fix variable type conflicts from shadowing in 5 modules
- Enforce mypy in CI (removed `|| true` fallback)

### Added

- PEP 561 `py.typed` marker for downstream type checking
- CLI smoke tests (22 subcommands verified via `--help`)
- Coverage threshold `--cov-fail-under=70` in CI
- Expanded ruff rules: A, C4, PT, RUF groups

## [v0.14.0] — 2026-04-18

### Changed

- Replace Black + isort with Ruff format/check (single toolchain)
- Modernize pre-commit hooks: add bandit, validate-pyproject, remove legacy hooks

### Added

- pip-audit and pip-licenses for security & license auditing
- vulture for dead code detection
- hypothesis for property-based testing (36 tests)
- Security & License Audit CI job
- Python 3.13 added to CI test matrix

## [v0.13.52] — 2026-04-18

### Added

- MCP tool wrappers for 6 new commands (cite-context, cluster, export-bibtex,
  eval-log, feedback, export-html) with integration tests (C2+C3)

## [v0.13.51] — 2026-04-18

### Added

- CHANGELOG.md auto-generated from git history (C5)

## [v0.13.50] — 2026-04-18

### Documentation

- update user-guide with 6 new CLI commands (v0.13.44-49)

## [v0.13.49] — 2026-04-18

### Added

- watch mode for topics (B8) — periodic new paper monitoring

## [v0.13.48] — 2026-04-18

### Added

- citation context extraction (B7) — in-text citation sentence extraction

## [v0.13.47] — 2026-04-18

### Added

- abstract enrichment pipeline (B6) — DOI + title-based S2 lookup

## [v0.13.46] — 2026-04-18

### Added

- paper similarity clustering (B5) — TF-IDF + agglomerative clustering

## [v0.13.45] — 2026-04-18

### Added

- report template system with 4 built-in formats (B4)

## [v0.13.44] — 2026-04-18

### Added

- BibTeX export from candidate records (B2)

## [v0.13.43] — 2026-04-18

### Fixed

- sync black/ruff versions in pre-commit, use pre-commit in CI, track skill config.toml

## [v0.13.42] — 2026-04-18

### Fixed

- set black target-version py312, add jinja2 to dev deps

## [v0.13.41] — 2026-04-18

### CI

- add GitHub Actions CI workflow (lint, test, typecheck)

## [v0.13.40] — 2026-04-18

### Added

- full multi-source parallel search + incremental runs via global index

## [v0.13.39] — 2026-04-17

### Added

- wire 5 standalone modules into pipeline flow (Tier A integration)

## [v0.13.38] — 2026-04-17

### Added

- add 7-Dimension Coherence Evaluation framework

## [v0.13.37] — 2026-04-17

### Added

- add Scientific KG Benchmark framework

## [v0.13.36] — 2026-04-17

### Added

- add RL query reformulation with Thompson sampling bandit

## [v0.13.35] — 2026-04-17

### Added

- add adaptive difficulty routing (v0.13.35)

## [v0.13.34] — 2026-04-17

### Added

- add multi-model consensus engine (v0.13.34)

## [v0.13.33] — 2026-04-17

### Added

- add query-typed retrieval stopping profiles (v0.13.33)

## [v0.13.32] — 2026-04-17

### Added

- forward citation traversal with budget-aware stopping

## [v0.13.31] — 2026-04-17

### Added

- pre-commitment protocol for conformity bias elimination

## [v0.13.30] — 2026-04-17

### Added

- retention regularization — drift detection and score penalty

## [v0.13.29] — 2026-04-17

### Added

- add full environment snapshot capture at stage boundaries (v0.13.29)

## [v0.13.28] — 2026-04-17

### Added

- add graduated rubric scoring with 4-level grades (v0.13.28)

## [v0.13.27] — 2026-04-17

### Added

- add hash-pinned tool definitions for MCP integrity (v0.13.27)

## [v0.13.26] — 2026-04-17

### Added

- add length normalization for LLM responses (v0.13.26)

## [v0.13.25] — 2026-04-17

### Added

- claim-level citation accuracy scoring (v0.13.25)

## [v0.13.24] — 2026-04-17

### Added

- non-destructive versioned memory with rollback (v0.13.24)

## [v0.13.23] — 2026-04-17

### Added

- failure taxonomy logging with JSONL persistence (v0.13.23)

## [v0.13.22] — 2026-04-17

### Added

- structured output enforcement for LLM responses (v0.13.22)

## [v0.13.21] — 2026-04-17

### Added

- segment-level memory entries with token-aware splitting

## [v0.13.20] — 2026-04-17

### Added

- Q2D query augmentation with domain synonym expansion

## [v0.13.19] — 2026-04-17

### Added

- citation budget stopping criteria for BFS expansion

## [v0.13.18] — 2026-04-17

### Added

- query noise removal with academic boilerplate filtering

## [v0.13.17] — 2026-04-17

### Added

- heuristic dissent preservation in template-mode synthesis

## [v0.13.16] — 2026-04-17

### Added

- true MMR diversity with Jaccard document similarity in screening

## [v0.13.15] — 2026-04-17

### Added

- add 4-layer confidence architecture (C4)

## [v0.13.14] — 2026-04-17

### Added

- add query-adaptive retrieval stopping criteria (C3)

## [v0.13.13] — 2026-04-17

### Added

- add KG quality evaluation framework (5-dimension, 3-layer)

## [v0.13.12] — 2026-04-17

### Added

- add Case-Based Reasoning (CBR) for strategy reuse

## [v0.13.11] — 2026-04-17

### Added

- add Pass@k + Pass[k] dual metrics evaluation (B6)

## [v0.13.10] — 2026-04-17

### Added

- add epistemic blinding audits (B5)

## [v0.13.9] — 2026-04-17

### Added

- memory consolidation engine (B4)

## [v0.13.8] — 2026-04-16

### Added

- multi-session coherence evaluation (B3)

## [v0.13.7] — 2026-04-16

### Added

- human-in-the-loop approval gates (B2)

## [v0.13.6] — 2026-04-16

### Added

- phase-aware model routing (B1)

## [v0.13.5] — 2026-04-16

### Fixed

- correct config access and synthesis filename in aggregate/export-html

## [v0.13.4] — 2026-04-16

### Added

- add HTML report export with Jinja2 templates (A5)

## [v0.13.3] — 2026-04-16

### Added

- bidirectional citation snowball with budget-aware stopping (A4)

## [v0.13.2] — 2026-04-16

### Added

- evidence-only aggregation (A3)

## [v0.13.1] — 2026-04-16

### Added

- three-channel eval logging (A2)

## [v0.13.0] — 2026-04-16

### Added

- add user feedback loop for screening weight adjustment (v0.13.0)

## [v0.12.14] — 2026-04-16

### Added

- add MCP zero-trust security (T3-6)

## [v0.12.13] — 2026-04-16

### Added

- add multi-agent architecture (T3-5)

## [v0.12.12] — 2026-04-15

### Added

- add tiered page dispatch (T3-4, v0.12.12)

## [v0.12.11] — 2026-04-15

### Added

- add self-improving retrieval (T3-3, v0.12.11)

## [v0.12.10] — 2026-04-15

### Added

- schema-grounded evaluation with per-stage validation

## [v0.12.9] — 2026-04-15

### Added

- content security gates with classification and taint tracking

## [v0.12.8] — 2026-04-15

### Added

- three-tier memory architecture (working/episodic/semantic)

## [v0.12.7] — 2026-04-15

### Added

- THINK→EXECUTE→REFLECT iterative gap-filling loop

## [v0.12.6] — 2026-04-15

### Chore

- bump pyproject.toml version to 0.12.6

## [v0.12.5] — 2026-04-15

### Added

- per-claim confidence scoring with multi-signal aggregation

## [v0.12.4] — 2026-04-15

### Chore

- bump pyproject.toml version to 0.12.4

## [v0.12.3] — 2026-04-15

### Added

- claim decomposition with evidence taxonomy

## [v0.12.2] — 2026-04-15

### Added

- MinerU (magic-pdf) conversion backend

## [v0.12.1] — 2026-04-15

### Added

- cross-encoder passage reranking for chunk retrieval

## [v0.12.0] — 2026-04-15

### Added

- Tier 1 enhancements — FACT verification, export formats, query refinement, structured evidence, enhanced comparison

## [v0.11.0] — 2026-04-15

### Added

- Batch 3 (O-T) — LLM providers, screening judge, summarization, diversity, RACE scoring

## [v0.10.0] — 2026-04-14

### Added

- Batch 2 enhancements (I-N) — safety gate, BFS expansion, sanitization, depth gate, checkpoints, hybrid retrieval

## [v0.9.0] — 2026-04-14

### Added

- v0.9.0 — audit logging, backward-preference citation, Q2D query augmentation, FTS5 index search, bibliography extraction, tool integrity hashing

## [v0.8.1] — 2026-04-14

### Chore

- bump version to 0.8.1

## [v0.8.0] — 2026-04-14

### Added

- add P1-P3 quality improvements (v0.8.0)

## [v0.7.1] — 2026-04-14

### Added

- enhanced report template with confidence levels and structured agent output schemas

## [v0.7.0] — 2026-04-14

### Chore

- bump version to 0.7.0

## [v0.6.0] — 2026-04-14

### Chore

- bump version to 0.6.0

## [v0.5.0] — 2026-04-14

### Chore

- bump version to 0.5.0

## [v0.4.0] — 2026-04-08

### Added

- v0.4.0 — merge diverged fixes + new features

## [v0.3.0] — 2026-04-05

### Documentation

- update all documentation for v0.3.0 release

## [v0.2.0] — 2026-04-05

### Added

- multi-backend PDF conversion with registry pattern

## [v0.1.0] — 2026-04-05

### Chore

- prepare v0.1.0 release for PyPI
