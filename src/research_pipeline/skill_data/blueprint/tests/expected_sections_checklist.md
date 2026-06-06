# Expected Sections & Acceptance Checklist

Use this checklist as the pass/fail contract for a generated blueprint. It
combines the seven quality gates with the acceptance criteria.

## Structure

- [ ] `## Contents` section with working internal links.
- [ ] All 20 required sections present, in order:
  - [ ] 1. Executive Product Thesis
  - [ ] 2. Source Research Interpretation
  - [ ] 3. Target Users and System Actors
  - [ ] 4. Product Goals and Non-Goals
  - [ ] 5. Research-to-Product Translation Map
  - [ ] 6. Adopt / Adapt / Merge / Defer / Reject Decisions
  - [ ] 7. Core Product Capabilities
  - [ ] 8. Workflow Model
  - [ ] 9. Product Experience Direction
  - [ ] 10. Logical Architecture
  - [ ] 11. Conceptual Information Model
  - [ ] 12. Decision Policies
  - [ ] 13. Risk, Governance, and Safety Model
  - [ ] 14. Evaluation Strategy
  - [ ] 15. MVP Scope
  - [ ] 16. Roadmap and Future Extensions
  - [ ] 17. Open Questions and Validation Plan
  - [ ] 18. Handoff Notes for Technical Design
  - [ ] 19. Recommended Next Stages
  - [ ] 20. Traceability Appendix

## Content quality

- [ ] Product thesis is clear and specific (not a paper summary).
- [ ] Research mechanisms are translated into product primitives.
- [ ] Major ideas are classified ADOPT / ADAPT / MERGE / DEFER / REJECT.
- [ ] Each research-derived decision includes a research citation.
- [ ] Each non-research design decision includes an explicit rationale and
      is limited to connecting, operationalizing, or governing
      research-backed capabilities.
- [ ] Core capabilities are explicit (not vague).
- [ ] Workflows include trigger, inputs, decision gates, steps, outputs,
      failure modes, and success criteria.
- [ ] Main end-to-end workflow has a Mermaid diagram.
- [ ] Logical architecture is conceptual (not technical) with a Mermaid
      diagram.
- [ ] Complex / safety-critical / high-risk workflows have extra Mermaid
      diagrams.
- [ ] §9 Product Experience Direction captures UX **intent**: primary user,
      job-to-be-done, experience thesis, primary interaction mode (with MVP
      stage + rationale), trust/control/transparency needs, human-in-the-loop
      (where needed), failure/recovery expectations, and a UX→architecture
      handoff.
- [ ] §9 stays UX intent: no screen layout, wireframes, CSS/visual design,
      exact CLI syntax/flags, exact MCP/API schemas, copywriting, or
      implementation tasks. Compact (1–2 pages, tables) in `standard` output.
- [ ] Conceptual information objects defined with lifecycle states.
- [ ] Decision policies are explicit with defaults and escalation rules.
- [ ] Risks and governance controls are explicit and realistic.
- [ ] `ACADEMIC` gaps → validation requirements, not MVP features.
- [ ] `ENGINEERING` gaps → product requirements.
- [ ] Evaluation strategy has ≥1 scenario per capability.
- [ ] MVP scope separates a minimal Core Value Path from Safety and
      Evaluation baselines, with an explicit success definition.
- [ ] Future roadmap exists with deferred items justified.
- [ ] Open questions are not hidden.
- [ ] Handoff notes explicitly list what technical design must decide.
- [ ] §19 Recommended Next Stages has a Pipeline Complexity Assessment (seven
      0–3 dimensions + total /21 + workflow class) and a stage-recommendation
      table covering architecture-design, tech-stack-selection, ux-design,
      security-review, test-design, architecture-update, and
      architecture-reconciliation.
- [ ] Every stage uses only RUN / SKIP / DEFER / ASK_USER, with confidence and
      reason; `architecture-design` is RUN unless explicitly justified; and
      `architecture-update` / `architecture-reconciliation` default to DEFER.

## Integrity & discipline

- [ ] Metadata is copied, not invented: skill version from `manifest.json`
      or `unknown`; **pipeline runs integrated** and **gap-closure rounds**
      are separate fields (no run count relabelled as a round count).
- [ ] The thesis leads with the primary research-backed architecture; a
      conditional/secondary/escalation-only mechanism is not promoted to
      product identity.
- [ ] Every primary actor / MVP domain is named (or implied) by the
      thesis; high-stakes domains seen only as evidence are
      Secondary/Future, not primary.
- [ ] Each major claim is research-backed, engineering-extrapolation,
      product-design-decision, or relocated (speculative → Open Questions;
      unsupported → removed or marked design hypothesis).
- [ ] No citation cell is blank: gap-derived items cite
      `[Source Report: Research Gaps — <name>]`; blank only for an explicit
      internal design hypothesis.
- [ ] §15 splits the core path into MVP-0 (smallest demonstrable) and MVP-1
      (first usable); a large Phase-1 system is not labelled MVP-0.
- [ ] Release gates from MEDIUM/LOW-confidence mechanisms are justified
      (HIGH risk + no cheaper control + why-now) or downgraded.
- [ ] Runtime/architecture-leaning wording is rephrased or surfaced as a
      WARNING (not silently accepted); only forbidden tech/vendor choices
      hard-fail.
- [ ] `standard` output respects the length budget and stays scannable in
      one pass; large tables are moved to appendices rather than the main
      body, or the output switches to `detailed`.
- [ ] Contents lists every numbered section AND every appendix present
      (Appendix A always; Appendix B only when included).
- [ ] Safe wording rewrites flagged by the self-check are applied before
      delivery (self-repair); the self-check reflects the post-repair
      document, and remaining WARNINGs need human/downstream judgement.
- [ ] An `## Appendix A: Blueprint Quality-Gate Self-Check` is present;
      every WARNING has a required action and a blocks-technical-design
      verdict; any FAIL is resolved before delivery.
- [ ] The **Product Experience Gate** rows are present in the self-check
      (primary user, job-to-be-done, experience thesis, interaction mode,
      trust/control/transparency, human-in-the-loop, failure/recovery, UX
      handoff) per `references/product-experience-direction.md`.
- [ ] The **Adaptive Stage-Gate Recommendation Gate** rows are present in the
      self-check (section exists, controlled decisions, RUN evidence, SKIP
      reason, DEFER revisit trigger, ASK_USER missing-info, PED informs
      recommendations) per `references/adaptive-stage-gate-routing.md`.
- [ ] *(optional)* An `## Appendix B: Design Decision Register` is included
      for architecture handoff — reversibility + revisit trigger, not a
      duplicate of §6.

## Hard fails

- [ ] No tech stack selected (language, framework, database, cloud,
      vendor, UI library, deployment model).
- [ ] No code generated.
- [ ] No implementation tasks/tickets generated.
- [ ] Quality gates run and pass — or failures are surfaced to the user
      after at most 3 revision attempts.
