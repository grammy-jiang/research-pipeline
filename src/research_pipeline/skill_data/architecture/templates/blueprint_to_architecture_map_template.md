# Blueprint-to-Architecture Traceability Map Template

Map every consumed blueprint section to the architecture section(s) it drives.
This is the contract that prevents reinventing the handoff each run.

```markdown
| Blueprint Section | Architecture Section(s) | How It Is Used |
|---|---|---|
| Executive Product Thesis | §1, §4 | System purpose, priorities, non-negotiables |
| Source Research Interpretation | §2, §22 | Evidence-backed architecture drivers |
| Target Users and System Actors | §8, §16, §17 | Actors, boundaries, approval actors |
| Product Goals and Non-Goals | §4, §5 | Goals → drivers; non-goals → constraints |
| Research-to-Product Translation Map | §5, §21 | Mechanisms needing technical decisions |
| Adopt / Adapt / Defer Decisions | §21, §22 | Seeds ADRs and deferred tech options |
| Core Product Capabilities | §9, §10 | Containers, components, responsibilities |
| Workflow Model | §15, §16 | Sequence diagrams, logs/traces, failure handling |
| Logical Architecture | §8, §9, §10 | Conceptual → architecture components (when justified) |
| Conceptual Information Model | §13, §14 | Schema objects, storage ownership, retention |
| Decision Policies | §12, §17, §18 | Policy modules, validation points, escalation |
| Risk / Governance / Safety Model | §17, §18, §22 | Trust boundaries, gates, mitigations |
| Evaluation Strategy | §19 | Test matrix, golden tests, AI evaluation |
| MVP Scope | §4, §9, §20 | Constrains architecture to MVP-0/MVP-1 |
| Roadmap | §20, §22 | MVP vs later extensibility |
| Open Questions | §3, §23 | What to ask, assume, or defer |
| Handoff Notes for Technical Design | all sections | Primary input for tech-stack selection |
| Traceability Appendix | §21, §24 | Decisions trace back to blueprint intent |
| Design Decision Register | §21 | Product-level decisions → architecture ADRs |
```

## Traceability quality gate

> Every major architecture decision must trace back to a blueprint section, a
> user clarification, an architecture rule-pack decision, or an explicit
> recorded architecture assumption. If a blueprint section is absent, write
> "not present in blueprint" rather than inventing a source.
