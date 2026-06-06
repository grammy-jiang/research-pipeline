# Checklist: Expected Sections

The `design` mode architecture document must contain `## Contents`,
`## Update History`, and all 27 numbered sections in order.

- [ ] `## Contents` (links every numbered section)
- [ ] 1. Executive Architecture Summary (with §1.x Generation Metadata)
- [ ] `## Update History` (table, ≥ 1 row)
- [ ] 2. Source Blueprint Interpretation
- [ ] 3. Clarification Summary
- [ ] 4. Architecture Goals and Constraints
- [ ] 5. Solution Strategy
- [ ] 6. Traditional Software vs AI-Agent Boundary
- [ ] 7. Recommended Tech Stack
- [ ] 8. System Context View
- [ ] 9. Container / Runtime View
- [ ] 10. Component View
- [ ] 11. AI / Skill / MCP Architecture
- [ ] 12. Interface Contracts
- [ ] 13. Data Contracts and Schemas
- [ ] 14. State, Storage, and Data Lifecycle (canonical state model: lifecycle
      states vs operational condition flags vs audit events kept distinct)
- [ ] 15. Workflow / Sequence Views
- [ ] 16. Observability, Logging, Telemetry, and Audit
- [ ] 17. Security and Trust Boundaries
- [ ] 18. Failure Handling and Recovery
- [ ] 19. Testing and Evaluation Architecture
- [ ] 20. Deployment Architecture
- [ ] 21. Architecture Decision Records
- [ ] 22. Technical Risks and Trade-offs
- [ ] 23. Experience Architecture (consumes blueprint §9; surfaces, user-visible
      states mapped to §14, feedback, error/recovery, human-review flow,
      trust/transparency, UX handoff — architecture-level UX support, not UX design)
- [ ] 24. Recommended Next Stages and Downstream Handoffs (consumes blueprint
      §19 routing; tech-stack / ux-design / security-review / test-design
      handoffs; update/reconciliation triggers)
- [ ] 25. Open Questions
- [ ] 26. Architecture Quality-Gate Self-Check
- [ ] 27. Handoff Notes for Implementation Planning

For `stack` mode, see `templates/architecture_tech_stack_template.md` (a separate
19-section `<topic-slug>-architecture-tech-stack.md` document).
