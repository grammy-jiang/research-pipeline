# Checklist: C4 Diagrams

- [ ] System Context view present (Mermaid).
- [ ] Container / Runtime view present (Mermaid).
- [ ] At least one Dynamic / sequence view of the main workflow present
      (Mermaid), including the AI→validation→commit path and the primary
      failure branch.
- [ ] Component view present **only** for the most complex container (named),
      or explicitly omitted with a reason.
- [ ] Deployment view present **only** when topology affects security, privacy,
      data locality, scaling, availability, or operations — not merely because
      an external LLM is used.
- [ ] No Code-level diagram at this stage.
- [ ] Each container/component has a responsibility and owner.
- [ ] The architecture does not convert every conceptual component into a
      separate container/service without rationale.
