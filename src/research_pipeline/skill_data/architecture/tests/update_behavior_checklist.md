# Checklist: Update Behavior

Applies when an architecture document already exists for the topic.

- [ ] Existing `<topic-slug>-architecture-design.md` detected; its generation
      metadata and Update History were read.
- [ ] Source-blueprint hash/timestamp and architecture skill version compared.
- [ ] An update mode was selected: regenerate / patch / compare / adr-only /
      resume.
- [ ] Default behavior respected:
  - blueprint changed substantially → regenerate
  - only clarification answers changed → patch affected sections + ADRs
  - tech-stack decision changed → patch tech stack, container view, deployment,
    ADRs, handoff notes
  - security/privacy constraint changed → regenerate security, deployment, data
    lifecycle, observability, ADRs
- [ ] ADRs superseded, never silently overwritten; new ADRs link to superseded
      ones.
- [ ] `## Update History` row appended (prior rows preserved).
- [ ] In `compare` mode, a diff-style review is produced without rewriting the
      document.
