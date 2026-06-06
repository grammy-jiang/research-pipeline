# Checklist: Manifest Coverage

`manifest.json` must expose the major workflow passes, not collapse them.

- [ ] `workflow_id` is `architecture`; `version` is present.
- [ ] `tasks` (the `design` graph) has all 25 tasks present: mode_resolver,
      resolve_blueprint_input, detect_existing_architecture,
      prepare_blueprint_context, parse_blueprint, traceability_map,
      clarify_architecture, solution_strategy, goals_constraints,
      provisional_tech_stack, traditional_vs_ai_boundary, skill_mcp_decision,
      tech_stack_boundary_coherence, c4_views, interfaces, data_contracts,
      security, observability, failure_handling, testing_evaluation, adrs,
      rule_pack_review, architecture_draft, quality_gate_self_check,
      final_architecture_document.
- [ ] `mode_resolver` is the entry task (depends_on `[]`) and
      `resolve_blueprint_input` depends on it.
- [ ] `stack_tasks` (the `stack` graph) has all 6 tasks present:
      stack_resolve_inputs, stack_decision_drivers, stack_selection,
      stack_architecture_impact, stack_quality_gate, stack_final_document.
- [ ] Every task `executor` (both graphs) points at a real `prompts/*.md` file.
- [ ] `quality_gate_self_check` exists as its own task (not folded into another).
- [ ] `detect_existing_architecture` is consumed downstream: it appears in the
      `depends_on` of prepare_blueprint_context, solution_strategy, adrs,
      architecture_draft, and final_architecture_document.
- [ ] `mandatory_gates` include rule_pack_review, quality_gate_self_check, and
      final_architecture_document; `stack_mandatory_gates` include
      stack_quality_gate and stack_final_document.

## Manifest execution model

The manifest is a task-graph specification. It is valid as **documentation** for
a prompt-driven agent and **ready** for a future runtime runner. For v0.1 the
skill must not assume a runner exists; if none does, follow the task order
manually.
