"""Tests for the bundled ``architecture`` skill.

The architecture skill is a pure prompt-driven transformation skill (no CLI/MCP
backend). These tests validate that the bundled skill files are well-formed,
discoverable by ``setup``, and stay faithful to the design contract
(``docs/architecture-skill-mode-split-improvement-plan.md`` and
``docs/architecture-remaining-modes-implementation-plan.md``): standard-only
SKILL.md frontmatter, the five implemented internal modes (``design`` / ``stack``
/ ``review`` / ``update`` / ``reconcile``), the 25-task ``design`` manifest, the
6-task ``stack`` manifest, the 3-task ``review`` / ``update`` / ``reconcile``
graphs sharing an artifact resolver, the 27-section ``design`` output template
and the stack/review/update/reconcile templates, the
prompts/templates/references/rule-packs, and complete worked examples that
themselves satisfy the structural gates (including the no-silent-mutation rule).
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _skill_root() -> Path:
    # Resolve via filesystem; skill files are bundled with the package.
    import research_pipeline

    return Path(research_pipeline.__file__).parent / "skill_data" / "architecture"


REQUIRED_PROMPTS = [
    "00_mode_resolver.md",
    "01_resolve_blueprint_input.md",
    "02_detect_existing_architecture.md",
    "03_prepare_blueprint_context.md",
    "04_parse_blueprint.md",
    "05_blueprint_to_architecture_traceability.md",
    "06_architecture_clarification.md",
    "07_solution_strategy.md",
    "08_architecture_goals_constraints.md",
    "09_provisional_tech_stack_selection.md",
    "10_traditional_vs_ai_boundary.md",
    "11_skill_mcp_decision.md",
    "12_tech_stack_boundary_coherence.md",
    "13_c4_views.md",
    "14_interface_contracts.md",
    "15_data_contracts.md",
    "16_security_trust_boundaries.md",
    "17_observability_audit.md",
    "18_failure_handling_recovery.md",
    "19_testing_evaluation_architecture.md",
    "20_generate_or_update_adrs.md",
    "21_rule_pack_review.md",
    "22_architecture_draft.md",
    "23_quality_gate_self_check.md",
    "24_final_architecture_document.md",
]

# The 6 ``stack`` mode prompts.
REQUIRED_STACK_PROMPTS = [
    "stack_01_resolve_inputs.md",
    "stack_02_technology_decision_drivers.md",
    "stack_03_stack_selection.md",
    "stack_04_architecture_impact_and_update.md",
    "stack_05_stack_quality_gate.md",
    "stack_06_final_stack_document.md",
]

REQUIRED_TEMPLATES = [
    "architecture_design_template.md",
    "architecture_tech_stack_template.md",
    "architecture_review_template.md",
    "architecture_update_template.md",
    "architecture_reconciliation_template.md",
    "contents_template.md",
    "generation_metadata_template.md",
    "update_history_template.md",
    "adr_template.md",
    "interface_contract_template.md",
    "observability_plan_template.md",
    "security_trust_boundary_template.md",
    "ai_responsibility_matrix_template.md",
    "tech_stack_decision_table.md",
    "blueprint_to_architecture_map_template.md",
]

REQUIRED_REFERENCES = [
    "input-discovery.md",
    "c4_model_summary.md",
    "adr_guidance.md",
    "security_trust_model_guide.md",
    "observability_event_catalogue.md",
    "mcp_adoption_guide.md",
    "mode-selection-guide.md",
    "experience-architecture-guide.md",
    "next-stages-and-handoffs-guide.md",
    "tech-stack-selection-guide.md",
    "artifact-discovery-guide.md",
    "architecture-review-guide.md",
    "architecture-update-guide.md",
    "architecture-reconciliation-guide.md",
]

REQUIRED_RULE_PACKS = [
    "boundary_rules.md",
    "data_rules.md",
    "interface_rules.md",
    "reliability_rules.md",
    "ai_boundary_rules.md",
    "observability_rules.md",
    "security_rules.md",
]

REQUIRED_EXAMPLES = [
    "translation_blueprint_excerpt.md",
    "translation_architecture_example.md",
    "translation_tech_stack_example.md",
    "translation_architecture_review_example.md",
    "translation_architecture_update_example.md",
    "translation_architecture_reconciliation_example.md",
]

# The shared resolver + the 6 review/update/reconcile mode prompts.
REQUIRED_MODE_PROMPTS = [
    "resolve_artifacts.md",
    "review_02_assess.md",
    "review_03_final_document.md",
    "update_02_apply_decisions.md",
    "update_03_final_document.md",
    "reconcile_02_detect_conflicts.md",
    "reconcile_03_final_document.md",
]

# Task ids for the three remaining-mode graphs.
REVIEW_TASK_IDS = [
    "review_resolve_artifacts",
    "review_assess",
    "review_final_document",
]
UPDATE_TASK_IDS = [
    "update_resolve_artifacts",
    "update_apply_decisions",
    "update_final_document",
]
RECONCILE_TASK_IDS = [
    "reconcile_resolve_artifacts",
    "reconcile_detect_conflicts",
    "reconcile_final_document",
]

REQUIRED_TEST_CHECKLISTS = [
    "expected_sections_checklist.md",
    "skill_trigger_checklist.md",
    "manifest_coverage_checklist.md",
    "traceability_map_checklist.md",
    "forbidden_output_checklist.md",
    "c4_diagram_checklist.md",
    "interface_contract_checklist.md",
    "observability_checklist.md",
    "security_checklist.md",
    "update_behavior_checklist.md",
]

# The 27 required ``design`` output sections, by heading text.
REQUIRED_SECTIONS = [
    "Executive Architecture Summary",
    "Source Blueprint Interpretation",
    "Clarification Summary",
    "Architecture Goals and Constraints",
    "Solution Strategy",
    "Traditional Software vs AI-Agent Boundary",
    "Recommended Tech Stack",
    "System Context View",
    "Container / Runtime View",
    "Component View",
    "AI / Skill / MCP Architecture",
    "Interface Contracts",
    "Data Contracts and Schemas",
    "State, Storage, and Data Lifecycle",
    "Workflow / Sequence Views",
    "Observability, Logging, Telemetry, and Audit",
    "Security and Trust Boundaries",
    "Failure Handling and Recovery",
    "Testing and Evaluation Architecture",
    "Deployment Architecture",
    "Architecture Decision Records",
    "Technical Risks and Trade-offs",
    "Experience Architecture",
    "Recommended Next Stages and Downstream Handoffs",
    "Open Questions",
    "Architecture Quality-Gate Self-Check",
    "Handoff Notes for Implementation Planning",
]

# The 19 required ``stack`` output sections, by heading text.
REQUIRED_STACK_SECTIONS = [
    "Generation Metadata",
    "Source Architecture Summary",
    "Technology Decision Drivers",
    "Stack Selection Summary",
    "Runtime and Language Selection",
    "Application Framework Selection",
    "Storage and Data Layer Selection",
    "Queue / Background Job Selection",
    "LLM / AI / Agent Stack Selection",
    "MCP / AI-Skill Integration Stack",
    "Observability and Audit Stack",
    "Testing Stack",
    "Deployment and Packaging Stack",
    "Alternatives Considered",
    "Risk and Reversibility",
    "ADR Candidates",
    "Architecture Impact Notes",
    "Architecture Update Required?",
    "Tech Stack Quality-Gate Self-Check",
]

# The 25 ``design`` manifest task ids (``tasks``).
REQUIRED_TASK_IDS = [
    "mode_resolver",
    "resolve_blueprint_input",
    "detect_existing_architecture",
    "prepare_blueprint_context",
    "parse_blueprint",
    "traceability_map",
    "clarify_architecture",
    "solution_strategy",
    "goals_constraints",
    "provisional_tech_stack",
    "traditional_vs_ai_boundary",
    "skill_mcp_decision",
    "tech_stack_boundary_coherence",
    "c4_views",
    "interfaces",
    "data_contracts",
    "security",
    "observability",
    "failure_handling",
    "testing_evaluation",
    "adrs",
    "rule_pack_review",
    "architecture_draft",
    "quality_gate_self_check",
    "final_architecture_document",
]

# The 6 ``stack`` manifest task ids (``stack_tasks``).
REQUIRED_STACK_TASK_IDS = [
    "stack_resolve_inputs",
    "stack_decision_drivers",
    "stack_selection",
    "stack_architecture_impact",
    "stack_quality_gate",
    "stack_final_document",
]


# --- SKILL.md ---


def test_skill_md_exists_and_has_frontmatter() -> None:
    skill = _skill_root() / "SKILL.md"
    assert skill.exists()
    text = skill.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    head, _, _ = text[4:].partition("\n---\n")
    for key in ("name:", "description:", "license:"):
        assert key in head, f"SKILL.md frontmatter missing: {key}"


def test_skill_md_frontmatter_is_standard_only() -> None:
    """Only standard fields are allowed in frontmatter.

    Non-standard top-level keys (``version``, ``compatibility``, ``aliases``,
    ``metadata``) can prevent skill loading and must stay out of frontmatter —
    they belong in ``manifest.json`` or the SKILL.md body.
    """
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    head, _, _ = text[4:].partition("\n---\n")
    # Top-level YAML keys are non-indented ``key:`` lines.
    top_level_keys = {
        m.group(1)
        for line in head.splitlines()
        if (m := re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):", line))
    }
    assert top_level_keys <= {"name", "description", "license"}, (
        f"non-standard frontmatter keys present: "
        f"{top_level_keys - {'name', 'description', 'license'}}"
    )


def test_skill_md_declares_correct_name() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    assert "name: architecture" in text


def test_skill_md_has_trigger_phrases() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8").lower()
    for phrase in (
        "convert this blueprint to architecture",
        "design the technical architecture",
        "choose the tech stack",
        "create c4 diagrams",
        "define module boundaries",
    ):
        assert phrase in text, f"SKILL.md missing trigger phrase: {phrase}"


def test_skill_md_names_aliases() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8").lower()
    for alias in (
        "blueprint-to-architecture",
        "technical-architecture",
        "architecture-design",
        "system-architecture",
        "tech-stack-design",
    ):
        assert alias in text, f"SKILL.md missing alias: {alias}"


def test_skill_md_has_anti_patterns_and_handoff() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "anti-pattern" in lowered
    # Upstream and downstream handoffs must be explicit.
    assert "blueprint" in lowered
    assert "implementation" in lowered
    # Core AI-boundary rule must be present.
    assert "deterministic" in lowered


def test_skill_md_description_fits_copilot_limit() -> None:
    """The folded description must stay under GitHub Copilot's 1024-char limit.

    Copilot silently refuses to load a skill whose ``description`` exceeds
    1024 characters, so this guards against the field creeping back over the
    limit. The folded value joins the YAML ``>`` block's lines with spaces.
    """
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    head, _, _ = text[4:].partition("\n---\n")
    m = re.search(
        r"(?ms)^description:\s*>\s*\n(.*?)(?=^[A-Za-z_][A-Za-z0-9_-]*:)", head
    )
    assert m, "description must be a folded (>) YAML block"
    folded = " ".join(line.strip() for line in m.group(1).strip().splitlines())
    assert len(folded) < 1024, f"description is {len(folded)} chars (limit 1024)"


# --- prompts / templates / references / rule-packs / examples / checklists ---


def test_all_prompt_files_exist_and_nonempty() -> None:
    prompts_dir = _skill_root() / "prompts"
    assert prompts_dir.is_dir()
    for name in REQUIRED_PROMPTS:
        path = prompts_dir / name
        assert path.exists(), f"Missing prompt: {name}"
        assert path.read_text(encoding="utf-8").strip(), f"Empty prompt: {name}"


def test_all_template_files_exist_and_nonempty() -> None:
    templates_dir = _skill_root() / "templates"
    assert templates_dir.is_dir()
    for name in REQUIRED_TEMPLATES:
        path = templates_dir / name
        assert path.exists(), f"Missing template: {name}"
        assert path.read_text(encoding="utf-8").strip(), f"Empty template: {name}"


def test_all_reference_files_exist_and_nonempty() -> None:
    refs_dir = _skill_root() / "references"
    assert refs_dir.is_dir()
    for name in REQUIRED_REFERENCES:
        path = refs_dir / name
        assert path.exists(), f"Missing reference: {name}"
        assert path.read_text(encoding="utf-8").strip(), f"Empty reference: {name}"


def test_all_rule_pack_files_exist_and_nonempty() -> None:
    packs_dir = _skill_root() / "rule-packs"
    assert packs_dir.is_dir()
    for name in REQUIRED_RULE_PACKS:
        path = packs_dir / name
        assert path.exists(), f"Missing rule pack: {name}"
        assert path.read_text(encoding="utf-8").strip(), f"Empty rule pack: {name}"


def test_all_example_files_exist_and_nonempty() -> None:
    ex_dir = _skill_root() / "examples"
    assert ex_dir.is_dir()
    for name in REQUIRED_EXAMPLES:
        path = ex_dir / name
        assert path.exists(), f"Missing example: {name}"
        assert path.read_text(encoding="utf-8").strip(), f"Empty example: {name}"


def test_all_test_checklist_files_exist_and_nonempty() -> None:
    tests_dir = _skill_root() / "tests"
    assert tests_dir.is_dir()
    for name in REQUIRED_TEST_CHECKLISTS:
        path = tests_dir / name
        assert path.exists(), f"Missing checklist: {name}"
        assert path.read_text(encoding="utf-8").strip(), f"Empty checklist: {name}"


# --- manifest.json ---


def test_manifest_parses_and_has_expected_shape() -> None:
    manifest = _skill_root() / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["workflow_id"] == "architecture"
    assert data["version"], "manifest must declare a version"
    task_ids = {task["id"] for task in data["tasks"]}
    for required in REQUIRED_TASK_IDS:
        assert required in task_ids, f"manifest missing task: {required}"
    # No collapsing: all 24 passes must be present.
    assert len(data["tasks"]) == len(REQUIRED_TASK_IDS)
    for gate in (
        "rule_pack_review",
        "quality_gate_self_check",
        "final_architecture_document",
    ):
        assert gate in data["mandatory_gates"], f"missing mandatory gate: {gate}"


def test_manifest_executors_reference_real_prompt_files() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    for task in data["tasks"]:
        executor = task["executor"]
        assert isinstance(executor, str) and executor.startswith("prompts/")
        assert (_skill_root() / executor).exists(), f"missing executor: {executor}"


def test_manifest_detect_existing_architecture_is_consumed_downstream() -> None:
    """Quality gate: detect_existing_architecture output must be consumed.

    The design (v0.3 fix) wires it into prepare context, solution strategy,
    ADRs, draft, and final document.
    """
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    consumers = {
        task["id"]
        for task in data["tasks"]
        if "detect_existing_architecture" in task.get("depends_on", [])
    }
    for expected in (
        "prepare_blueprint_context",
        "solution_strategy",
        "adrs",
        "architecture_draft",
        "final_architecture_document",
    ):
        assert expected in consumers, (
            f"detect_existing_architecture not consumed by {expected}"
        )


def test_manifest_quality_gate_self_check_is_its_own_task() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    task = next(t for t in data["tasks"] if t["id"] == "quality_gate_self_check")
    assert task["executor"] == "prompts/23_quality_gate_self_check.md"


# --- output template ---


def test_output_template_has_all_27_sections_and_contents() -> None:
    template = (
        _skill_root() / "templates" / "architecture_design_template.md"
    ).read_text(encoding="utf-8")
    assert "## Contents" in template
    assert "## Update History" in template
    for section in REQUIRED_SECTIONS:
        assert section in template, f"template missing section: {section}"


def test_output_template_has_generation_metadata() -> None:
    template = (
        _skill_root() / "templates" / "architecture_design_template.md"
    ).read_text(encoding="utf-8")
    assert "Generation Metadata" in template
    assert "Architecture skill version" in template


# --- worked example must itself satisfy the structural gates ---


def test_example_output_is_complete() -> None:
    example = (
        _skill_root() / "examples" / "translation_architecture_example.md"
    ).read_text(encoding="utf-8")
    assert "## Contents" in example
    assert "## Update History" in example
    for section in REQUIRED_SECTIONS:
        assert section in example, f"example missing section: {section}"
    # Required C4 views (context, container) + a dynamic view → at least 3
    # Mermaid blocks.
    assert example.count("```mermaid") >= 3
    # Carries a Traditional-vs-AI matrix and a tech-stack table.
    assert "Traditional Software" in example and "AI / LLM" in example
    # ADRs for major decisions.
    assert "ADR-0001" in example


def test_example_defers_or_justifies_mcp() -> None:
    example = (
        (_skill_root() / "examples" / "translation_architecture_example.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "mcp" in example
    # The example models a defer decision recorded as an ADR.
    assert "defer" in example


def test_example_avoids_implementation_leakage() -> None:
    """The architecture stage selects a stack but must not cross into
    implementation (no DDL / migrations)."""
    example = (
        (_skill_root() / "examples" / "translation_architecture_example.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    for term in ("create table", "alter table", "drop table"):
        assert term not in example, f"example leaks implementation DDL: {term!r}"


# --- discoverability by setup ---


def test_skill_is_discoverable_by_setup() -> None:
    """``setup`` auto-discovers any skill_data/*/SKILL.md; confirm architecture."""
    from research_pipeline.cli.cmd_setup import _find_skill_sources

    sources = _find_skill_sources()
    names = {p.name for p in sources}
    assert "architecture" in names, "setup did not discover the architecture skill"


# --- v0.2.0 quality-control pass (metadata consistency, hybrid decision
# review, technology validity, probe availability, impl boundary) ---

# Gate names that must appear in the §24 self-check (template + example) and in
# the quality-gate self-check prompt.
V020_GATE_NAMES = [
    "Metadata consistency",
    "Hybrid decision review",
    "Technology-specific validity",
    "Probe/evaluator availability",
    "Architecture-vs-implementation boundary",
]


def test_manifest_skill_version_bumped() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    assert data["version"] == "0.7.0"


def test_self_check_prompt_covers_new_gates() -> None:
    text = (
        (_skill_root() / "prompts" / "23_quality_gate_self_check.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    for needle in (
        "metadata consistency",
        "hybrid decision review",
        "technology-specific validity",
        "availability",
        "architecture-vs-implementation boundary",
    ):
        assert needle in text, f"self-check prompt missing: {needle}"


def test_clarification_prompt_has_decision_review_classification() -> None:
    text = (
        (_skill_root() / "prompts" / "06_architecture_clarification.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "review requirement" in text
    assert "requires user review before implementation planning" in text


def test_failure_prompt_has_probe_availability_policy() -> None:
    text = (
        (_skill_root() / "prompts" / "18_failure_handling_recovery.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "availability policy" in text
    assert "auto-accept" in text


def test_tech_stack_prompt_has_validity_rule() -> None:
    text = (
        (_skill_root() / "prompts" / "09_provisional_tech_stack_selection.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "tamper-evident" in text or "application-enforced" in text


def test_template_has_decision_source_and_review_columns() -> None:
    template = (
        _skill_root() / "templates" / "architecture_design_template.md"
    ).read_text(encoding="utf-8")
    assert "| Source | Decision Evidence | Review Requirement |" in template


def test_template_self_check_lists_new_gates() -> None:
    template = (
        _skill_root() / "templates" / "architecture_design_template.md"
    ).read_text(encoding="utf-8")
    for gate in V020_GATE_NAMES:
        assert gate in template, f"template §24 missing gate: {gate}"


def test_template_labels_proposed_namespaces() -> None:
    template = (
        _skill_root() / "templates" / "architecture_design_template.md"
    ).read_text(encoding="utf-8")
    assert "proposed module namespaces" in template


def test_example_models_new_gates_and_columns() -> None:
    example = (
        _skill_root() / "examples" / "translation_architecture_example.md"
    ).read_text(encoding="utf-8")
    assert "| Source | Decision Evidence | Review Requirement |" in example
    for gate in V020_GATE_NAMES:
        assert gate in example, f"example §24 missing gate: {gate}"
    assert "proposed module namespaces" in example
    # The example must model honest tech claims.
    assert "tamper-evident" in example


# --- v0.3.0 hardening pass (data egress, residual-claim scan, self-check
# skepticism, state-semantics consistency, output budget) ---

# Gate names that must appear in the §24 self-check (template + example) and in
# the quality-gate self-check prompt.
V030_GATE_NAMES = [
    "Residual invalid-claim scan",
    "Data egress / external model use",
    "State-semantics consistency",
    "Standard-vs-detailed budget",
]


def test_self_check_prompt_covers_v030_gates() -> None:
    text = (
        (_skill_root() / "prompts" / "23_quality_gate_self_check.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    for needle in (
        "residual invalid-claim scan",
        "data egress",
        "state-semantics consistency",
        "standard-vs-detailed budget",
        "pass with warning",
    ):
        assert needle in text, f"self-check prompt missing: {needle}"


def test_clarification_prompt_has_data_egress_decision() -> None:
    text = (
        (_skill_root() / "prompts" / "06_architecture_clarification.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "data egress" in text
    for cls in ("external_allowed", "local_only", "unknown_requires_user_review"):
        assert cls in text, f"clarification prompt missing egress class: {cls}"


def test_data_contracts_prompt_has_canonical_state_model() -> None:
    text = (_skill_root() / "prompts" / "15_data_contracts.md").read_text(
        encoding="utf-8"
    )
    assert "canonical state model" in text.lower()
    for cat in ("Lifecycle states", "Operational condition flags", "Audit events"):
        assert cat in text, f"data-contracts prompt missing state category: {cat}"


def test_draft_prompt_has_output_detail_budget() -> None:
    text = (_skill_root() / "prompts" / "22_architecture_draft.md").read_text(
        encoding="utf-8"
    )
    assert "output detail budget" in text.lower()
    assert "appendices" in text.lower()


def test_template_has_data_egress_and_state_model() -> None:
    template = (
        _skill_root() / "templates" / "architecture_design_template.md"
    ).read_text(encoding="utf-8")
    assert "Data egress" in template
    assert "external_allowed" in template
    for cat in ("Lifecycle states", "Operational condition flags", "Audit events"):
        assert cat in template, f"template §14 missing state category: {cat}"


def test_template_self_check_lists_v030_gates() -> None:
    template = (
        _skill_root() / "templates" / "architecture_design_template.md"
    ).read_text(encoding="utf-8")
    for gate in V030_GATE_NAMES:
        assert gate in template, f"template §24 missing gate: {gate}"
    # WARNING vocabulary standardized (no leftover WARN/ slashes legend).
    assert "PASS / WARNING / FAIL" in template


def test_example_models_v030_gates() -> None:
    example = (
        _skill_root() / "examples" / "translation_architecture_example.md"
    ).read_text(encoding="utf-8")
    for gate in V030_GATE_NAMES:
        assert gate in example, f"example §24 missing gate: {gate}"
    assert "external_allowed" in example
    for cat in ("Lifecycle states", "Operational condition flags", "Audit events"):
        assert cat in example, f"example §14 missing state category: {cat}"
    assert "0.6.0" in example


# --- v0.4.0 hardening pass (verification-table security gates, data-egress
# table, residual-scan strengthening) ---


def test_security_prompt_requires_verification_table_and_egress_table() -> None:
    text = (
        (_skill_root() / "prompts" / "16_security_trust_boundaries.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "verification table" in text
    assert "data egress / external model use" in text
    # Must steer away from checkbox gates and borrowed grant wording.
    assert "checkbox" in text
    assert "blocks release?" in text


def test_self_check_prompt_has_security_gate_format_gate() -> None:
    text = (
        (_skill_root() / "prompts" / "23_quality_gate_self_check.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "security gate verification format" in text
    assert "verification method" in text


def test_security_template_has_verification_table_and_egress() -> None:
    text = (
        _skill_root() / "templates" / "security_trust_boundary_template.md"
    ).read_text(encoding="utf-8")
    assert "Required Implementation Evidence" in text
    assert "Verification Method" in text
    assert "Data Egress / External Model Use" in text


def test_template_self_check_has_security_gate_format_row() -> None:
    template = (
        _skill_root() / "templates" / "architecture_design_template.md"
    ).read_text(encoding="utf-8")
    assert "Security gate verification format" in template


def test_example_security_gates_are_verification_table_not_checkboxes() -> None:
    example = (
        _skill_root() / "examples" / "translation_architecture_example.md"
    ).read_text(encoding="utf-8")
    # §17.12 verification-table columns present.
    assert "Required Implementation Evidence" in example
    assert "Verification Method" in example
    # §17.9 data-egress table present.
    assert "Data Egress / External Model Use" in example
    # The worked example must contain NO ambiguous unchecked checkboxes.
    assert "- [ ]" not in example, "example must not use unchecked checkboxes"
    assert "Security gate verification format" in example


# --- v0.5.0 hardening pass (decision provenance, log-egress default, warning
# surfacing, build-sequencing cap) ---

V050_GATE_NAMES = [
    "Decision evidence / provenance",
    "Raw source-content logging policy",
    "Warning surfacing",
    "Architecture-stage sequencing cap",
]


def test_clarification_prompt_has_decision_evidence() -> None:
    text = (
        (_skill_root() / "prompts" / "06_architecture_clarification.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "decision evidence" in text
    for v in ("confirmed_in_interactive_answer", "architecture_assumption"):
        assert v in text, f"clarification prompt missing evidence value: {v}"


def test_security_prompt_forbids_raw_source_in_logs() -> None:
    text = (
        (_skill_root() / "prompts" / "16_security_trust_boundaries.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "forbidden in logs by default" in text
    assert "log-snapshot test" in text


def test_draft_prompt_surfaces_warnings_and_has_budget_targets() -> None:
    text = (_skill_root() / "prompts" / "22_architecture_draft.md").read_text(
        encoding="utf-8"
    )
    assert "Surface warnings" in text
    assert "Architecture Warnings Requiring Attention" in text
    # Concrete standard-budget targets.
    assert "≤ 1 page" in text or "<= 1 page" in text


def test_final_prompt_caps_build_sequencing() -> None:
    text = (_skill_root() / "prompts" / "24_final_architecture_document.md").read_text(
        encoding="utf-8"
    )
    assert "Cap build sequencing" in text
    assert "five" in text.lower()


def test_self_check_prompt_covers_v050_gates() -> None:
    text = (
        (_skill_root() / "prompts" / "23_quality_gate_self_check.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    for needle in (
        "decision evidence / provenance",
        "raw source-content logging policy",
        "warning surfacing",
        "architecture-stage sequencing cap",
    ):
        assert needle in text, f"self-check prompt missing: {needle}"


def test_template_has_warnings_section_and_v050_gates() -> None:
    template = (
        _skill_root() / "templates" / "architecture_design_template.md"
    ).read_text(encoding="utf-8")
    assert "Architecture Warnings Requiring Attention" in template
    assert "| Source | Decision Evidence |" in template
    for gate in V050_GATE_NAMES:
        assert gate in template, f"template §24 missing gate: {gate}"


def test_example_models_v050_pass_with_warning_and_provenance() -> None:
    example = (
        _skill_root() / "examples" / "translation_architecture_example.md"
    ).read_text(encoding="utf-8")
    # Provenance column + values.
    assert "| Source | Decision Evidence |" in example
    assert "architecture_assumption" in example
    # Models the PASS-with-warning tier and surfaces it.
    assert "WARNING" in example
    assert "Architecture Warnings Requiring Attention" in example
    for gate in V050_GATE_NAMES:
        assert gate in example, f"example §24 missing gate: {gate}"
    # Raw-source-content logging gate present in §17.12.
    assert "Raw source content never written to logs" in example


# --- v0.6.0 mode-split pass (design vs stack modes; Experience Architecture;
# Recommended Next Stages and Downstream Handoffs; provisional-tech discipline) ---

# Design-mode §26 self-check gate names added by the mode split.
V060_GATE_NAMES = [
    "Product Experience Direction consumed",
    "Experience Architecture produced",
    "Recommended Next Stages consumed",
    "Tech stack provisional when stack mode recommended",
    "Downstream handoffs present",
]


def test_skill_md_documents_modes() -> None:
    """SKILL.md must document the internal modes and the mode resolver."""
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "## Modes" in text
    for mode in ("design", "stack", "update", "review", "reconcile"):
        assert mode in lowered, f"SKILL.md missing mode: {mode}"
    # The two stack/design outputs are both documented.
    assert "-architecture-tech-stack.md" in text
    assert "-architecture-design.md" in text
    assert "00_mode_resolver.md" in text


def test_mode_resolver_prompt_exists_and_routes() -> None:
    text = (_skill_root() / "prompts" / "00_mode_resolver.md").read_text(
        encoding="utf-8"
    )
    lowered = text.lower()
    for mode in ("design", "stack", "update", "review", "reconcile"):
        assert mode in lowered, f"mode resolver missing mode: {mode}"
    # Routes to the two implemented task graphs.
    assert "stack_tasks" in text
    assert "mode_resolution.json" in text


def test_all_stack_prompt_files_exist_and_nonempty() -> None:
    prompts_dir = _skill_root() / "prompts"
    for name in REQUIRED_STACK_PROMPTS:
        path = prompts_dir / name
        assert path.exists(), f"Missing stack prompt: {name}"
        assert path.read_text(encoding="utf-8").strip(), f"Empty stack prompt: {name}"


def test_manifest_has_mode_resolver_entry_task() -> None:
    """mode_resolver is the entry task; resolve_blueprint_input depends on it."""
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    tasks = {t["id"]: t for t in data["tasks"]}
    assert "mode_resolver" in tasks
    assert tasks["mode_resolver"]["depends_on"] == []
    assert "mode_resolver" in tasks["resolve_blueprint_input"]["depends_on"]
    # The design graph keeps all 25 tasks (no collapsing).
    assert len(data["tasks"]) == len(REQUIRED_TASK_IDS)


def test_manifest_has_stack_task_graph() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    assert "stack_tasks" in data, "manifest missing stack_tasks graph"
    stack_ids = {t["id"] for t in data["stack_tasks"]}
    for required in REQUIRED_STACK_TASK_IDS:
        assert required in stack_ids, f"stack_tasks missing: {required}"
    assert len(data["stack_tasks"]) == len(REQUIRED_STACK_TASK_IDS)
    for gate in ("stack_quality_gate", "stack_final_document"):
        assert gate in data["stack_mandatory_gates"], (
            f"missing stack mandatory gate: {gate}"
        )


def test_manifest_stack_executors_reference_real_prompt_files() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    for task in data["stack_tasks"]:
        executor = task["executor"]
        assert isinstance(executor, str) and executor.startswith("prompts/")
        assert (_skill_root() / executor).exists(), f"missing executor: {executor}"


def test_stack_template_has_all_19_sections_and_update_verdict() -> None:
    template = (
        _skill_root() / "templates" / "architecture_tech_stack_template.md"
    ).read_text(encoding="utf-8")
    assert "## Contents" in template
    assert "## Update History" in template
    for section in REQUIRED_STACK_SECTIONS:
        assert section in template, f"stack template missing section: {section}"
    # The decision table + the link to update mode.
    assert "Selected Technology" in template and "Reversibility" in template
    assert "Architecture Update Required?" in template


def test_design_template_consumes_blueprint_9_and_19() -> None:
    template = (
        _skill_root() / "templates" / "architecture_design_template.md"
    ).read_text(encoding="utf-8")
    # Experience Architecture (§9 consumption) with its key sub-structures.
    assert "## 23. Experience Architecture" in template
    assert "Interaction Surface Matrix" in template
    assert "User-Visible State Model" in template
    assert "UX Handoff to UX-Design" in template
    # Recommended Next Stages (§19 consumption) + the four downstream handoffs.
    assert "## 24. Recommended Next Stages and Downstream Handoffs" in template
    for handoff in (
        "Tech-Stack Selection Handoff",
        "UX-Design Handoff",
        "Security-Review Handoff",
        "Test-Design / E2E Handoff",
    ):
        assert handoff in template, f"design template missing handoff: {handoff}"
    # Provisional-tech discipline lives in §7.
    assert "Provisional Tech Assumptions" in template
    assert "Must Be Confirmed In Stack Mode?" in template


def test_template_self_check_lists_v060_gates() -> None:
    template = (
        _skill_root() / "templates" / "architecture_design_template.md"
    ).read_text(encoding="utf-8")
    for gate in V060_GATE_NAMES:
        assert gate in template, f"template §26 missing gate: {gate}"


def test_self_check_prompt_covers_v060_gates() -> None:
    text = (
        (_skill_root() / "prompts" / "23_quality_gate_self_check.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    for needle in (
        "product experience direction consumed",
        "recommended next stages consumed",
        "provisional-tech discipline",
        "downstream handoffs present",
    ):
        assert needle in text, f"self-check prompt missing v0.6.0 gate: {needle}"


def test_parse_prompt_consumes_blueprint_19() -> None:
    text = (_skill_root() / "prompts" / "04_parse_blueprint.md").read_text(
        encoding="utf-8"
    )
    assert "recommended_next_stages" in text
    assert "product_experience" in text


def test_provisional_tech_prompt_has_handoff_discipline() -> None:
    text = (
        (_skill_root() / "prompts" / "09_provisional_tech_stack_selection.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "provisional tech assumptions" in text
    assert "tech-stack selection handoff" in text
    assert "stack` mode" in text or "stack mode" in text


def test_example_models_v060_sections() -> None:
    example = (
        _skill_root() / "examples" / "translation_architecture_example.md"
    ).read_text(encoding="utf-8")
    assert "## 23. Experience Architecture" in example
    assert "## 24. Recommended Next Stages and Downstream Handoffs" in example
    assert "Provisional Tech Assumptions" in example
    assert "Tech-Stack Selection Handoff" in example
    for gate in V060_GATE_NAMES:
        assert gate in example, f"example §26 missing gate: {gate}"


def test_stack_example_is_complete_and_models_update_verdict() -> None:
    example = (
        _skill_root() / "examples" / "translation_tech_stack_example.md"
    ).read_text(encoding="utf-8")
    assert "## Contents" in example
    assert "## Update History" in example
    for section in REQUIRED_STACK_SECTIONS:
        assert section in example, f"stack example missing section: {section}"
    # Explicit Architecture Update Required verdict and honest tech claims.
    assert "Architecture Update Required?" in example
    assert "tamper-evident" in example
    # Honours the architecture's MCP deferral.
    assert "Defer" in example or "DEFER" in example
    # No implementation leakage (DDL / migrations).
    lowered = example.lower()
    for term in ("create table", "alter table", "drop table"):
        assert term not in lowered, f"stack example leaks DDL: {term!r}"


# --- v0.7.0 remaining-modes pass (review / update / reconcile + shared resolver;
# Phase-5 guard checks) ---


def test_mode_resolver_marks_all_modes_implemented() -> None:
    """The resolver no longer defers review/reconcile and routes to their graphs."""
    text = (_skill_root() / "prompts" / "00_mode_resolver.md").read_text(
        encoding="utf-8"
    )
    lowered = text.lower()
    assert "all five modes are implemented" in lowered
    assert "recognized — deferred" not in text
    assert "deferred" not in lowered
    # Routes to the three new graphs.
    for graph in ("review_tasks", "update_tasks", "reconcile_tasks"):
        assert graph in text, f"mode resolver missing routing to {graph}"
    # Bare architecture + existing architecture defaults to review (not update).
    assert "review" in lowered and "safest" in lowered


def test_shared_resolver_prompt_and_guide() -> None:
    prompt = (_skill_root() / "prompts" / "resolve_artifacts.md").read_text(
        encoding="utf-8"
    )
    guide = (_skill_root() / "references" / "artifact-discovery-guide.md").read_text(
        encoding="utf-8"
    )
    for needle in ("Resolved Input Artifacts", "topic", "ASK_USER"):
        assert needle in prompt, f"resolver prompt missing: {needle}"
    # Required-architecture fail-fast (message may be line-wrapped).
    assert "No architecture design document" in prompt
    assert "architecture --mode design" in prompt
    assert "Resolved Input Artifacts" in guide
    assert "Candidate scoring" in guide or "candidate" in guide.lower()


def test_all_mode_prompt_files_exist_and_nonempty() -> None:
    prompts_dir = _skill_root() / "prompts"
    for name in REQUIRED_MODE_PROMPTS:
        path = prompts_dir / name
        assert path.exists(), f"Missing mode prompt: {name}"
        assert path.read_text(encoding="utf-8").strip(), f"Empty mode prompt: {name}"


def test_manifest_has_review_update_reconcile_graphs() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    for graph, ids, gate in (
        ("review_tasks", REVIEW_TASK_IDS, "review_mandatory_gates"),
        ("update_tasks", UPDATE_TASK_IDS, "update_mandatory_gates"),
        ("reconcile_tasks", RECONCILE_TASK_IDS, "reconcile_mandatory_gates"),
    ):
        assert graph in data, f"manifest missing graph: {graph}"
        graph_ids = {t["id"] for t in data[graph]}
        for required in ids:
            assert required in graph_ids, f"{graph} missing task: {required}"
        assert len(data[graph]) == len(ids)
        assert gate in data, f"manifest missing {gate}"
        # The final document is the mandatory gate.
        assert any("final_document" in g for g in data[gate])
        # Each graph starts from the shared resolver and depends on mode_resolver.
        first = data[graph][0]
        assert first["executor"] == "prompts/resolve_artifacts.md"
        assert "mode_resolver" in first["depends_on"]


def test_manifest_mode_graph_executors_exist() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    for graph in ("review_tasks", "update_tasks", "reconcile_tasks"):
        for task in data[graph]:
            executor = task["executor"]
            assert executor.startswith("prompts/")
            assert (_skill_root() / executor).exists(), f"missing executor: {executor}"


def test_review_template_has_score_and_issue_classes() -> None:
    template = (
        _skill_root() / "templates" / "architecture_review_template.md"
    ).read_text(encoding="utf-8")
    assert "Score Breakdown" in template
    # Blocking / Warning / Polish are separate sections.
    for section in ("Blocking Issues", "Warnings", "Polish Improvements"):
        assert section in template, f"review template missing: {section}"
    assert "Review Quality-Gate Self-Check" in template
    # Non-mutating discipline stated.
    assert "non-mutating" in template.lower()


def test_update_template_has_accepted_decisions_and_no_overwrite() -> None:
    template = (
        _skill_root() / "templates" / "architecture_update_template.md"
    ).read_text(encoding="utf-8")
    assert "Accepted Decisions Applied" in template
    assert "Sections Requiring Update" in template
    assert "Update Quality-Gate Self-Check" in template
    # No default overwrite of the design document.
    lowered = template.lower()
    assert "does not overwrite" in lowered or "not overwrite" in lowered


def test_reconcile_template_has_findings_and_update_verdict() -> None:
    template = (
        _skill_root() / "templates" / "architecture_reconciliation_template.md"
    ).read_text(encoding="utf-8")
    assert "Conflict Summary" in template
    assert "Missing Architecture Support" in template
    assert "Architecture Update Required?" in template
    assert "Reconciliation Quality-Gate Self-Check" in template
    # No default patch.
    lowered = template.lower()
    assert "does not patch" in lowered or "not patch" in lowered


def test_no_mode_silently_mutates_architecture() -> None:
    """The no-silent-mutation rule is explicit in each mode's prompt/template."""
    review = (_skill_root() / "prompts" / "review_03_final_document.md").read_text(
        encoding="utf-8"
    )
    update = (_skill_root() / "prompts" / "update_03_final_document.md").read_text(
        encoding="utf-8"
    )
    reconcile = (
        _skill_root() / "prompts" / "reconcile_03_final_document.md"
    ).read_text(encoding="utf-8")
    assert "non-mutating" in review.lower() or "do not edit" in review.lower()
    assert "do not overwrite" in update.lower()
    assert "do not" in reconcile.lower() and "patch" in reconcile.lower()


def test_remaining_mode_examples_model_the_discipline() -> None:
    ex_dir = _skill_root() / "examples"
    review = (ex_dir / "translation_architecture_review_example.md").read_text(
        encoding="utf-8"
    )
    update = (ex_dir / "translation_architecture_update_example.md").read_text(
        encoding="utf-8"
    )
    reconcile = (
        ex_dir / "translation_architecture_reconciliation_example.md"
    ).read_text(encoding="utf-8")
    # Review: score breakdown + classified issues + Resolved Input Artifacts.
    assert "Score Breakdown" in review and "Resolved Input Artifacts" in review
    assert "Blocking Issues" in review and "non-mutating" in review.lower()
    # Update: accepted decisions, changed sections, no overwrite of the design.
    assert "Accepted Decisions Applied" in update
    assert "not" in update.lower() and "overwrit" in update.lower()
    assert "Resolved Input Artifacts" in update
    # Reconcile: findings traceable, Architecture Update Required verdict, no patch.
    assert "Architecture Update Required?" in reconcile
    assert "did **not** patch" in reconcile or "not patch" in reconcile.lower()
    assert "Resolved Input Artifacts" in reconcile
