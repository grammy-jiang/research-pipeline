"""Tests for the bundled ``ux-design`` skill.

The ux-design skill is a pure prompt-driven transformation skill (no CLI/MCP
backend). These tests validate that the bundled skill files are well-formed,
discoverable by ``setup``, and stay faithful to the design contract
(``docs/ux-design-skill-implementation-plan.md``): standard-only SKILL.md
frontmatter, the 13-task manifest, the 22-section + Appendix-A output template,
the prompts/templates/references/checklists, and a complete worked example that
itself satisfies the structural gates (separating Skill Operator UX from Target
Software UX, with user stories, E2E scenario seeds, and architecture feedback).
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _skill_root() -> Path:
    # Resolve via filesystem; skill files are bundled with the package.
    import research_pipeline

    return Path(research_pipeline.__file__).parent / "skill_data" / "ux-design"


REQUIRED_PROMPTS = [
    "01_input_discovery.md",
    "02_architecture_parse.md",
    "03_blueprint_parse.md",
    "04_clarification.md",
    "05_skill_operator_ux.md",
    "06_target_software_ux.md",
    "07_user_stories.md",
    "08_surface_specific_ux.md",
    "09_error_recovery_ux.md",
    "10_e2e_scenario_seeds.md",
    "11_architecture_feedback.md",
    "12_final_document.md",
    "13_quality_gate_self_check.md",
]

REQUIRED_TEMPLATES = [
    "ux-design-template.md",
    "user-story-template.md",
    "e2e-scenario-template.md",
]

REQUIRED_REFERENCES = [
    "ux-question-bank.md",
    "surface-ux-guide.md",
    "e2e-scenario-seed-guide.md",
    "architecture-feedback-guide.md",
]

REQUIRED_EXAMPLES = [
    "translation-system-ux-design.md",
]

REQUIRED_TEST_CHECKLISTS = [
    "expected_sections_checklist.md",
    "forbidden_content_checklist.md",
]

# The 22 required output sections, by heading text.
REQUIRED_SECTIONS = [
    "Generation Metadata",
    "Source Architecture Interpretation",
    "Source Blueprint Interpretation",
    "UX Goals and Non-Goals",
    "Skill Operator UX",
    "Target Software UX",
    "Users, Roles, and Jobs-to-Be-Done",
    "UX Decision Summary",
    "UX Assumptions",
    "User Stories",
    "Core User Journeys",
    "Surface-Specific UX",
    "Human-in-the-Loop UX",
    "Trust, Control, and Transparency UX",
    "Error, Empty, Loading, Degraded, and Recovery States",
    "Notifications and Feedback",
    "Accessibility and Internationalization",
    "UX Observability",
    "Acceptance Criteria",
    "E2E Scenario Seeds",
    "Architecture Feedback / Required Architecture Updates",
    "Handoff Notes for Implementation Planning",
]

# The 13 manifest task ids.
REQUIRED_TASK_IDS = [
    "input_discovery",
    "architecture_parse",
    "blueprint_parse",
    "clarification",
    "skill_operator_ux",
    "target_software_ux",
    "user_stories",
    "surface_specific_ux",
    "error_recovery_ux",
    "e2e_scenario_seeds",
    "architecture_feedback",
    "final_document",
    "quality_gate_self_check",
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
    """Only standard fields are allowed in frontmatter (Copilot loadability)."""
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    head, _, _ = text[4:].partition("\n---\n")
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
    assert "name: ux-design" in text


def test_skill_md_has_trigger_phrases() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8").lower()
    for phrase in (
        "design the ux",
        "create ux design",
        "generate user stories",
        "design user journeys",
        "create interaction flows",
        "generate e2e scenario seeds",
        "turn architecture into ux design",
    ):
        assert phrase in text, f"SKILL.md missing trigger phrase: {phrase}"


def test_skill_md_has_negative_triggers_and_handoffs() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8").lower()
    assert "do not use for" in text
    # Negative triggers / boundary.
    for phrase in ("executable tests", "tech stack", "implementation plan"):
        assert phrase in text, f"SKILL.md missing negative trigger: {phrase}"
    # Upstream and downstream handoffs must be explicit.
    assert "architecture" in text and "implementation-plan" in text


def test_skill_md_separates_operator_and_target_ux() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    assert "Skill Operator UX" in text
    assert "Target Software UX" in text


def test_skill_md_description_fits_copilot_limit() -> None:
    """The folded description must stay under GitHub Copilot's 1024-char limit."""
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    head, _, _ = text[4:].partition("\n---\n")
    m = re.search(
        r"(?ms)^description:\s*>\s*\n(.*?)(?=^[A-Za-z_][A-Za-z0-9_-]*:)", head
    )
    assert m, "description must be a folded (>) YAML block"
    folded = " ".join(line.strip() for line in m.group(1).strip().splitlines())
    assert len(folded) < 1024, f"description is {len(folded)} chars (limit 1024)"


# --- prompts / templates / references / examples / checklists ---


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
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    assert data["workflow_id"] == "ux-design"
    assert data["version"] == "0.2.0"
    task_ids = {task["id"] for task in data["tasks"]}
    for required in REQUIRED_TASK_IDS:
        assert required in task_ids, f"manifest missing task: {required}"
    assert len(data["tasks"]) == len(REQUIRED_TASK_IDS)
    for gate in ("final_document", "quality_gate_self_check"):
        assert gate in data["mandatory_gates"], f"missing mandatory gate: {gate}"


def test_manifest_executors_reference_real_prompt_files() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    for task in data["tasks"]:
        executor = task["executor"]
        assert isinstance(executor, str) and executor.startswith("prompts/")
        assert (_skill_root() / executor).exists(), f"missing executor: {executor}"


def test_manifest_final_document_consumes_architecture_feedback() -> None:
    """Architecture feedback (§21) must flow into the final document."""
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    final = next(t for t in data["tasks"] if t["id"] == "final_document")
    assert "architecture_feedback" in final["depends_on"]


# --- output template ---


def test_output_template_has_all_22_sections_and_contents() -> None:
    template = (_skill_root() / "templates" / "ux-design-template.md").read_text(
        encoding="utf-8"
    )
    assert "## Contents" in template
    assert "## Update History" in template
    assert "## Appendix A. UX Quality-Gate Self-Check" in template
    for section in REQUIRED_SECTIONS:
        assert section in template, f"template missing section: {section}"


def test_output_template_has_generation_metadata() -> None:
    template = (_skill_root() / "templates" / "ux-design-template.md").read_text(
        encoding="utf-8"
    )
    assert "Generation Metadata" in template
    assert "UX skill version" in template


def test_output_template_separates_operator_and_target_ux() -> None:
    template = (_skill_root() / "templates" / "ux-design-template.md").read_text(
        encoding="utf-8"
    )
    assert "## 5. Skill Operator UX" in template
    assert "## 6. Target Software UX" in template


# --- worked example must itself satisfy the structural gates ---


def test_example_output_is_complete() -> None:
    example = (
        _skill_root() / "examples" / "translation-system-ux-design.md"
    ).read_text(encoding="utf-8")
    assert "## Contents" in example
    assert "## Update History" in example
    assert "## Appendix A. UX Quality-Gate Self-Check" in example
    for section in REQUIRED_SECTIONS:
        assert section in example, f"example missing section: {section}"
    # Story-driven, with a journey diagram and Gherkin E2E seeds.
    assert "## User Story:" in example
    assert "```mermaid" in example
    assert "```gherkin" in example
    # Operator vs target UX kept separate.
    assert "## 5. Skill Operator UX" in example
    assert "## 6. Target Software UX" in example
    # Mandatory architecture-feedback section with a reconcile decision.
    assert "Architecture Feedback / Required Architecture Updates" in example
    assert "architecture --mode reconcile" in example


def test_example_avoids_executable_tests_and_visual_design() -> None:
    """The skill emits E2E *seeds*, not tests, and no pixel-level design."""
    example = (
        (_skill_root() / "examples" / "translation-system-ux-design.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    # No executable-test scaffolding.
    for term in ("import pytest", "def test_", "assert "):
        assert term not in example, f"example leaks executable-test code: {term!r}"
    # No CSS / pixel-level styling.
    for term in ("background-color", "px;", "<div", "font-size"):
        assert term not in example, f"example leaks visual styling: {term!r}"


# --- discoverability by setup ---


def test_skill_is_discoverable_by_setup() -> None:
    """``setup`` auto-discovers any skill_data/*/SKILL.md; confirm ux-design."""
    from research_pipeline.cli.cmd_setup import _find_skill_sources

    sources = _find_skill_sources()
    names = {p.name for p in sources}
    assert "ux-design" in names, "setup did not discover the ux-design skill"
