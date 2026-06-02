"""Tests for the bundled ``blueprint`` skill.

The blueprint skill is a pure prompt-driven transformation skill (no CLI/MCP
backend). These tests validate that the bundled skill files are well-formed,
discoverable by ``setup``, and stay faithful to the design contract:
standard-only SKILL.md frontmatter, the 18-section output template, the five
prompts, the references, and an implementation-neutral example output.
"""

from __future__ import annotations

import json
from pathlib import Path


def _skill_root() -> Path:
    # Resolve via filesystem; skill files are bundled with the package.
    import research_pipeline

    return Path(research_pipeline.__file__).parent / "skill_data" / "blueprint"


REQUIRED_PROMPTS = [
    "01_extract_research_items.md",
    "02_translate_to_product_primitives.md",
    "03_resolve_ideas.md",
    "04_generate_blueprint.md",
    "05_quality_gate.md",
]

REQUIRED_TEMPLATES = [
    "product_blueprint_template.md",
    "translation_map_template.md",
    "workflow_template.md",
    "logical_architecture_template.md",
    "evaluation_strategy_template.md",
]

REQUIRED_REFERENCES = [
    "input-mapping.md",
    "gap-type-mapping.md",
    "borderline-cases.md",
    "troubleshooting.md",
]

# The 18 required output sections, by heading text.
REQUIRED_SECTIONS = [
    "Executive Product Thesis",
    "Source Research Interpretation",
    "Target Users and System Actors",
    "Product Goals and Non-Goals",
    "Research-to-Product Translation Map",
    "Adopt / Adapt / Merge / Defer / Reject Decisions",
    "Core Product Capabilities",
    "Workflow Model",
    "Logical Architecture",
    "Conceptual Information Model",
    "Decision Policies",
    "Risk, Governance, and Safety Model",
    "Evaluation Strategy",
    "MVP Scope",
    "Roadmap and Future Extensions",
    "Open Questions and Validation Plan",
    "Handoff Notes for Technical Design",
    "Traceability Appendix",
]


def test_skill_md_exists_and_has_frontmatter() -> None:
    skill = _skill_root() / "SKILL.md"
    assert skill.exists()
    text = skill.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    head, _, _ = text[4:].partition("\n---\n")
    for key in ("name:", "description:", "license:"):
        assert key in head, f"SKILL.md frontmatter missing: {key}"


def test_skill_md_frontmatter_is_standard_only() -> None:
    """Only standard Copilot CLI fields are allowed (commit b6a9951).

    Non-standard keys (``version``, ``compatibility``, ``aliases``,
    ``metadata``) can prevent skill loading and must stay out of frontmatter.
    """
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    head, _, _ = text[4:].partition("\n---\n")
    for forbidden in ("version:", "compatibility:", "aliases:", "metadata:"):
        assert forbidden not in head, (
            f"Non-standard frontmatter key present: {forbidden}"
        )


def test_skill_md_declares_correct_name() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    assert "name: blueprint" in text


def test_skill_md_has_trigger_phrases() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    for phrase in ("create a blueprint", "design the product", "product-blueprint"):
        assert phrase in text.lower(), f"SKILL.md missing trigger phrase: {phrase}"


def test_skill_md_has_non_goals_no_tech_stack() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8").lower()
    assert "non-goal" in text
    # Must explicitly forbid tech-stack selection.
    assert "framework" in text and "database" in text
    assert "do not select" in text or "do **not** select" in text


def test_skill_md_hands_off_to_other_skills() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    # Upstream and downstream handoffs must be explicit.
    assert "research-pipeline" in text
    assert "technical" in text.lower()


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


def test_manifest_parses_and_has_expected_shape() -> None:
    manifest = _skill_root() / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["workflow_id"] == "blueprint"
    task_ids = {task["id"] for task in data["tasks"]}
    # Core stages and gates must be present.
    for required in (
        "extract-research-items",
        "translate-to-primitives",
        "resolve-ideas",
        "compose-blueprint",
        "quality-gate",
    ):
        assert required in task_ids, f"manifest missing task: {required}"
    assert "quality-gate" in data["mandatory_gates"]


def test_manifest_executors_reference_real_prompt_files() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    for task in data["tasks"]:
        prompt = task.get("executor", {}).get("prompt")
        if prompt:
            assert (_skill_root() / prompt).exists(), f"missing executor: {prompt}"


def test_output_template_has_all_18_sections_and_contents() -> None:
    template = (_skill_root() / "templates" / "product_blueprint_template.md").read_text(
        encoding="utf-8"
    )
    assert "## Contents" in template
    for section in REQUIRED_SECTIONS:
        assert section in template, f"template missing section: {section}"


def test_example_output_is_complete_and_neutral() -> None:
    """The shipped example must itself satisfy the structural gates."""
    example = (
        _skill_root() / "tests" / "sample_outputs" / "product_blueprint_example.md"
    ).read_text(encoding="utf-8")
    # Contents + all 18 sections.
    assert "## Contents" in example
    for section in REQUIRED_SECTIONS:
        assert section in example, f"example missing section: {section}"
    # Both required Mermaid diagrams (workflow + architecture) → at least 2.
    assert example.count("```mermaid") >= 2
    # Carries research citations.
    assert "[2312.01234]" in example or "[Park et al., 2023]" in example


def test_example_output_avoids_tech_stack_terms() -> None:
    """Gate 3 (implementation neutrality) on the shipped example."""
    example = (
        _skill_root() / "tests" / "sample_outputs" / "product_blueprint_example.md"
    ).read_text(encoding="utf-8")
    lowered = example.lower()
    forbidden = [
        "postgresql",
        "mongodb",
        "fastapi",
        "react",
        " aws ",
        "docker",
        "kubernetes",
        "redis",
        "pinecone",
        "create table",
    ]
    for term in forbidden:
        assert term not in lowered, f"example leaks tech-stack term: {term!r}"


def test_sample_inputs_exist() -> None:
    inputs = _skill_root() / "tests" / "sample_inputs"
    assert (inputs / "strong_report_excerpt.md").exists()
    assert (inputs / "weak_report_excerpt.md").exists()


def test_skill_is_discoverable_by_setup() -> None:
    """``setup`` auto-discovers any skill_data/*/SKILL.md; confirm blueprint."""
    from research_pipeline.cli.cmd_setup import _find_skill_sources

    sources = _find_skill_sources()
    names = {p.name for p in sources}
    assert "blueprint" in names, "setup did not discover the blueprint skill"
