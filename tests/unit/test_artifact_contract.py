"""Structural guard tests for the Cross-Skill Artifact Contract.

The contract (`references/artifact-contract.md`, shipped inside each skill)
standardizes the document interface across the design-chain skills so each
generated Markdown is both a human report and a machine-readable handoff
artifact. These are simple text-based tests over the bundled skill templates,
references, and prompts (see ``docs`` cross-skill-artifact-contract plan §22).
"""

from __future__ import annotations

from pathlib import Path


def _skill_data() -> Path:
    import research_pipeline

    return Path(research_pipeline.__file__).parent / "skill_data"


# Skills that produce artifacts and must carry the contract.
CONTRACT_SKILLS = ["blueprint", "architecture", "ux-design"]

# Priority artifact-producing templates per skill (contract §19).
PRIORITY_TEMPLATES = {
    "blueprint": ["product_blueprint_template.md"],
    "architecture": [
        "architecture_design_template.md",
        "architecture_tech_stack_template.md",
        "architecture_update_template.md",
        "architecture_reconciliation_template.md",
        "architecture_review_template.md",
    ],
    "ux-design": ["ux-design-template.md"],
}

# The five architecture mode templates must record Resolved Input Artifacts.
ARCHITECTURE_MODE_TEMPLATES = PRIORITY_TEMPLATES["architecture"]


def _template_text(skill: str, name: str) -> str:
    return (_skill_data() / skill / "templates" / name).read_text(encoding="utf-8")


def _all_priority_templates() -> list[tuple[str, str]]:
    return [(s, t) for s, ts in PRIORITY_TEMPLATES.items() for t in ts]


# --- the contract reference file itself ---


def test_contract_file_present_in_every_skill() -> None:
    for skill in CONTRACT_SKILLS:
        path = _skill_data() / skill / "references" / "artifact-contract.md"
        assert path.exists(), f"{skill} missing references/artifact-contract.md"
        assert path.read_text(encoding="utf-8").strip()


def test_contract_files_are_identical() -> None:
    """The canonical contract is duplicated per-skill; the copies must match."""
    texts = {
        skill: (
            _skill_data() / skill / "references" / "artifact-contract.md"
        ).read_text(encoding="utf-8")
        for skill in CONTRACT_SKILLS
    }
    distinct = set(texts.values())
    assert len(distinct) == 1, "artifact-contract.md copies have drifted out of sync"


def test_contract_file_has_registry_and_vocabulary() -> None:
    text = (
        _skill_data() / "architecture" / "references" / "artifact-contract.md"
    ).read_text(encoding="utf-8")
    # Artifact type registry.
    for t in (
        "product_blueprint",
        "architecture_design",
        "architecture_tech_stack",
        "architecture_review",
        "ux_design",
    ):
        assert t in text, f"contract missing artifact type: {t}"
    # Controlled vocabulary clusters.
    for v in ("Decision status", "Stage decision", "Quality gate status"):
        assert v in text, f"contract missing controlled vocabulary: {v}"
    # Topic-slug + filename rules.
    assert "Topic Slug" in text
    assert "<topic-slug>-<artifact-type-name>.md" in text
    # Controlled status values.
    for v in ("accepted", "provisional", "superseded", "NOT_APPLICABLE"):
        assert v in text, f"contract missing controlled value: {v}"


# --- §22 template guards ---


def test_templates_include_generation_metadata() -> None:
    for skill, name in _all_priority_templates():
        text = _template_text(skill, name)
        assert "Generation Metadata" in text, (
            f"{skill}/{name} lacks Generation Metadata"
        )


def test_templates_include_artifact_type() -> None:
    for skill, name in _all_priority_templates():
        text = _template_text(skill, name)
        assert "Artifact Type" in text, f"{skill}/{name} lacks Artifact Type"


def test_templates_include_topic_slug() -> None:
    for skill, name in _all_priority_templates():
        text = _template_text(skill, name)
        assert "Topic Slug" in text, f"{skill}/{name} lacks Topic Slug"


def test_templates_include_source_artifacts_consumed() -> None:
    for skill, name in _all_priority_templates():
        text = _template_text(skill, name)
        assert "Source Artifacts Consumed" in text, (
            f"{skill}/{name} lacks Source Artifacts Consumed"
        )


def test_architecture_mode_templates_include_resolved_inputs() -> None:
    for name in ARCHITECTURE_MODE_TEMPLATES:
        text = _template_text("architecture", name)
        assert "Resolved Input Artifacts" in text, (
            f"architecture/{name} lacks Resolved Input Artifacts"
        )


def test_templates_include_recommended_next_stage() -> None:
    for skill, name in _all_priority_templates():
        text = _template_text(skill, name)
        assert "Recommended Next Stage" in text, (
            f"{skill}/{name} lacks Recommended Next Stage(s)"
        )


def test_templates_include_quality_gate_self_check() -> None:
    for skill, name in _all_priority_templates():
        text = _template_text(skill, name)
        assert "Quality-Gate Self-Check" in text, (
            f"{skill}/{name} lacks Quality-Gate Self-Check"
        )


def test_self_checks_include_cross_skill_contract_gate() -> None:
    for skill, name in _all_priority_templates():
        text = _template_text(skill, name)
        assert "Cross-Skill Artifact Contract Gate" in text, (
            f"{skill}/{name} lacks the Cross-Skill Artifact Contract Gate"
        )


# --- §22 skill + prompt guards ---


def test_skills_reference_artifact_contract() -> None:
    """Each artifact-producing skill points at the contract (SKILL.md or prompts)."""
    for skill in CONTRACT_SKILLS:
        root = _skill_data() / skill
        referenced = any(
            "artifact-contract.md" in p.read_text(encoding="utf-8")
            for p in [root / "SKILL.md", *(root / "prompts").glob("*.md")]
        )
        assert referenced, f"{skill} never references artifact-contract.md"


def test_generation_prompts_require_contract_compliance() -> None:
    """The main generation / final-document prompts instruct contract compliance."""
    prompts = [
        _skill_data() / "blueprint" / "prompts" / "04_generate_blueprint.md",
        _skill_data() / "architecture" / "prompts" / "22_architecture_draft.md",
        _skill_data() / "architecture" / "prompts" / "stack_06_final_stack_document.md",
        _skill_data() / "architecture" / "prompts" / "review_03_final_document.md",
        _skill_data() / "architecture" / "prompts" / "update_03_final_document.md",
        _skill_data() / "architecture" / "prompts" / "reconcile_03_final_document.md",
        _skill_data() / "ux-design" / "prompts" / "12_final_document.md",
    ]
    for p in prompts:
        text = p.read_text(encoding="utf-8")
        assert "Cross-Skill Artifact Contract" in text, (
            f"{p.name} lacks a contract-compliance instruction"
        )
