"""Structural guard tests for the Cross-Skill Artifact Contract (producer side).

The contract (`references/artifact-contract.md`, shipped inside the blueprint skill)
standardizes the document interface for the design chain so each generated Markdown is
both a human report and a machine-readable handoff artifact. The downstream consumers
(``architecture``, ``ux-design``) now live in the separate ``design-pipeline`` repo;
their template/prompt conformance is tested there. This file validates the blueprint
(producer) side that remains in research-pipeline.
"""

from __future__ import annotations

from pathlib import Path


def _skill_data() -> Path:
    import research_pipeline

    return Path(research_pipeline.__file__).parent / "skill_data"


# The artifact-producing skill that remains in this repo and must carry the contract.
CONTRACT_SKILLS = ["blueprint"]

# Priority artifact-producing templates per skill (contract §19).
PRIORITY_TEMPLATES = {
    "blueprint": ["product_blueprint_template.md"],
}


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


def test_contract_file_has_registry_and_vocabulary() -> None:
    text = (
        _skill_data() / "blueprint" / "references" / "artifact-contract.md"
    ).read_text(encoding="utf-8")
    # Artifact type registry (the contract defines the whole chain, incl. consumers).
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
    """The main blueprint generation prompt instructs contract compliance."""
    prompts = [
        _skill_data() / "blueprint" / "prompts" / "04_generate_blueprint.md",
    ]
    for p in prompts:
        text = p.read_text(encoding="utf-8")
        assert "Cross-Skill Artifact Contract" in text, (
            f"{p.name} lacks a contract-compliance instruction"
        )
