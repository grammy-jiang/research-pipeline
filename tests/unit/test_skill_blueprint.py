"""Tests for the bundled ``blueprint`` skill.

The blueprint skill is a pure prompt-driven transformation skill (no CLI/MCP
backend). These tests validate that the bundled skill files are well-formed,
discoverable by ``setup``, and stay faithful to the design contract:
standard-only SKILL.md frontmatter, the 20-section output template, the five
prompts, the references, and an implementation-neutral example output.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def _skill_root() -> Path:
    # Resolve via filesystem; skill files are bundled with the package.
    import research_pipeline

    return Path(research_pipeline.__file__).parent / "skill_data" / "blueprint"


_NUMBERED_SECTION_RE = re.compile(r"^## (?P<number>\d+)\. (?P<title>.+)$", re.MULTILINE)


def _numbered_sections(text: str) -> list[tuple[int, str]]:
    """Return ordered top-level numbered section headings."""
    return [
        (int(match.group("number")), match.group("title").strip())
        for match in _NUMBERED_SECTION_RE.finditer(text)
    ]


def _numbered_section_titles(text: str) -> list[str]:
    return [title for _, title in _numbered_sections(text)]


def _product_blueprint_template_text() -> str:
    return (_skill_root() / "templates" / "product_blueprint_template.md").read_text(
        encoding="utf-8"
    )


def _required_sections_from_template() -> list[str]:
    sections = _numbered_sections(_product_blueprint_template_text())
    numbers = [number for number, _ in sections]
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"template sections are not contiguous from 1: {numbers}"
    )
    return [title for _, title in sections]


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

# Required output sections, by heading text. The product blueprint template is
# the source of truth so section additions, removals, and renames cannot drift
# from a hand-maintained test constant.
REQUIRED_SECTIONS = _required_sections_from_template()


def test_required_sections_match_template_numbered_headings() -> None:
    assert _required_sections_from_template() == REQUIRED_SECTIONS


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


def test_output_template_has_all_numbered_sections_and_contents() -> None:
    template = _product_blueprint_template_text()
    assert "## Contents" in template
    assert _numbered_section_titles(template) == REQUIRED_SECTIONS


def test_example_output_is_complete_and_neutral() -> None:
    """The shipped example must itself satisfy the structural gates."""
    example = (
        _skill_root() / "tests" / "sample_outputs" / "product_blueprint_example.md"
    ).read_text(encoding="utf-8")
    # Contents + every numbered section from the template, in order.
    assert "## Contents" in example
    assert _numbered_section_titles(example) == REQUIRED_SECTIONS
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


# --- Post-generation quality-control improvements (v0.2.0 skill) ---


def test_template_metadata_separates_runs_from_rounds() -> None:
    """§1.5/§1.6 must keep pipeline runs and gap-closure rounds distinct."""
    template = (
        _skill_root() / "templates" / "product_blueprint_template.md"
    ).read_text(encoding="utf-8")
    assert "Pipeline runs integrated" in template
    assert "Gap-closure rounds" in template
    # The ambiguous combined field must be gone.
    assert "Research-pipeline rounds" not in template
    # Skill version must be sourced from manifest, not invented.
    assert "manifest.json" in template


def test_template_actor_table_has_scope_column() -> None:
    template = (
        _skill_root() / "templates" / "product_blueprint_template.md"
    ).read_text(encoding="utf-8")
    assert "| Actor | Scope |" in template


def test_template_mvp_splits_mvp0_mvp1_and_baselines() -> None:
    template = (
        _skill_root() / "templates" / "product_blueprint_template.md"
    ).read_text(encoding="utf-8")
    for heading in ("MVP-0", "MVP-1", "Safety Baseline", "Evaluation Baseline"):
        assert heading in template, f"§14 missing subsection: {heading}"


def test_template_and_example_have_self_check_appendix() -> None:
    for rel in (
        ("templates", "product_blueprint_template.md"),
        ("tests", "sample_outputs", "product_blueprint_example.md"),
    ):
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        assert "Blueprint Quality-Gate Self-Check" in text, f"{rel} missing self-check"


def test_borderline_cases_has_warning_tier() -> None:
    text = (_skill_root() / "references" / "borderline-cases.md").read_text(
        encoding="utf-8"
    )
    assert "Warning tier" in text
    # The four classification levels must be named.
    for level in ("Allowed", "Warning", "Research-derived exception", "Forbidden"):
        assert level in text, f"borderline-cases missing level: {level}"


def test_quality_gate_prompt_covers_new_checks() -> None:
    text = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "metadata integrity",
        "source fidelity",
        "Scope control",
        "self-check",
    ):
        assert needle.lower() in text.lower(), f"gate prompt missing: {needle}"


def test_manifest_skill_version_bumped() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    assert data["version"] == "0.8.0"


# --- v0.3.0 skill: thesis emphasis, MVP-0/MVP-1, gap-citation, actionable
# self-check, release-gate confidence ---


def test_thesis_emphasis_control_present() -> None:
    prompt = (_skill_root() / "prompts" / "04_generate_blueprint.md").read_text(
        encoding="utf-8"
    )
    template = (
        _skill_root() / "templates" / "product_blueprint_template.md"
    ).read_text(encoding="utf-8")
    assert "Emphasis control" in prompt or "primary" in prompt.lower()
    assert "Emphasis" in template


def test_gap_citation_fallback_present() -> None:
    for rel in (
        ("prompts", "04_generate_blueprint.md"),
        ("prompts", "05_quality_gate.md"),
    ):
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        assert "Source Report: Research Gaps" in text, f"{rel} missing gap-citation"
    # The example must model the fallback (no blank gap citation).
    example = (
        _skill_root() / "tests" / "sample_outputs" / "product_blueprint_example.md"
    ).read_text(encoding="utf-8")
    assert "Source Report: Research Gaps" in example


def test_self_check_is_actionable() -> None:
    for rel in (
        ("templates", "product_blueprint_template.md"),
        ("tests", "sample_outputs", "product_blueprint_example.md"),
    ):
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        assert "Required Action" in text, f"{rel} self-check not actionable"
        assert "Blocks Technical Design" in text, f"{rel} missing blocks-TD column"


def test_release_gate_confidence_check_present() -> None:
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    assert "Release-gate confidence" in gate or "release gate" in gate.lower()


def test_example_uses_mvp0_mvp1_split() -> None:
    example = (
        _skill_root() / "tests" / "sample_outputs" / "product_blueprint_example.md"
    ).read_text(encoding="utf-8")
    assert "MVP-0" in example and "MVP-1" in example


def test_borderline_cases_has_example_rewrites() -> None:
    text = (_skill_root() / "references" / "borderline-cases.md").read_text(
        encoding="utf-8"
    )
    assert "Example rewrites" in text


def test_example_metadata_is_not_invented() -> None:
    """The example must model copied (not fabricated) metadata."""
    example = (
        _skill_root() / "tests" / "sample_outputs" / "product_blueprint_example.md"
    ).read_text(encoding="utf-8")
    assert "Pipeline runs integrated" in example
    assert "Gap-closure rounds" in example
    # Must not reintroduce the conflated field the improvement plan flagged.
    assert "Research-pipeline rounds: 15" not in example


# --- v0.4.0 skill: Contents/appendix consistency, self-repair, optional
# Appendix B register ---


def _contents_block(text: str) -> str:
    """Return the text from '## Contents' up to the first horizontal rule."""
    start = text.index("## Contents")
    end = text.index("\n---", start)
    return text[start:end]


def test_template_and_example_contents_list_self_check_appendix() -> None:
    """Imp.1: the Contents must include the self-check appendix link."""
    for rel in (
        ("templates", "product_blueprint_template.md"),
        ("tests", "sample_outputs", "product_blueprint_example.md"),
    ):
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        contents = _contents_block(text)
        assert "Appendix A" in contents, f"{rel} Contents omits Appendix A"


def test_quality_gate_has_self_repair_pass() -> None:
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    assert "Self-repair" in gate
    assert "detect → repair → re-check" in gate or "post-repair" in gate


def test_compose_prompt_lists_appendices_in_contents_rule() -> None:
    prompt = (_skill_root() / "prompts" / "04_generate_blueprint.md").read_text(
        encoding="utf-8"
    )
    assert "every appendix actually present" in prompt
    # Optional Appendix B (design decision register) must be described.
    assert "Design Decision Register" in prompt


def test_template_has_optional_appendix_b() -> None:
    template = (
        _skill_root() / "templates" / "product_blueprint_template.md"
    ).read_text(encoding="utf-8")
    assert "Appendix B: Design Decision Register" in template
    assert "Revisit Trigger" in template


def test_mvp_split_mandatory_trigger_present() -> None:
    for rel in (
        ("prompts", "04_generate_blueprint.md"),
        ("templates", "product_blueprint_template.md"),
    ):
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        # Tolerate line wrapping and blockquote markers between words.
        normalized = " ".join(text.replace(">", " ").split())
        assert "more than 4 major capabilities" in normalized, (
            f"{rel} missing MVP trigger"
        )


# --- v0.5.0 skill: §9 Product Experience Direction (UX-intent support) ---

# Files that must carry the new §9 section, in order, after Workflow Model and
# before Logical Architecture.
_PE_SECTION = "## 9. Product Experience Direction"
_PE_DOCS = (
    ("templates", "product_blueprint_template.md"),
    ("tests", "sample_outputs", "product_blueprint_example.md"),
)


def test_product_experience_reference_exists_and_nonempty() -> None:
    ref = _skill_root() / "references" / "product-experience-direction.md"
    assert ref.exists(), "missing references/product-experience-direction.md"
    text = ref.read_text(encoding="utf-8")
    assert text.strip()
    # Boundary rule and gate must be documented.
    assert "Boundary rule" in text
    assert "Product Experience Gate" in text
    assert "UX intent" in text


def test_template_and_example_have_product_experience_direction() -> None:
    for rel in _PE_DOCS:
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        assert _PE_SECTION in text, f"{rel} missing §9 Product Experience Direction"
        # The section heading must precede the Logical Architecture heading.
        assert text.index(_PE_SECTION) < text.index("## 10. Logical Architecture"), (
            f"{rel}: §9 must come before Logical Architecture"
        )


def test_template_and_example_are_now_20_sections() -> None:
    """§19 insert renumbers the tail; the document must run 1..20 in order."""
    for rel in _PE_DOCS:
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        numbers = [number for number, _ in _numbered_sections(text)]
        assert numbers == list(range(1, 21)), f"{rel} sections not 1..20: {numbers}"


def test_contents_lists_product_experience_direction() -> None:
    for rel in _PE_DOCS:
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        contents = _contents_block(text)
        assert "Product Experience Direction" in contents, f"{rel} Contents omits §9"


def test_template_section_has_required_ux_subsections() -> None:
    template = (
        _skill_root() / "templates" / "product_blueprint_template.md"
    ).read_text(encoding="utf-8")
    for needle in (
        "Primary Experience Thesis",
        "Primary User / Operator",
        "Primary Job-to-Be-Done",
        "Primary Interaction Mode",
        "Critical Trust, Control, and Transparency Requirements",
        "Human-in-the-Loop Experience",
        "Failure and Recovery Expectations",
        "UX Assumptions for Architecture",
        "Product Experience Handoff to Architecture",
    ):
        assert needle in template, f"§9 template missing: {needle}"


def test_example_has_product_experience_handoff() -> None:
    example = (
        _skill_root() / "tests" / "sample_outputs" / "product_blueprint_example.md"
    ).read_text(encoding="utf-8")
    assert "Product Experience Handoff to Architecture" in example


def test_quality_gate_has_product_experience_gate() -> None:
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    assert "Product experience direction" in gate or "Product Experience Gate" in gate


def test_template_and_example_self_check_has_product_experience_gate() -> None:
    for rel in (
        ("templates", "product_blueprint_template.md"),
        ("tests", "sample_outputs", "product_blueprint_example.md"),
    ):
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        assert "Product Experience Gate" in text, f"{rel} self-check missing PE gate"
        assert "Primary user identified" in text, f"{rel} missing PE gate row"


def test_compose_prompt_covers_product_experience() -> None:
    prompt = (_skill_root() / "prompts" / "04_generate_blueprint.md").read_text(
        encoding="utf-8"
    )
    assert "Product experience direction" in prompt
    assert "references/product-experience-direction.md" in prompt


def test_skill_md_declares_20_sections_with_product_experience() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    assert "20 sections" in text
    assert "9. Product Experience Direction" in text
    assert "19. Recommended Next Stages" in text


def test_skill_md_forbids_detailed_ux_design() -> None:
    """The skill must route detailed UI/UX away (it is UX intent only)."""
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8").lower()
    assert "wireframe" in text
    assert "ux intent" in text


def test_skill_md_description_fits_copilot_limit() -> None:
    """GitHub Copilot CLI rejects skills whose description exceeds 1024 chars."""
    import re

    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    head, _, _ = text[4:].partition("\n---\n")
    match = re.search(r"description: >\n(.*?)\nlicense:", head, re.S)
    assert match, "could not parse folded description block"
    description = " ".join(line.strip() for line in match.group(1).splitlines())
    assert len(description) <= 1024, f"description is {len(description)} chars (>1024)"


def test_manifest_mentions_product_experience() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    assert "Product Experience" in data["description"]
    compose = next(t for t in data["tasks"] if t["id"] == "compose-blueprint")
    assert "product_experience_direction_present" in compose["validation"]
    assert "all_20_sections_present" in compose["validation"]


# --- v0.6.0 skill: §19 Recommended Next Stages (adaptive stage-gate routing) ---

_RNS_SECTION = "## 19. Recommended Next Stages"
_RNS_DOCS = (
    ("templates", "product_blueprint_template.md"),
    ("tests", "sample_outputs", "product_blueprint_example.md"),
)
_CONTROLLED_DECISIONS = ("RUN", "SKIP", "DEFER", "ASK_USER")


def test_adaptive_routing_reference_exists_and_nonempty() -> None:
    ref = _skill_root() / "references" / "adaptive-stage-gate-routing.md"
    assert ref.exists(), "missing references/adaptive-stage-gate-routing.md"
    text = ref.read_text(encoding="utf-8")
    assert text.strip()
    # Decision vocabulary, complexity scoring, and the gate must be documented.
    assert "Decision vocabulary" in text
    assert "Adaptive Stage-Gate Recommendation Gate" in text
    for decision in _CONTROLLED_DECISIONS:
        assert decision in text, f"routing reference missing decision: {decision}"


def test_template_and_example_have_recommended_next_stages() -> None:
    """§19 inserts before the (renumbered) §20 Traceability Appendix."""
    for rel in _RNS_DOCS:
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        assert _RNS_SECTION in text, f"{rel} missing §19 Recommended Next Stages"
        assert text.index(_RNS_SECTION) < text.index("## 20. Traceability Appendix"), (
            f"{rel}: §19 must come before the Traceability Appendix"
        )


def test_contents_lists_recommended_next_stages() -> None:
    for rel in _RNS_DOCS:
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        contents = _contents_block(text)
        assert "Recommended Next Stages" in contents, f"{rel} Contents omits §19"


def test_template_section_has_required_routing_subsections() -> None:
    template = (
        _skill_root() / "templates" / "product_blueprint_template.md"
    ).read_text(encoding="utf-8")
    for needle in (
        "Pipeline Complexity Assessment",
        "Stage Recommendations",
        "Recommended Pipeline",
        "Stage-Gate Decision Log",
        "User-facing complexity",
        "Technical ambiguity",
        "Security / privacy risk",
        "AI / LLM uncertainty",
        "Integration complexity",
        "Human-review complexity",
        "Testing / E2E importance",
    ):
        assert needle in template, f"§19 template missing: {needle}"


def test_routing_uses_controlled_decision_vocabulary() -> None:
    """Template + example must surface only RUN / SKIP / DEFER / ASK_USER."""
    for rel in _RNS_DOCS:
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        for decision in _CONTROLLED_DECISIONS:
            assert decision in text, f"{rel} §19 missing decision: {decision}"


def test_example_stage_recommendation_defaults() -> None:
    """architecture-design RUN; architecture-update/-reconciliation DEFER."""
    example = (
        _skill_root() / "tests" / "sample_outputs" / "product_blueprint_example.md"
    ).read_text(encoding="utf-8")
    assert "| architecture-design | RUN |" in example
    assert "| architecture-update | DEFER |" in example
    assert "| architecture-reconciliation | DEFER |" in example


def test_compose_prompt_covers_recommended_next_stages() -> None:
    prompt = (_skill_root() / "prompts" / "04_generate_blueprint.md").read_text(
        encoding="utf-8"
    )
    assert "Recommended next stages" in prompt
    assert "references/adaptive-stage-gate-routing.md" in prompt
    for decision in _CONTROLLED_DECISIONS:
        assert decision in prompt, f"compose prompt missing decision: {decision}"


def test_quality_gate_has_adaptive_stage_gate() -> None:
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    assert "Adaptive stage-gate recommendation" in gate
    assert "Adaptive Stage-Gate Recommendation Gate" in gate


def test_template_and_example_self_check_has_adaptive_stage_gate() -> None:
    for rel in _RNS_DOCS:
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        assert "Adaptive Stage-Gate Recommendation Gate" in text, (
            f"{rel} self-check missing adaptive stage-gate"
        )
        assert "Recommended Next Stages section exists" in text, (
            f"{rel} missing adaptive stage-gate row"
        )


def test_skill_md_mentions_adaptive_routing() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    assert "RUN / SKIP / DEFER / ASK_USER" in text
    assert "Recommended Next Stages" in text
    assert "references/adaptive-stage-gate-routing.md" in text


def test_manifest_mentions_recommended_next_stages() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    assert "Recommended Next Stages" in data["description"]
    compose = next(t for t in data["tasks"] if t["id"] == "compose-blueprint")
    assert "recommended_next_stages_present" in compose["validation"]


# --- v0.7.0 skill: adaptive-routing clarity (Depends On, linear-vs-conditional
# split, ASK_USER rationale, interaction-mode classification, heuristic label) ---

_V07_DOCS = (
    ("templates", "product_blueprint_template.md"),
    ("tests", "sample_outputs", "product_blueprint_example.md"),
)


def test_stage_recommendations_have_depends_on_column() -> None:
    """§19.2 must carry an explicit ``Depends On`` column (template + example)."""
    for rel in _V07_DOCS:
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        assert "Depends On" in text, f"{rel} §19.2 missing Depends On column"
    # The compose prompt and routing reference must instruct it.
    prompt = (_skill_root() / "prompts" / "04_generate_blueprint.md").read_text(
        encoding="utf-8"
    )
    ref = (_skill_root() / "references" / "adaptive-stage-gate-routing.md").read_text(
        encoding="utf-8"
    )
    assert "Depends On" in prompt
    assert "Depends On" in ref


def test_recommended_pipeline_splits_linear_and_conditional() -> None:
    """§19 must separate the linear core path from conditional follow-up gates."""
    for rel in _V07_DOCS:
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        assert "Recommended Linear Path" in text, (
            f"{rel} missing Recommended Linear Path"
        )
        assert "Conditional Follow-up Gates" in text, (
            f"{rel} missing Conditional Follow-up Gates"
        )
    ref = (_skill_root() / "references" / "adaptive-stage-gate-routing.md").read_text(
        encoding="utf-8"
    )
    assert "Recommended Linear Path" in ref
    assert "Conditional Follow-up Gates" in ref


def test_ask_user_decision_rationale_present() -> None:
    """§19 must explain ASK_USER decisions (or their absence)."""
    for rel in _V07_DOCS:
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        assert "ASK_USER Decision Rationale" in text, (
            f"{rel} missing ASK_USER Decision Rationale"
        )
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    assert "ASK_USER Decision Rationale" in gate or "ASK_USER absence" in gate


def test_complexity_score_labelled_as_heuristic() -> None:
    """§19.1 must flag the complexity score as a routing heuristic, not an estimate."""
    for rel in _V07_DOCS:
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        assert "routing heuristic" in text, f"{rel} §19.1 missing heuristic label"
    ref = (_skill_root() / "references" / "adaptive-stage-gate-routing.md").read_text(
        encoding="utf-8"
    )
    assert "routing heuristic" in ref


def test_interaction_modes_have_classification_column() -> None:
    """§9.4/§9.5 must classify each interaction mode with a controlled value."""
    for rel in _V07_DOCS:
        text = _skill_root().joinpath(*rel).read_text(encoding="utf-8")
        assert "| Mode | Classification |" in text, (
            f"{rel} §9.4 missing Classification column"
        )
    ref = (_skill_root() / "references" / "product-experience-direction.md").read_text(
        encoding="utf-8"
    )
    # Controlled vocabulary + the AI-Skill-vs-MCP disambiguation must be documented.
    assert "wrapper / integration surface" in ref
    assert "primary surface" in ref


def test_compose_prompt_covers_v07_routing_clarity() -> None:
    prompt = (_skill_root() / "prompts" / "04_generate_blueprint.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "Recommended Linear Path",
        "Conditional Follow-up Gates",
        "ASK_USER Decision Rationale",
        "routing heuristic",
        "Classification",
    ):
        assert needle in prompt, f"compose prompt missing v0.7.0 element: {needle}"


def test_quality_gate_covers_v07_routing_clarity() -> None:
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "Depends On",
        "Recommended Linear Path",
        "Conditional Follow-up Gates",
        "routing heuristic",
    ):
        assert needle in gate, f"quality gate missing v0.7.0 check: {needle}"


# --- issue #81: deterministic cross-phase coherence guard ---
#
# The blueprint quality-gate is a pure-LLM stage; it silently passed
# cross-phase-incoherent blueprints (an MVP-0 node whose required servicer is
# staged MVP-1, or an MVP control gated on a non-blocking open question). A
# deterministic guard (scripts/check_blueprint_coherence.py, wired as the
# check-coherence manifest task between compose-blueprint and quality-gate)
# now catches this. These tests pin that behaviour with a minimal
# coherent/incoherent fixture pair.

_COHERENCE_SCRIPT = _skill_root() / "scripts" / "check_blueprint_coherence.py"
_COHERENCE_FIXTURES = _skill_root() / "tests" / "coherence_fixtures"


def _run_coherence_guard(fixture: str) -> tuple[int, dict]:
    """Run the guard over a fixture; return (exit_code, parsed JSON result)."""
    proc = subprocess.run(
        [sys.executable, str(_COHERENCE_SCRIPT), str(_COHERENCE_FIXTURES / fixture)],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, json.loads(proc.stdout)


def test_coherence_guard_script_exists_and_nonempty() -> None:
    assert _COHERENCE_SCRIPT.exists(), "missing scripts/check_blueprint_coherence.py"
    assert _COHERENCE_SCRIPT.read_text(encoding="utf-8").strip()


def test_coherence_fixtures_exist() -> None:
    assert (_COHERENCE_FIXTURES / "coherent_blueprint.md").exists()
    assert (_COHERENCE_FIXTURES / "incoherent_blueprint.md").exists()


def test_coherent_fixture_passes_guard() -> None:
    code, result = _run_coherence_guard("coherent_blueprint.md")
    assert code == 0, f"coherent fixture unexpectedly failed: {result['findings']}"
    assert result["all_passed"] is True
    assert result["fail_count"] == 0


def test_incoherent_fixture_fails_on_phase_inversion_and_open_dependency() -> None:
    code, result = _run_coherence_guard("incoherent_blueprint.md")
    assert code == 1
    assert result["all_passed"] is False
    failed_checks = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    # Both known defect classes from issue #81 must be caught deterministically.
    assert "phase_inversion" in failed_checks, failed_checks
    assert "open_dependency" in failed_checks, failed_checks


def test_manifest_has_coherence_gate_between_compose_and_quality() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    ids = [task["id"] for task in data["tasks"]]
    assert "check-coherence" in ids, "manifest missing check-coherence task"
    assert (
        ids.index("compose-blueprint")
        < ids.index("check-coherence")
        < ids.index("quality-gate")
    ), "check-coherence must sit between compose-blueprint and quality-gate"
    task = next(t for t in data["tasks"] if t["id"] == "check-coherence")
    assert task["executor"]["kind"] == "deterministic_script"
    assert "check_blueprint_coherence.py" in task["executor"]["command"]
    quality_gate = next(t for t in data["tasks"] if t["id"] == "quality-gate")
    assert "check-coherence" in quality_gate["depends_on"]
    assert "check-coherence" in data["mandatory_gates"]


def test_template_documents_coherence_anchors() -> None:
    template = (
        _skill_root() / "templates" / "product_blueprint_template.md"
    ).read_text(encoding="utf-8")
    assert "<!-- coherence:" in template, "template missing a coherence anchor example"
    assert "phase inversion" in template.lower()


def test_quality_gate_splits_conditions_atomically() -> None:
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    # One-condition-per-check restructuring + the deterministic pre-gate.
    assert "one condition per check" in gate.lower()
    assert "check_blueprint_coherence" in gate
    # The 3-clause release-gate condition is now split per clause (issue #81
    # called out Gate 6's release-gate AND being verified holistically).
    for clause in ("6b.i", "6b.ii", "6b.iii"):
        assert clause in gate, f"release-gate clause not split: {clause}"


# --- issue #85: citation fidelity, agent-mode authorization boundary, and
# amend-without-new-research playbook ---


def _run_coherence_guard_with_source(
    blueprint: Path, source_report: Path
) -> tuple[int, dict]:
    """Run the deterministic guard with source-report citation validation."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_COHERENCE_SCRIPT),
            str(blueprint),
            "--source-report",
            str(source_report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, json.loads(proc.stdout)


def test_coherence_guard_fails_when_blueprint_citation_missing_from_references(
    tmp_path: Path,
) -> None:
    source_report = tmp_path / "source-report.md"
    source_report.write_text(
        """# Source Report

## References

- [2312.01234] Evaluator-gated memory writes.
""",
        encoding="utf-8",
    )
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        """# Product Blueprint

## Contents

- [1. Executive Product Thesis](#1-executive-product-thesis)

---

## 1. Executive Product Thesis

This thesis cites evidence that is absent from the source references.
[9999.00000]
""",
        encoding="utf-8",
    )

    code, result = _run_coherence_guard_with_source(blueprint, source_report)

    assert code == 1
    assert result["all_passed"] is False
    failed_checks = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert "citation_not_in_references" in failed_checks


def test_coherence_guard_accepts_blueprint_citation_present_in_references(
    tmp_path: Path,
) -> None:
    source_report = tmp_path / "source-report.md"
    source_report.write_text(
        """# Source Report

## References

- [2312.01234] Evaluator-gated memory writes.
""",
        encoding="utf-8",
    )
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        """# Product Blueprint

## Contents

- [1. Executive Product Thesis](#1-executive-product-thesis)

---

## 1. Executive Product Thesis

This thesis cites evidence present in the source references. [2312.01234]
""",
        encoding="utf-8",
    )

    code, result = _run_coherence_guard_with_source(blueprint, source_report)

    assert code == 0
    assert result["all_passed"] is True
    failed_checks = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert "citation_not_in_references" not in failed_checks


def test_coherence_guard_fails_on_silent_confidence_upgrade(tmp_path: Path) -> None:
    source_report = tmp_path / "source-report.md"
    source_report.write_text(
        """# Source Report

## Confidence-Graded Findings

- MEDIUM — Selective forgetting improves signal but has loss risk. [2402.01234]

## References

- [2402.01234] Selective forgetting.
""",
        encoding="utf-8",
    )
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        """# Product Blueprint

## Contents

- [1. Executive Product Thesis](#1-executive-product-thesis)

---

## 1. Executive Product Thesis

| Claim | Confidence | Citation |
|---|---|---|
| Selective forgetting is release-ready. | HIGH | [2402.01234] |
""",
        encoding="utf-8",
    )

    code, result = _run_coherence_guard_with_source(blueprint, source_report)

    assert code == 1
    failed_checks = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert "confidence_silently_upgraded" in failed_checks


def test_manifest_wires_source_report_into_deterministic_guard() -> None:
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    task = next(t for t in data["tasks"] if t["id"] == "check-coherence")
    command = task["executor"]["command"]
    assert "--source-report" in command
    assert "<topic-slug>-research-report.md" in command


def test_quality_gate_covers_load_bearing_citation_fidelity() -> None:
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "load-bearing claims",
        "thesis emphasis",
        "primary interaction mode",
        "primary actor",
        "re-read the cited source-report section",
        "product-design decision",
        "citation string exists in the source report's `## References`",
        "confidence grade was not silently upgraded",
    ):
        assert needle in gate, f"Gate 2 missing citation-fidelity rule: {needle}"


def test_product_experience_reference_requires_agent_authorization_boundary() -> None:
    ref = (_skill_root() / "references" / "product-experience-direction.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "agent-callable / tool-driven",
        "READ/ACT authorization boundary",
        "matching §13 risk row",
        "agent authority-confusion",
        "prompt injection",
        "do not define exact MCP tool schemas",
    ):
        assert needle in ref, f"product-experience reference missing: {needle}"


def test_quality_gate_requires_agent_mode_authorization_boundary() -> None:
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "agent-callable / tool-driven",
        "READ/ACT authorization-boundary statement",
        "promoted secondary",
        "matching §13 risk row",
        "agent authority-confusion",
        "prompt injection",
    ):
        assert needle in gate, f"Gate 8 missing authorization-boundary rule: {needle}"


def test_troubleshooting_has_amend_without_new_report_playbook() -> None:
    text = (_skill_root() / "references" / "troubleshooting.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "Amend an existing blueprint with a new decision (no new research report)",
        "load-bearing fact classes",
        "interaction mode → §3, §8, §9.4/9.5, §10, §16, §18, §19, Appendix A",
        "MVP roster → §7, §12, §13, §14, §15, Appendix A",
        "surgical amendment",
        "pre-delivery propagation check",
    ):
        assert needle in text, f"troubleshooting playbook missing: {needle}"


def test_compose_prompt_has_pre_delivery_propagation_check() -> None:
    prompt = (_skill_root() / "prompts" / "04_generate_blueprint.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "Pre-delivery propagation check",
        "amend an existing blueprint",
        "interaction mode → §3, §8, §9.4/9.5, §10, §16, §18, §19, Appendix A",
        "MVP roster → §7, §12, §13, §14, §15, Appendix A",
    ):
        assert needle in prompt, f"compose prompt missing propagation check: {needle}"
