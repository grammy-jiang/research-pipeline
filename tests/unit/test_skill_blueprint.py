"""Tests for the bundled ``blueprint`` skill.

The blueprint skill is a pure prompt-driven transformation skill (no CLI/MCP
backend). These tests validate that the bundled skill files are well-formed,
discoverable by ``setup``, and stay faithful to the design contract:
standard-only SKILL.md frontmatter, the 20-section output template, the five
core prompts plus input-quality prompt, the references, and an
implementation-neutral example output.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


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


def _frontmatter(text: str) -> str:
    assert text.startswith("---\n"), "file must start with YAML frontmatter"
    head, _, _ = text[4:].partition("\n---\n")
    return head


def _frontmatter_description(text: str) -> str:
    match = re.search(r"description: >\n(.*?)\nlicense:", _frontmatter(text), re.S)
    assert match, "could not parse folded description block"
    return " ".join(line.strip() for line in match.group(1).splitlines()).lower()


def _block_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def _numbered_list_titles(block: str) -> list[str]:
    return [
        match.group("title").strip()
        for match in re.finditer(
            r"^\s*(?:- \[ \] )?(?P<number>\d+)\. (?P<title>.+)$",
            block,
            re.MULTILINE,
        )
    ]


def _when_to_trigger_phrases(text: str) -> list[str]:
    block = _block_between(text, "## When To Trigger", "Do **not** trigger for:")
    quoted = {
        " ".join(match.group(1).split()).lower()
        for match in re.finditer(r'"([^"]+)"', block)
    }
    aliases_line = re.search(r"- The aliases (?P<line>.*)", block)
    aliases = set(re.findall(r"`([^`]+)`", aliases_line.group("line")))
    return sorted(quoted | {alias.lower() for alias in aliases})


REQUIRED_PROMPTS = [
    "00_assess_input_quality.md",
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


def test_hand_authored_section_lists_match_template_headings() -> None:
    """The template headings are the only source of truth for the 20 sections."""
    expected = _required_sections_from_template()
    skill_text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    prompt_text = (_skill_root() / "prompts" / "04_generate_blueprint.md").read_text(
        encoding="utf-8"
    )
    checklist_text = (
        _skill_root() / "tests" / "expected_sections_checklist.md"
    ).read_text(encoding="utf-8")

    section_lists = {
        "SKILL.md": _numbered_list_titles(
            _block_between(
                skill_text,
                "The blueprint must contain these 20 sections",
                "Use `templates/product_blueprint_template.md` as the skeleton.",
            )
        ),
        "prompts/04_generate_blueprint.md": _numbered_list_titles(
            _block_between(prompt_text, "## Required sections", "## Thesis")
        ),
        "tests/expected_sections_checklist.md": _numbered_list_titles(
            _block_between(
                checklist_text,
                "All 20 required sections present",
                "## Content quality",
            )
        ),
    }

    assert section_lists == dict.fromkeys(section_lists, expected), (
        "hand-authored section list drifted from template headings"
    )


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


def test_skill_frontmatter_description_covers_when_to_trigger_phrases() -> None:
    """Discovery reads frontmatter first, so body trigger phrases must not drift."""
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    description = _frontmatter_description(text)
    trigger_phrases = set(_when_to_trigger_phrases(text))
    intentionally_omitted = {
        # This is a handover subsection title in the research-pipeline skill,
        # not a likely user invocation phrase for this skill.
        "system-design handover",
    }

    unknown_omissions = intentionally_omitted - trigger_phrases
    assert not unknown_omissions, (
        "frontmatter trigger omission allowlist contains unknown phrases: "
        f"{sorted(unknown_omissions)}"
    )
    missing = sorted(
        phrase
        for phrase in trigger_phrases - intentionally_omitted
        if phrase not in description
    )
    assert not missing, (
        f"SKILL.md frontmatter description misses When To Trigger phrases: {missing}"
    )


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


def test_manifest_has_orthogonal_intake_quality_and_extraction_owners() -> None:
    """Intake, quality assessment, and extraction must not share one prompt owner."""
    data = json.loads((_skill_root() / "manifest.json").read_text(encoding="utf-8"))
    tasks = {task["id"]: task for task in data["tasks"]}

    assert tasks["intake"]["executor"]["kind"] == "document_discovery"
    assert "prompt" not in tasks["intake"]["executor"]
    assert tasks["assess-input-quality"]["executor"]["prompt"] == (
        "prompts/00_assess_input_quality.md"
    )
    assert tasks["extract-research-items"]["executor"]["prompt"] == (
        "prompts/01_extract_research_items.md"
    )


def test_resolve_prompt_is_single_owner_of_mvp_membership_decisions() -> None:
    """Prompt 02 may rank primitives, but Prompt 03 owns MVP inclusion."""
    translate = (
        _skill_root() / "prompts" / "02_translate_to_product_primitives.md"
    ).read_text(encoding="utf-8")
    resolve = (_skill_root() / "prompts" / "03_resolve_ideas.md").read_text(
        encoding="utf-8"
    )

    assert "mvp candidate" not in translate.lower()
    assert "final mvp inclusion is decided only in prompt 03" in translate.lower()
    assert "| Source Idea | Research Citation | Decision |" in resolve
    assert "| Rationale | MVP? | Related Risks |" in resolve


def test_quality_gate_references_compose_prompt_as_generation_rule_owner() -> None:
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    assert "prompts/04_generate_blueprint.md is the authoritative owner" in gate
    assert "verify outcomes; do not redefine generation rules" in gate


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
    assert data["version"] == "0.9.0"


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


# --- issue #83: trustworthy golden fixture + labelled negative/adversarial
# fixtures + weak-input output fixture ---
#
# The blueprint skill's only golden output fixture itself reproduced both
# coherence defects the deterministic guard (issue #81) is meant to catch: an
# MVP-0 admission gate whose contradiction servicer was staged MVP-1
# (servicer-reachability → ``phase_inversion``) and an MVP-0 redundancy gate
# whose dedup precondition was a non-blocking open question
# (precondition-currency → ``open_dependency``). It also carried no coherence
# anchors, so the guard passed it vacuously, and there were no negative fixtures
# proving the checklists reject a violating document, nor a weak-input output
# fixture. These tests pin the regenerated coherent oracle, the labelled
# regression pairs, the mutation-derived negative set, and the weak-input
# output fixture.

_GOLDEN_EXAMPLE = (
    _skill_root() / "tests" / "sample_outputs" / "product_blueprint_example.md"
)
_WEAK_OUTPUT = (
    _skill_root() / "tests" / "sample_outputs" / "weak_input_blueprint_example.md"
)
_REGRESSIONS_DIR = _skill_root() / "tests" / "regressions"
_MUTATIONS_DIR = _skill_root() / "tests" / "mutations"
_MINI_SOURCE_REPORT = _MUTATIONS_DIR / "mini_source_report.md"
_NEUTRALITY_FORBIDDEN_TERMS = frozenset(
    {
        "postgresql",
        "mongodb",
        "mysql",
        "sqlite",
        "redis",
        "fastapi",
        "django",
        "react",
        "aws",
        "gcp",
        "azure",
        "docker",
        "kubernetes",
        "pinecone",
    }
)


def _run_guard_over(blueprint: Path, source: Path | None = None) -> tuple[int, dict]:
    """Run the deterministic coherence guard over a path (optional source)."""
    args = [sys.executable, str(_COHERENCE_SCRIPT), str(blueprint)]
    if source is not None:
        args += ["--source-report", str(source)]
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    return proc.returncode, json.loads(proc.stdout)


def _label_field(text: str, key: str) -> str | None:
    """Read a ``key=value`` field from a fixture's leading label comment."""
    match = re.search(rf"\b{re.escape(key)}=(\S+)", text)
    return match.group(1) if match else None


def _mutation_fixtures() -> list[Path]:
    """Every single-mutation negative fixture (files with a mutation label)."""
    if not _MUTATIONS_DIR.is_dir():
        return []
    return sorted(
        path
        for path in _MUTATIONS_DIR.glob("*.md")
        if path.read_text(encoding="utf-8").lstrip().startswith("<!-- mutation:")
    )


def test_golden_example_passes_coherence_guard_with_anchors() -> None:
    """The shipped exemplar is an independently-verified, phase-clean oracle.

    Run without ``--source-report``: the cross-phase coherence graph is the
    trustworthy-oracle requirement. Source citation/confidence fidelity is a
    separate line-based heuristic covered by dedicated tests above.
    """
    code, result = _run_guard_over(_GOLDEN_EXAMPLE)
    assert code == 0, result["findings"]
    assert result["all_passed"] is True
    # Must actually carry anchors — an anchorless fixture passes only vacuously
    # (with a ``no_coherence_anchors`` warning) and cannot be a trustworthy
    # expected-object. This is the assertion that fails on the pre-#83 fixture.
    assert result["node_count"] > 0, "golden fixture carries no coherence anchors"
    fails = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert not fails, fails
    warnings = {f["check"] for f in result["findings"] if f["level"] == "WARNING"}
    assert "no_coherence_anchors" not in warnings


def test_weak_input_output_fixture_is_coherent_and_disciplined() -> None:
    """A weak-input OUTPUT fixture exists and models weak-input discipline."""
    assert _WEAK_OUTPUT.exists(), "missing weak-input output fixture"
    text = _WEAK_OUTPUT.read_text(encoding="utf-8")
    # Follows the full template, like the strong exemplar.
    assert "## Contents" in text
    assert _numbered_section_titles(text) == REQUIRED_SECTIONS
    # Declares the weak input and keeps assumption / open-question discipline
    # instead of fabricating confidence grades.
    assert "**Input quality:** weak" in text
    assert "Assumption" in text
    assert "confidence grade is invented" in text.lower()
    assert "no invented confidence" in text.lower()
    # A small product must not self-score as complex: it uses the lightweight
    # routing band, unlike the strong exemplar's ``complex (13+)``.
    assert "**Total Score:** 7 / 21" in text
    assert "lightweight" in text
    assert "complex (13+)" not in text
    # Implementation-neutral, same gate as the strong exemplar.
    lowered = text.lower()
    for term in (
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
    ):
        assert term not in lowered, f"weak fixture leaks tech-stack term: {term!r}"
    # Phase-clean and anchored.
    code, result = _run_guard_over(_WEAK_OUTPUT)
    assert code == 0, result["findings"]
    assert result["all_passed"] is True
    assert result["node_count"] > 0, "weak fixture carries no coherence anchors"


def test_issue83_negative_fixture_set_is_present() -> None:
    """The regression pairs and the mutation set must all ship."""
    for pair in ("servicer-reachability", "precondition-currency"):
        assert (_REGRESSIONS_DIR / pair / "bad.md").exists(), pair
        assert (_REGRESSIONS_DIR / pair / "fixed.md").exists(), pair
    mutation_names = {path.name for path in _mutation_fixtures()}
    for expected in (
        "invert-mvp-tag.md",
        "sever-servicer-edge.md",
        "blank-citation.md",
        "swap-confidence-grade.md",
        "forbidden-term.md",
    ):
        assert expected in mutation_names, f"missing mutation fixture: {expected}"
    assert _MINI_SOURCE_REPORT.exists()


@pytest.mark.parametrize("pair", ["servicer-reachability", "precondition-currency"])
def test_regression_pair_bad_fails_labelled_check_and_fixed_passes(pair: str) -> None:
    """Each pair proves a named coherence check catches a real defect.

    ``bad.md`` reproduces one golden-fixture defect and must FAIL its labelled
    check; ``fixed.md`` applies the golden-fixture fix and must PASS cleanly.
    """
    bad = _REGRESSIONS_DIR / pair / "bad.md"
    fixed = _REGRESSIONS_DIR / pair / "fixed.md"
    expected_check = _label_field(bad.read_text(encoding="utf-8"), "check")
    assert expected_check, f"{pair}/bad.md missing a `check=` label"

    code, result = _run_guard_over(bad)
    assert code == 1, f"{pair}/bad.md unexpectedly passed the guard"
    fails = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert expected_check in fails, (pair, expected_check, fails)

    code_fixed, result_fixed = _run_guard_over(fixed)
    assert code_fixed == 0, result_fixed["findings"]
    assert result_fixed["all_passed"] is True
    assert result_fixed["node_count"] > 0


@pytest.mark.parametrize("path", _mutation_fixtures(), ids=lambda p: p.name)
def test_mutation_negative_triggers_labelled_check(path: Path) -> None:
    """Each single-mutation fixture triggers exactly the check it is labelled with.

    Coherence mutations are caught by the deterministic guard at their labelled
    level; the neutrality mutation is caught by the implementation-neutrality
    gate (a forbidden tech-stack term), which the coherence guard deliberately
    does not police.
    """
    text = path.read_text(encoding="utf-8")
    detector = _label_field(text, "detector") or "coherence"

    if detector == "neutrality":
        term = _label_field(text, "term")
        assert term, f"{path.name} neutrality mutation missing a `term=` label"
        assert term.lower() in text.lower(), f"{path.name} missing its forbidden term"
        assert term.lower() in _NEUTRALITY_FORBIDDEN_TERMS, (
            f"{term!r} is not a known forbidden tech-stack term"
        )
        # The coherence guard must NOT flag a neutrality-only violation.
        _code, result = _run_guard_over(path)
        assert not {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
        return

    expected_check = _label_field(text, "check")
    level = (_label_field(text, "level") or "FAIL").upper()
    assert expected_check, f"{path.name} missing a `check=` label"
    source = _MINI_SOURCE_REPORT if "needs-source-report" in text else None

    _code, result = _run_guard_over(path, source)
    hits = {f["check"] for f in result["findings"] if f["level"] == level}
    assert expected_check in hits, (
        path.name,
        expected_check,
        level,
        result["findings"],
    )


# --- issue #92: recompute self-check against the body (deterministic vendor /
# mechanism leak scan) ---
#
# The blueprint self-check (Appendix A) recorded each gate's status as a
# generation-time narrated verdict that was never recomputed against the final
# body, so an implementation-neutrality row could read ``PASS`` while the body
# carried named deployment products, vendor CLIs, and wire-level config flags
# (``Claude Code`` ``permissions.deny``, ``Codex`` ``execpolicy``,
# ``--available-tools``, ``SKILL.md``). The deterministic guard now re-derives
# the neutrality verdict from a fresh body scan so the Appendix A row is a
# computed result, not a stale generation-time claim.


def _vendor_leak_blueprint(body_line: str) -> str:
    """A minimal, otherwise-coherent blueprint carrying one extra body line."""
    return f"""# Product Blueprint

## Contents

- [1. Executive Product Thesis](#1-executive-product-thesis)

---

## 1. Executive Product Thesis

Gated admission over durable agent memory. [2312.01234]

{body_line}

<!-- coherence: id=wf1.gate stage=MVP-0 requires=wf1.servicer -->
<!-- coherence: id=wf1.servicer stage=MVP-0 -->
"""


def test_coherence_guard_fails_on_vendor_cli_and_config_leak(tmp_path: Path) -> None:
    """A named vendor CLI plus its wire-level config flags FAIL the body scan."""
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        _vendor_leak_blueprint(
            "Gate egress by configuring Claude Code `permissions.deny` and "
            "`--sandbox`; the Codex `execpolicy` mirrors it."
        ),
        encoding="utf-8",
    )
    code, result = _run_guard_over(blueprint)
    assert code == 1
    assert result["all_passed"] is False
    fails = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert "vendor_leak" in fails, result["findings"]


def test_coherence_guard_fails_on_skill_infra_token_leak(tmp_path: Path) -> None:
    """Skill-infrastructure tokens (``SKILL.md``, ``allowed-tools``) also leak."""
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        _vendor_leak_blueprint(
            "Register the capability in `SKILL.md` under `allowed-tools`."
        ),
        encoding="utf-8",
    )
    code, result = _run_guard_over(blueprint)
    assert code == 1
    fails = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert "vendor_leak" in fails, result["findings"]


def test_coherence_guard_allows_vendor_name_as_cited_research_anchor(
    tmp_path: Path,
) -> None:
    """A vendor name on a line that cites a paper is an allowed research anchor.

    ``Codex`` is a legitimate research subject; a cited line is a
    research-evaluation anchor, not an implementation leak.
    """
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        _vendor_leak_blueprint(
            "The Codex code model established the code-generation baseline "
            "[2107.03374]."
        ),
        encoding="utf-8",
    )
    code, result = _run_guard_over(blueprint)
    fails = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert "vendor_leak" not in fails, result["findings"]
    assert code == 0, result["findings"]


def test_golden_and_weak_fixtures_have_no_vendor_leak() -> None:
    """The shipped exemplars must not themselves trip the neutrality body scan."""
    for fixture in (_GOLDEN_EXAMPLE, _WEAK_OUTPUT):
        _code, result = _run_guard_over(fixture)
        fails = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
        assert "vendor_leak" not in fails, (fixture.name, result["findings"])


def test_quality_gate_recomputes_self_check_against_body() -> None:
    """Gate 0 scans for vendor leaks; Appendix A rows are recomputed, not restated."""
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    # The deterministic vendor/mechanism leak scan is wired into the Gate 0
    # pre-gate alongside the phase-inversion checks.
    assert "vendor_leak" in gate
    # Appendix A neutrality / traceability / MVP-discipline rows are the
    # post-repair body-scan result, never a restated generation-time claim.
    for needle in (
        "post-repair body-scan result",
        "never a restated generation-time",
        "recomputed",
    ):
        assert needle in gate, f"quality gate missing recompute rule: {needle}"


# --- issue #93: strengthen the coherence guard ---
#
# The deterministic guard was toothless on under-anchored blueprints: a document
# full of MVP-0/MVP-1 staging prose but zero anchors emitted only a WARNING, so
# the phase-inversion graph never ran. And the graph saw only ``requires`` edges,
# missing consumed-signal inversions and orphan references (a consumed signal
# whose producer is undeclared or staged later). These tests pin: (1) an
# anchor-coverage FAIL when staging language is present but no anchors exist,
# (2) an Open-Questions coverage FAIL, (3) a ``consumes=`` signal edge with a
# phase check, and (4) orphan-reference resolution for consumed signals.


def test_coherence_guard_fails_on_missing_anchors_when_staging_present(
    tmp_path: Path,
) -> None:
    """MVP-staging prose with zero anchors is a FAIL, not a silent WARNING."""
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        """# Product Blueprint

## Contents

- [1. MVP Scope](#1-mvp-scope)

---

## 1. MVP Scope

The admission gate is a non-negotiable MVP-0 control; scoped promotion is
deferred to MVP-1.
""",
        encoding="utf-8",
    )
    code, result = _run_guard_over(blueprint)
    assert code == 1
    fails = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert "missing_coherence_anchors" in fails, result["findings"]


def test_coherence_guard_warns_not_fails_when_no_staging_and_no_anchors(
    tmp_path: Path,
) -> None:
    """A doc with neither staging language nor anchors stays a WARNING."""
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        """# Product Blueprint

## Contents

- [1. Executive Product Thesis](#1-executive-product-thesis)

---

## 1. Executive Product Thesis

A conceptual overview with no phased commitments.
""",
        encoding="utf-8",
    )
    code, result = _run_guard_over(blueprint)
    warnings = {f["check"] for f in result["findings"] if f["level"] == "WARNING"}
    fails = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert "no_coherence_anchors" in warnings
    assert "missing_coherence_anchors" not in fails
    assert code == 0


def test_coherence_guard_fails_on_unanchored_open_questions(tmp_path: Path) -> None:
    """An Open Questions section without a stage=open anchor FAILs coverage."""
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        """# Product Blueprint

## Contents

- [1. MVP Scope](#1-mvp-scope)
- [2. Open Questions and Validation Plan](#2-open-questions-and-validation-plan)

---

## 1. MVP Scope

<!-- coherence: id=mvp0.core stage=MVP-0 -->

The MVP-0 core is a gated admission path.

## 2. Open Questions and Validation Plan

| Question | Blocks MVP? |
|---|---|
| Optimal dedup similarity threshold | No |
""",
        encoding="utf-8",
    )
    code, result = _run_guard_over(blueprint)
    assert code == 1
    fails = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert "unanchored_open_questions" in fails, result["findings"]


def test_coherence_guard_fails_on_signal_inversion_via_consumes(
    tmp_path: Path,
) -> None:
    """An MVP-0 node consuming a signal produced at MVP-1 is a signal inversion."""
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        """# Product Blueprint

## Contents

- [1. MVP Scope](#1-mvp-scope)

---

## 1. MVP Scope

<!-- coherence: id=mvp0.gate stage=MVP-0 consumes=sig.debiased -->
<!-- coherence: id=sig.debiased stage=MVP-1 -->

The MVP-0 gate consumes a de-biased segment signal.
""",
        encoding="utf-8",
    )
    code, result = _run_guard_over(blueprint)
    assert code == 1
    fails = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert "signal_inversion" in fails, result["findings"]


def test_coherence_guard_accepts_consumes_from_earlier_stage(tmp_path: Path) -> None:
    """An MVP-1 node consuming an MVP-0 signal is phase-clean."""
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        """# Product Blueprint

## Contents

- [1. MVP Scope](#1-mvp-scope)

---

## 1. MVP Scope

<!-- coherence: id=mvp1.review stage=MVP-1 consumes=sig.quarantine -->
<!-- coherence: id=sig.quarantine stage=MVP-0 -->

The MVP-1 review consumes the MVP-0 quarantine signal.
""",
        encoding="utf-8",
    )
    code, result = _run_guard_over(blueprint)
    fails = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert "signal_inversion" not in fails, result["findings"]
    assert code == 0, result["findings"]


def test_coherence_guard_fails_on_orphan_consumes_reference(tmp_path: Path) -> None:
    """A consumed signal whose producer is undeclared is an orphan reference."""
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        """# Product Blueprint

## Contents

- [1. MVP Scope](#1-mvp-scope)

---

## 1. MVP Scope

<!-- coherence: id=mvp0.gate stage=MVP-0 consumes=sig.missing -->

The MVP-0 gate consumes a signal no producer registers.
""",
        encoding="utf-8",
    )
    code, result = _run_guard_over(blueprint)
    assert code == 1
    fails = {f["check"] for f in result["findings"] if f["level"] == "FAIL"}
    assert "orphan_reference" in fails, result["findings"]


def test_golden_example_models_full_anchor_coverage() -> None:
    """The golden exemplar demonstrates consumes edges + capability registration.

    Issue #93 requires the golden fixture to carry full anchor coverage so it
    still passes the strengthened guard: a ``consumes=`` signal edge and
    capability/object registration anchors that §12/§15 references resolve to.
    """
    text = _GOLDEN_EXAMPLE.read_text(encoding="utf-8")
    assert "consumes=" in text, "golden fixture models no consumes edge"
    assert re.search(r"id=cap\.", text), "golden fixture registers no capabilities"
    code, result = _run_guard_over(_GOLDEN_EXAMPLE)
    assert code == 0, result["findings"]
    assert result["all_passed"] is True


def test_compose_prompt_mandates_full_anchor_coverage() -> None:
    prompt = (_skill_root() / "prompts" / "04_generate_blueprint.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "consumes=",
        "every staged",
        "consumed-signal producer",
        "open question",
    ):
        assert needle in prompt, (
            f"compose prompt missing anchor-coverage rule: {needle}"
        )


def test_template_documents_consumes_edge() -> None:
    template = (
        _skill_root() / "templates" / "product_blueprint_template.md"
    ).read_text(encoding="utf-8")
    assert "consumes=" in template, "template does not document the consumes edge"


# --- issue #94: altitude ceiling for agent/tool-surface sections ---
#
# Agent/tool-decomposed blueprints kept leaking mechanism (dedup keys,
# compaction, single-PEP, retry-bounding, transport) into the body and keying
# policy/MVP/eval to named tool identifiers rather than READ/ACT/AUTH authority
# classes. A dedup-key discussion names no vendor, so Gate 3 (vendors) passes it
# while it sits a full altitude below the blueprint. These tests pin a
# WARNING-tier mechanism-vocabulary linter (exempt inside the §18 handoff) and a
# named-tool-identifier linter over the policy/eval/MVP sections.


def test_coherence_guard_warns_on_mechanism_vocab_outside_handoff(
    tmp_path: Path,
) -> None:
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        """# Product Blueprint

## Contents

- [1. Logical Architecture](#1-logical-architecture)

---

## 1. Logical Architecture

A retried submit is deduplicated by a dedup key at a single PEP, with
wire-level idempotency and bounded transport retries.
""",
        encoding="utf-8",
    )
    _code, result = _run_guard_over(blueprint)
    warns = {f["check"] for f in result["findings"] if f["level"] == "WARNING"}
    assert "mechanism_altitude" in warns, result["findings"]


def test_coherence_guard_allows_mechanism_vocab_in_handoff(tmp_path: Path) -> None:
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        """# Product Blueprint

## Contents

- [1. Executive Product Thesis](#1-executive-product-thesis)
- [18. Handoff Notes for Technical Design](#18-handoff-notes-for-technical-design)

---

## 1. Executive Product Thesis

A trustworthy memory keeper that never double-egresses a retried submit.

## 18. Handoff Notes for Technical Design

Technical design owns the dedup key, compaction strategy, single-PEP locus,
retry-bounding, and transport.
""",
        encoding="utf-8",
    )
    _code, result = _run_guard_over(blueprint)
    warns = {f["check"] for f in result["findings"] if f["level"] == "WARNING"}
    assert "mechanism_altitude" not in warns, result["findings"]


def test_coherence_guard_warns_on_tool_identifier_in_policy_row(
    tmp_path: Path,
) -> None:
    blueprint = tmp_path / "blueprint.md"
    blueprint.write_text(
        """# Product Blueprint

## Contents

- [12. Decision Policies](#12-decision-policies)

---

## 12. Decision Policies

| Policy | Inputs | Default |
|---|---|---|
| Depth | `set_depth` and `get_risk_report` gate the ACT path | conservative |
""",
        encoding="utf-8",
    )
    _code, result = _run_guard_over(blueprint)
    warns = {f["check"] for f in result["findings"] if f["level"] == "WARNING"}
    assert "tool_identifier_altitude" in warns, result["findings"]


def test_golden_example_has_no_altitude_warnings() -> None:
    """The shipped exemplar stays above the mechanism/tool-identifier altitude."""
    _code, result = _run_guard_over(_GOLDEN_EXAMPLE)
    warns = {f["check"] for f in result["findings"] if f["level"] == "WARNING"}
    assert "mechanism_altitude" not in warns, result["findings"]
    assert "tool_identifier_altitude" not in warns, result["findings"]


def test_compose_prompt_has_tool_surface_altitude_ceiling() -> None:
    prompt = (_skill_root() / "prompts" / "04_generate_blueprint.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "authority class",
        "READ / ACT / AUTH",
        "invariant, not the realization",
        "example-tool map",
    ):
        assert needle in prompt, f"compose prompt missing altitude ceiling: {needle}"


def test_quality_gate_has_altitude_gate() -> None:
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    assert "mechanism_altitude" in gate
    assert "authority class" in gate


# --- issue #95: actor-channel completeness ---
#
# When a blueprint has more than one actor channel (interactive human, headless
# automation, non-human client) the skill did not enforce that EVERY escalation
# / accept / authorization path is defined for EACH channel. The recurring
# defect: a "human accepts residual risk" step silently assumes an inline
# interactive human, leaving a headless channel with an undefined authorization
# path. These tests pin the §3 channel enumeration, the §8 per-channel path
# requirement, and the completeness gate.


def test_template_enumerates_actor_channels() -> None:
    template = (
        _skill_root() / "templates" / "product_blueprint_template.md"
    ).read_text(encoding="utf-8")
    for needle in (
        "Channel Class",
        "interactive human",
        "headless automation",
        "non-human client",
    ):
        assert needle in template, f"template §3 missing channel class: {needle}"


def test_template_workflow_requires_per_channel_paths() -> None:
    template = (
        _skill_root() / "templates" / "product_blueprint_template.md"
    ).read_text(encoding="utf-8")
    assert "Authorization / Escalation Paths (per channel)" in template


def test_compose_prompt_requires_actor_channel_completeness() -> None:
    prompt = (_skill_root() / "prompts" / "04_generate_blueprint.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "actor channel",
        "interactive human / headless automation / non-human client",
        "out-of-band human",
        "inline human on a headless channel",
    ):
        assert needle in prompt, f"compose prompt missing channel rule: {needle}"


def test_quality_gate_enforces_actor_channel_completeness() -> None:
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "actor-channel completeness",
        "out-of-band human",
        "scoped off",
        "headless",
    ):
        assert needle in gate, (
            f"quality gate missing channel-completeness rule: {needle}"
        )


# --- issue #96: one authoritative home per decision + conservative MVP-0 default
#
# The template scattered each decision across the decisions/risk/MVP/routing
# tables and appendices, so one decision was restated ~12 times and a stale copy
# falsified a self-check. And an unvalidated, non-research-derived capability was
# recorded as the normative MVP-0 default instead of defaulting conservative and
# surfacing an owner gate — the classic build-trap default polarity. These tests
# pin the single-decision-home convention and the conservative MVP-0 default.


def test_template_has_single_decision_home_convention() -> None:
    template = (
        _skill_root() / "templates" / "product_blueprint_template.md"
    ).read_text(encoding="utf-8")
    for needle in (
        "one authoritative home",
        "stable ID",
        "reference it by ID",
        "one-line statement",
    ):
        assert needle in template, f"template missing decision-home rule: {needle}"


def test_compose_prompt_has_conservative_mvp0_default() -> None:
    prompt = (_skill_root() / "prompts" / "04_generate_blueprint.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "unvalidated",
        "surface-agnostic",
        "default it OUT of MVP-0",
        "wizard-of-oz",
        "owner gate",
    ):
        assert needle in prompt, f"compose prompt missing MVP-0 default rule: {needle}"


def test_quality_gate_warns_on_unvalidated_mvp0_default() -> None:
    gate = (_skill_root() / "prompts" / "05_quality_gate.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        "unvalidated capability is the MVP-0 default",
        "gating test",
    ):
        assert needle in gate, f"quality gate missing MVP-0 default warning: {needle}"
