"""CLI handler for the 'validate' command.

Checks a final research report for completeness against the required
template sections, confidence-level annotations, evidence citations,
and gap classifications.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from research_pipeline.quality.fact_scoring import compute_fact_score
from research_pipeline.quality.race_scoring import compute_race_score

logger = logging.getLogger(__name__)

# Core required section headings (always required, case-insensitive check)
REQUIRED_SECTIONS = [
    "executive summary",
    "research question",
    "methodology",
    "papers reviewed",
    "research landscape",
    "research gaps",
    "practical recommendations",
    "references",
    "appendix",
]

STRUCTURED_REQUIRED_SECTIONS = [
    "executive summary",
    "scope and corpus",
    "methodology",
    "taxonomy of approaches",
    "evidence matrix",
    "recurring mechanisms and patterns",
    "assumption map",
    "contradiction map",
    "evidence strength map",
    "operational implications",
    "production readiness",
    "reusable mechanism inventory",
    "design implications",
    "unresolved questions",
    "risk register",
    "traceability appendix",
]

# Conditional sections (include when evidence justifies them)
CONDITIONAL_SECTIONS = [
    "methodology comparison",
    "confidence-graded findings",
    "trade-off analysis",
    "points of agreement",
    "points of contradiction",
    "reproducibility notes",
    "evidence map",
    "readiness assessment",
    "future directions",
]

# Optional but expected sections (not validated, just noted)
OPTIONAL_SECTIONS = [
    "prior run comparison",
]

CONFIDENCE_PATTERN = re.compile(
    r"(🟢|🟡|🔴|high confidence|medium confidence|low confidence"
    r"|confidence.*?:.*?(high|medium|low))",
    re.IGNORECASE,
)

EVIDENCE_CITATION_PATTERN = re.compile(
    r"(\[[\w.,\-\s]+\]|\bE\d{3}\b|\b[A-Z]{2,5}-\d{3}\b)"
)

GAP_TYPE_PATTERN = re.compile(r"(ACADEMIC|ENGINEERING)", re.IGNORECASE)

GAP_SEVERITY_PATTERN = re.compile(r"(HIGH|MEDIUM|LOW)", re.IGNORECASE)


def _extract_headings(text: str) -> list[str]:
    """Extract markdown headings from text."""
    headings = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            # Remove # characters and clean up
            heading = stripped.lstrip("#").strip()
            headings.append(heading.lower())
    return headings


def _check_sections(
    text: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Check for required, conditional, and optional sections.

    Returns:
        Tuple of (present, missing_required, present_conditional, missing_optional).
    """
    headings = _extract_headings(text)
    present = []
    missing_required = []
    present_conditional = []
    missing_optional = []

    for section in REQUIRED_SECTIONS:
        found = any(section in h for h in headings)
        if found:
            present.append(section)
        else:
            missing_required.append(section)

    for section in CONDITIONAL_SECTIONS:
        found = any(section in h for h in headings)
        if found:
            present_conditional.append(section)

    for section in OPTIONAL_SECTIONS:
        found = any(section in h for h in headings)
        if not found:
            missing_optional.append(section)

    return present, missing_required, present_conditional, missing_optional


def _check_structured_sections(text: str) -> tuple[list[str], list[str]]:
    """Check sections for the structured Step 2 synthesis template."""
    headings = _extract_headings(text)
    present = []
    missing = []
    for section in STRUCTURED_REQUIRED_SECTIONS:
        found = any(section in h for h in headings)
        if found:
            present.append(section)
        else:
            missing.append(section)
    return present, missing


def _check_confidence_levels(text: str) -> dict[str, int]:
    """Count confidence level annotations in the report."""
    counts = {"high": 0, "medium": 0, "low": 0, "emoji": 0}

    for line in text.splitlines():
        if "🟢" in line:
            counts["emoji"] += 1
            counts["high"] += 1
        if "🟡" in line:
            counts["emoji"] += 1
            counts["medium"] += 1
        if "🔴" in line:
            counts["emoji"] += 1
            counts["low"] += 1

    # Also count text-based confidence mentions
    text_lower = text.lower()
    counts["high"] += len(re.findall(r"confidence.*?:.*?high", text_lower))
    counts["medium"] += len(re.findall(r"confidence.*?:.*?medium", text_lower))
    counts["low"] += len(re.findall(r"confidence.*?:.*?low", text_lower))

    return counts


def _check_evidence_citations(text: str) -> int:
    """Count evidence citations [arxiv_id] in the report."""
    return len(EVIDENCE_CITATION_PATTERN.findall(text))


def _check_gap_classification(text: str) -> dict[str, int]:
    """Check for gap type and severity classification."""
    return {
        "academic_gaps": len(re.findall(r"\bACADEMIC\b", text)),
        "engineering_gaps": len(re.findall(r"\bENGINEERING\b", text)),
        "high_severity": len(
            re.findall(
                r"(?:severity|HIGH).*?HIGH|HIGH.*?(?:severity)", text, re.IGNORECASE
            )
        ),
    }


def _check_tables(text: str) -> int:
    """Count markdown tables in the report."""
    table_count = 0
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            if not in_table:
                table_count += 1
                in_table = True
        else:
            in_table = False
    return table_count


def _check_mermaid(text: str) -> int:
    """Count mermaid diagrams."""
    return len(re.findall(r"```mermaid", text))


def _check_contents(text: str) -> bool:
    """Check for a top-level ``## Contents`` table-of-contents section.

    The section must contain at least two Markdown links so a single empty
    heading does not count.
    """
    match = re.search(
        r"^##\s+contents\s*$(.*?)(?=^##\s|\Z)",
        text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not match:
        return False
    body = match.group(1)
    return len(re.findall(r"\[[^\]]+\]\([^)]+\)", body)) >= 2


def _check_latex(text: str) -> int:
    """Count LaTeX formulas."""
    inline = len(re.findall(r"(?<!\$)\$(?!\$).+?\$(?!\$)", text))
    display = len(re.findall(r"\$\$.+?\$\$", text, re.DOTALL))
    return inline + display


def validate_report(
    report_path: Path,
    paper_ids: list[str] | None = None,
    paper_titles: list[str] | None = None,
) -> dict[str, Any]:
    """Validate a research report against template requirements.

    Args:
        report_path: Path to the markdown report file.
        paper_ids: Known paper IDs for FACT citation verification.
        paper_titles: Known paper titles for FACT citation verification.

    Returns:
        Validation result dict with findings and score.
    """
    text = report_path.read_text()

    present, missing_required, present_conditional, missing_optional = _check_sections(
        text
    )
    structured_present, structured_missing = _check_structured_sections(text)
    confidence_counts = _check_confidence_levels(text)
    citation_count = _check_evidence_citations(text)
    gap_class = _check_gap_classification(text)
    table_count = _check_tables(text)
    mermaid_count = _check_mermaid(text)
    latex_count = _check_latex(text)
    has_contents = _check_contents(text)

    # Calculate completeness score (core sections + bonus for conditional).
    # Reports may use either the legacy 14-section shape or the structured
    # Step 2 synthesis shape.
    legacy_core_score = len(present) / max(len(REQUIRED_SECTIONS), 1)
    structured_core_score = len(structured_present) / max(
        len(STRUCTURED_REQUIRED_SECTIONS), 1
    )
    structured_mode = structured_core_score > legacy_core_score
    core_section_score = max(legacy_core_score, structured_core_score)
    conditional_bonus = (
        len(present_conditional) / max(len(CONDITIONAL_SECTIONS), 1) * 0.2
    )
    section_score = min(core_section_score + conditional_bonus, 1.0)

    has_confidence = (
        confidence_counts["high"]
        + confidence_counts["medium"]
        + confidence_counts["low"]
    ) > 0
    has_citations = citation_count >= 3
    has_tables = table_count >= 2
    has_gaps = (gap_class["academic_gaps"] + gap_class["engineering_gaps"]) > 0
    has_assumption_and_contradiction = (
        "assumption map" in structured_present
        and "contradiction map" in structured_present
    )

    quality_checks = {
        "has_confidence_levels": has_confidence,
        "has_evidence_citations": has_citations,
        "has_tables": has_tables,
        "has_gap_classification": has_gaps or has_assumption_and_contradiction,
        "has_mermaid_diagram": mermaid_count > 0 or structured_mode,
        "has_latex_formula": latex_count > 0,
        "has_table_of_contents": has_contents,
    }

    quality_score = sum(1 for v in quality_checks.values() if v) / len(quality_checks)

    overall_score = round(0.6 * section_score + 0.4 * quality_score, 2)

    issues = []
    effective_missing = structured_missing if structured_mode else missing_required
    if effective_missing:
        issues.append(
            f"Missing {len(effective_missing)} required section(s): "
            + ", ".join(effective_missing)
        )
    if not has_confidence:
        issues.append("No confidence-level annotations found")
    if not has_citations:
        issues.append(
            f"Insufficient evidence citations ({citation_count} found, need ≥3)"
        )
    if not has_gaps and not has_assumption_and_contradiction:
        issues.append("No gap classification (ACADEMIC/ENGINEERING) found")
    if mermaid_count == 0 and not structured_mode:
        issues.append("No Mermaid diagrams found (required for methodology)")
    if not has_contents:
        issues.append("No '## Contents' table-of-contents section found")
    if latex_count == 0:
        issues.append("No LaTeX formulas found (required for quantitative claims)")
    if table_count < 2:
        issues.append(f"Too few tables ({table_count} found, need ≥2)")

    verdict = "PASS" if overall_score >= 0.7 and not effective_missing else "FAIL"

    # RACE report quality scoring (additive)
    race = compute_race_score(text)

    # FACT citation verification (when paper corpus is available)
    fact_result = None
    if paper_ids:
        fact = compute_fact_score(
            text,
            paper_ids=paper_ids,
            paper_titles=paper_titles or [],
        )
        fact_result = fact.model_dump()
        if fact.citation_accuracy < 0.5:
            issues.append(
                f"Low citation accuracy ({fact.citation_accuracy:.0%}) — "
                f"{len(fact.unsupported_citations)} unsupported citation(s)"
            )
        if fact.effective_citation_ratio < 0.3:
            issues.append(
                f"Low citation coverage ({fact.effective_citation_ratio:.0%}) — "
                f"{len(fact.uncited_papers)} paper(s) never cited"
            )

    result: dict[str, Any] = {
        "report_path": str(report_path),
        "verdict": verdict,
        "overall_score": overall_score,
        "section_score": round(section_score, 2),
        "quality_score": round(quality_score, 2),
        "sections": {
            "present": present,
            "missing_required": effective_missing,
            "present_conditional": present_conditional,
            "missing_optional": missing_optional,
            "structured_present": structured_present,
        },
        "confidence_levels": confidence_counts,
        "evidence_citations": citation_count,
        "gap_classification": gap_class,
        "formatting": {
            "tables": table_count,
            "mermaid_diagrams": mermaid_count,
            "latex_formulas": latex_count,
        },
        "quality_checks": quality_checks,
        "issues": issues,
        "race_score": race.model_dump(),
    }
    if fact_result is not None:
        result["fact_score"] = fact_result
    return result


def run_validate(
    report: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    output: Path | None = None,
) -> None:
    """Validate a research report for completeness and quality.

    When --run-id is provided, also loads paper IDs and titles from the
    run's shortlist for FACT citation verification.

    Args:
        report: Path to the report markdown file.
        workspace: Workspace root (used with run_id to find synthesis).
        run_id: Pipeline run ID (to find synthesis_report.md).
        output: Path to write validation JSON report.
    """
    report_path = report
    paper_ids: list[str] = []
    paper_titles: list[str] = []

    # If no direct report path, try to find synthesis in run dir
    if run_id and workspace:
        from research_pipeline.config.loader import load_config
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        config = load_config()
        ws = workspace or Path(config.workspace)
        _run_id, run_root = init_run(ws, run_id)

        if report_path is None:
            synth_dir = get_stage_dir(run_root, "summarize")
            candidates = [
                synth_dir / "synthesis_report.md",
                run_root / "synthesis" / "synthesis_report.md",
            ]
            for candidate in candidates:
                if candidate.exists():
                    report_path = candidate
                    break

        # Load paper IDs/titles from shortlist for FACT verification
        shortlist_path = get_stage_dir(run_root, "screen") / "shortlist.json"
        if shortlist_path.exists():
            try:
                raw_sl = json.loads(shortlist_path.read_text(encoding="utf-8"))
                for entry in raw_sl:
                    paper = entry.get("paper", entry)
                    aid = paper.get("arxiv_id", "")
                    title = paper.get("title", "")
                    if aid:
                        paper_ids.append(aid)
                    if title:
                        paper_titles.append(title)
                logger.info("Loaded %d paper IDs for FACT verification", len(paper_ids))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load shortlist for FACT: %s", exc)

    if report_path is None or not report_path.exists():
        logger.error("No report found. Use --report PATH or --run-id with --workspace.")
        return

    logger.info("Validating report: %s", report_path)
    result = validate_report(
        report_path,
        paper_ids=paper_ids or None,
        paper_titles=paper_titles or None,
    )

    # Log summary
    verdict = result["verdict"]
    score = result["overall_score"]
    issues = result["issues"]
    logger.info("Verdict: %s (score: %.2f)", verdict, score)

    if issues:
        for issue in issues:
            logger.warning("Issue: %s", issue)
    else:
        logger.info("No issues found — report meets all requirements.")

    # Log FACT results if present
    if "fact_score" in result:
        fact = result["fact_score"]
        logger.info(
            "FACT: accuracy=%.2f, coverage=%.2f, verified=%d/%d",
            fact["citation_accuracy"],
            fact["effective_citation_ratio"],
            fact["verified_citations"],
            fact["total_citations"],
        )

    # Write output
    if output:
        output.write_text(json.dumps(result, indent=2))
        logger.info("Validation report written to %s", output)
    else:
        # Write next to the report
        out_path = report_path.parent / "validation_result.json"
        out_path.write_text(json.dumps(result, indent=2))
        logger.info("Validation report written to %s", out_path)
