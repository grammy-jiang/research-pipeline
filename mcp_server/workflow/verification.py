"""Structural verification of pipeline stage outputs.

Implements the external verification pattern (CVA): stage outputs are
validated by structural checks (file existence, size, schema conformance),
NOT by a second LLM call. Self-referential verification causes
polarization (more LLM rounds ≠ better judgment).

Each verifier is deterministic, fast, and independently testable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VerificationResult:
    """Result of a structural verification check."""

    def __init__(
        self, passed: bool, details: str, checks: dict[str, bool] | None = None
    ) -> None:
        self.passed = passed
        self.details = details
        self.checks = checks or {}

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"VerificationResult({status}: {self.details})"


def _run_dir(workspace: str, run_id: str) -> Path:
    """Resolve the run directory."""
    return Path(workspace) / run_id


def verify_plan(workspace: str, run_id: str) -> VerificationResult:
    """Verify plan stage output: query_plan.json exists with required fields."""
    checks: dict[str, bool] = {}
    run_path = _run_dir(workspace, run_id) / "plan"

    plan_file = run_path / "query_plan.json"
    checks["plan_file_exists"] = plan_file.exists()
    if not plan_file.exists():
        return VerificationResult(False, "query_plan.json not found", checks)

    try:
        plan = json.loads(plan_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        checks["plan_parseable"] = False
        return VerificationResult(False, f"query_plan.json parse error: {exc}", checks)

    checks["plan_parseable"] = True
    checks["has_must_terms"] = bool(plan.get("must_terms"))
    checks["has_query_variants"] = len(plan.get("query_variants", [])) >= 2

    all_passed = all(checks.values())
    return VerificationResult(
        all_passed,
        f"Plan verification: {sum(checks.values())}/{len(checks)} checks passed",
        checks,
    )


def verify_search(workspace: str, run_id: str) -> VerificationResult:
    """Verify search stage: candidates exist with required fields."""
    checks: dict[str, bool] = {}
    run_path = _run_dir(workspace, run_id) / "search"

    candidates_file = run_path / "candidates.jsonl"
    checks["candidates_file_exists"] = candidates_file.exists()
    if not candidates_file.exists():
        return VerificationResult(False, "candidates.jsonl not found", checks)

    try:
        lines = [
            line.strip()
            for line in candidates_file.read_text().splitlines()
            if line.strip()
        ]
        candidates = [json.loads(line) for line in lines]
    except (json.JSONDecodeError, OSError) as exc:
        checks["candidates_parseable"] = False
        return VerificationResult(False, f"candidates.jsonl parse error: {exc}", checks)

    checks["candidates_parseable"] = True
    checks["has_candidates"] = len(candidates) >= 1
    checks["all_have_paper_id"] = all(c.get("paper_id") for c in candidates)
    checks["all_have_title"] = all(c.get("title") for c in candidates)

    all_passed = all(checks.values())
    return VerificationResult(
        all_passed,
        f"Search verification: {len(candidates)} candidates, "
        f"{sum(checks.values())}/{len(checks)} checks passed",
        checks,
    )


def verify_screen(workspace: str, run_id: str) -> VerificationResult:
    """Verify screen stage: shortlist exists with scored papers."""
    checks: dict[str, bool] = {}
    run_path = _run_dir(workspace, run_id) / "screen"

    shortlist_file = run_path / "shortlist.json"
    checks["shortlist_file_exists"] = shortlist_file.exists()
    if not shortlist_file.exists():
        return VerificationResult(False, "shortlist.json not found", checks)

    try:
        shortlist = json.loads(shortlist_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        checks["shortlist_parseable"] = False
        return VerificationResult(False, f"shortlist.json parse error: {exc}", checks)

    checks["shortlist_parseable"] = True
    papers = shortlist if isinstance(shortlist, list) else shortlist.get("papers", [])
    checks["has_papers"] = len(papers) >= 1

    all_passed = all(checks.values())
    return VerificationResult(
        all_passed,
        f"Screen verification: {len(papers)} papers shortlisted, "
        f"{sum(checks.values())}/{len(checks)} checks passed",
        checks,
    )


def verify_download(workspace: str, run_id: str) -> VerificationResult:
    """Verify download stage: PDFs exist and are non-trivial."""
    checks: dict[str, bool] = {}
    run_path = _run_dir(workspace, run_id) / "download" / "pdf"

    checks["pdf_dir_exists"] = run_path.is_dir()
    if not run_path.is_dir():
        return VerificationResult(False, "download/pdf directory not found", checks)

    pdfs = list(run_path.glob("*.pdf"))
    checks["has_pdfs"] = len(pdfs) >= 1
    checks["all_non_trivial"] = all(p.stat().st_size >= 10_000 for p in pdfs)

    all_passed = all(checks.values())
    return VerificationResult(
        all_passed,
        f"Download verification: {len(pdfs)} PDFs, "
        f"{sum(checks.values())}/{len(checks)} checks passed",
        checks,
    )


def verify_convert(workspace: str, run_id: str) -> VerificationResult:
    """Verify convert stage: Markdown files exist and are non-empty."""
    checks: dict[str, bool] = {}

    # Check both rough and fine conversion directories
    run_path = _run_dir(workspace, run_id)
    md_dirs = [
        run_path / "convert" / "markdown",
        run_path / "convert_rough" / "markdown",
        run_path / "convert_fine" / "markdown",
    ]

    md_files: list[Path] = []
    for md_dir in md_dirs:
        if md_dir.is_dir():
            md_files.extend(md_dir.glob("*.md"))

    checks["has_markdown_files"] = len(md_files) >= 1
    if not md_files:
        return VerificationResult(False, "No markdown files found", checks)

    checks["all_non_empty"] = all(f.stat().st_size >= 500 for f in md_files)

    all_passed = all(checks.values())
    return VerificationResult(
        all_passed,
        f"Convert verification: {len(md_files)} markdown files, "
        f"{sum(checks.values())}/{len(checks)} checks passed",
        checks,
    )


def verify_extract(workspace: str, run_id: str) -> VerificationResult:
    """Verify extract stage: extraction manifest exists."""
    checks: dict[str, bool] = {}
    run_path = _run_dir(workspace, run_id) / "extract"

    checks["extract_dir_exists"] = run_path.is_dir()
    if not run_path.is_dir():
        return VerificationResult(False, "extract directory not found", checks)

    # Check for any extraction output files
    extract_files = list(run_path.iterdir()) if run_path.is_dir() else []
    checks["has_extraction_output"] = len(extract_files) >= 1

    all_passed = all(checks.values())
    return VerificationResult(
        all_passed,
        f"Extract verification: {len(extract_files)} output files, "
        f"{sum(checks.values())}/{len(checks)} checks passed",
        checks,
    )


def verify_analyze(
    analyses: list[dict[str, Any]],
    expected_count: int,
) -> VerificationResult:
    """Verify analyze stage: analyses have required structure.

    Unlike other verifiers, this checks in-memory data since analyses
    come from sampling, not from disk.
    """
    checks: dict[str, bool] = {}

    checks["has_analyses"] = len(analyses) >= 1
    checks["count_matches"] = len(analyses) >= expected_count

    if analyses:
        checks["all_have_rating"] = all(
            isinstance(a.get("rating"), int | float) and 1 <= a.get("rating", 0) <= 5
            for a in analyses
        )
        checks["all_have_findings"] = all(
            len(a.get("findings", [])) >= 1 for a in analyses
        )
    else:
        checks["all_have_rating"] = False
        checks["all_have_findings"] = False

    all_passed = all(checks.values())
    return VerificationResult(
        all_passed,
        f"Analyze verification: {len(analyses)}/{expected_count} papers, "
        f"{sum(checks.values())}/{len(checks)} checks passed",
        checks,
    )


def verify_synthesize(synthesis: dict[str, Any]) -> VerificationResult:
    """Verify synthesize stage: synthesis has required structure."""
    checks: dict[str, bool] = {}

    checks["has_themes"] = len(synthesis.get("themes", [])) >= 1
    checks["has_readiness"] = bool(synthesis.get("readiness_verdict"))

    all_passed = all(checks.values())
    return VerificationResult(
        all_passed,
        f"Synthesize verification: "
        f"{sum(checks.values())}/{len(checks)} checks passed",
        checks,
    )


# Registry of stage verifiers for disk-based stages
STAGE_VERIFIERS: dict[str, Any] = {
    "plan": verify_plan,
    "search": verify_search,
    "screen": verify_screen,
    "download": verify_download,
    "convert": verify_convert,
    "extract": verify_extract,
}
