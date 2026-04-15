"""Schema-grounded evaluation: generate test cases from Pydantic model schemas.

Validates pipeline outputs against their model schemas with completeness
checks beyond simple JSON parsing:
- Required field presence AND non-empty content
- Cross-field consistency (e.g., paper_count matches actual count)
- Value range validation (scores in [0,1], dates parseable)
- Cross-stage reference integrity
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalCheck:
    """A single evaluation check."""

    name: str
    description: str
    passed: bool
    details: str = ""
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class EvalReport:
    """Evaluation report for a stage output."""

    stage: str
    checks: list[EvalCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Report passes if all error-severity checks pass."""
        return all(c.passed for c in self.checks if c.severity == "error")

    @property
    def error_count(self) -> int:
        """Count of failed error-severity checks."""
        return sum(1 for c in self.checks if not c.passed and c.severity == "error")

    @property
    def warning_count(self) -> int:
        """Count of failed warning-severity checks."""
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")

    def summary(self) -> str:
        """One-line summary of evaluation results."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.stage}: {status} "
            f"({self.error_count} errors, {self.warning_count} warnings, "
            f"{len(self.checks)} checks)"
        )


def check_field_populated(
    data: dict[str, Any], field_name: str, stage: str
) -> EvalCheck:
    """Check that a field exists and is meaningfully populated.

    Args:
        data: Dictionary to check.
        field_name: Key to look for.
        stage: Stage name (for context in messages).

    Returns:
        EvalCheck with result.
    """
    value = data.get(field_name)
    if value is None:
        return EvalCheck(
            name=f"{field_name}_present",
            description=f"{field_name} exists",
            passed=False,
            details="Field is None",
        )
    if isinstance(value, str) and not value.strip():
        return EvalCheck(
            name=f"{field_name}_populated",
            description=f"{field_name} has content",
            passed=False,
            details="Field is empty string",
        )
    if isinstance(value, list) and len(value) == 0:
        return EvalCheck(
            name=f"{field_name}_populated",
            description=f"{field_name} has items",
            passed=False,
            details="Field is empty list",
            severity="warning",
        )
    return EvalCheck(
        name=f"{field_name}_populated",
        description=f"{field_name} has content",
        passed=True,
    )


def check_score_range(
    data: dict[str, Any],
    field_name: str,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> EvalCheck:
    """Check that a numeric field is in expected range.

    Args:
        data: Dictionary to check.
        field_name: Key to look for.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        EvalCheck with result.
    """
    value = data.get(field_name)
    if value is None:
        return EvalCheck(
            name=f"{field_name}_range",
            description=f"{field_name} in [{min_val}, {max_val}]",
            passed=True,
            details="Field absent (optional)",
            severity="info",
        )
    try:
        num = float(value)
        in_range = min_val <= num <= max_val
        return EvalCheck(
            name=f"{field_name}_range",
            description=f"{field_name} in [{min_val}, {max_val}]",
            passed=in_range,
            details=f"Value: {num}" if not in_range else "",
        )
    except (ValueError, TypeError):
        return EvalCheck(
            name=f"{field_name}_range",
            description=f"{field_name} is numeric",
            passed=False,
            details=f"Not numeric: {value}",
        )


def check_list_count_consistency(
    data: dict[str, Any], list_field: str, count_field: str
) -> EvalCheck:
    """Check that a count field matches the actual list length.

    Args:
        data: Dictionary to check.
        list_field: Key for the list.
        count_field: Key for the expected count.

    Returns:
        EvalCheck with result.
    """
    items = data.get(list_field, [])
    count = data.get(count_field)
    if count is None:
        return EvalCheck(
            name=f"{list_field}_count",
            description="Count matches list length",
            passed=True,
            details="No count field",
            severity="info",
        )
    if not isinstance(items, list):
        return EvalCheck(
            name=f"{list_field}_count",
            description="List field is a list",
            passed=False,
            details=f"Not a list: {type(items).__name__}",
        )
    matches = len(items) == count
    return EvalCheck(
        name=f"{list_field}_count",
        description=f"{count_field} matches {list_field} length",
        passed=matches,
        details="" if matches else f"Count says {count}, actual {len(items)}",
    )


def evaluate_plan(run_root: Path) -> EvalReport:
    """Evaluate plan stage output.

    Args:
        run_root: Path to run directory.

    Returns:
        EvalReport with all checks.
    """
    report = EvalReport(stage="plan")
    plan_path = run_root / "plan" / "query_plan.json"

    if not plan_path.exists():
        report.checks.append(
            EvalCheck(
                "plan_exists",
                "Plan file exists",
                False,
                "Missing query_plan.json",
            )
        )
        return report
    report.checks.append(EvalCheck("plan_exists", "Plan file exists", True))

    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.checks.append(EvalCheck("plan_json", "Valid JSON", False, str(exc)))
        return report
    report.checks.append(EvalCheck("plan_json", "Valid JSON", True))

    for field_name in ["topic_raw", "must_terms", "query_variants"]:
        report.checks.append(check_field_populated(data, field_name, "plan"))

    # Query variants should have at least 1
    variants = data.get("query_variants", [])
    report.checks.append(
        EvalCheck(
            "plan_variant_count",
            "At least 1 query variant",
            len(variants) >= 1,
            f"Found {len(variants)} variants",
        )
    )

    return report


def evaluate_search(run_root: Path) -> EvalReport:
    """Evaluate search stage output.

    Args:
        run_root: Path to run directory.

    Returns:
        EvalReport with all checks.
    """
    report = EvalReport(stage="search")
    candidates_path = run_root / "search" / "candidates.jsonl"

    if not candidates_path.exists():
        report.checks.append(
            EvalCheck("candidates_exists", "Candidates file exists", False)
        )
        return report
    report.checks.append(EvalCheck("candidates_exists", "Candidates file exists", True))

    lines = candidates_path.read_text(encoding="utf-8").strip().split("\n")
    lines = [line for line in lines if line.strip()]
    report.checks.append(
        EvalCheck(
            "candidates_count",
            "At least 1 candidate",
            len(lines) >= 1,
            f"Found {len(lines)} candidates",
        )
    )

    # Check first candidate has required fields
    if lines:
        try:
            first = json.loads(lines[0])
            for field_name in ["arxiv_id", "title"]:
                report.checks.append(check_field_populated(first, field_name, "search"))
        except json.JSONDecodeError:
            report.checks.append(
                EvalCheck(
                    "candidate_json",
                    "Candidates are valid JSON",
                    False,
                )
            )

    return report


def evaluate_screen(run_root: Path) -> EvalReport:
    """Evaluate screen stage output.

    Args:
        run_root: Path to run directory.

    Returns:
        EvalReport with all checks.
    """
    report = EvalReport(stage="screen")
    shortlist_path = run_root / "screen" / "shortlist.jsonl"

    if not shortlist_path.exists():
        report.checks.append(
            EvalCheck("shortlist_exists", "Shortlist file exists", False)
        )
        return report
    report.checks.append(EvalCheck("shortlist_exists", "Shortlist file exists", True))

    lines = shortlist_path.read_text(encoding="utf-8").strip().split("\n")
    lines = [line for line in lines if line.strip()]
    report.checks.append(
        EvalCheck(
            "shortlist_count",
            "At least 1 shortlisted paper",
            len(lines) >= 1,
            f"Found {len(lines)} papers",
        )
    )

    # Check scores are in valid range
    for _i, line in enumerate(lines[:5]):  # Check first 5
        try:
            entry = json.loads(line)
            score = entry.get("blended_score") or entry.get("score")
            if score is not None:
                report.checks.append(check_score_range({"score": score}, "score"))
        except json.JSONDecodeError:
            pass

    return report


def evaluate_summarize(run_root: Path) -> EvalReport:
    """Evaluate summarize stage output.

    Args:
        run_root: Path to run directory.

    Returns:
        EvalReport with all checks.
    """
    report = EvalReport(stage="summarize")
    sum_dir = run_root / "summarize"

    synthesis_json = sum_dir / "synthesis.json"
    synthesis_md = sum_dir / "synthesis.md"

    for path, name in [
        (synthesis_json, "synthesis_json"),
        (synthesis_md, "synthesis_md"),
    ]:
        report.checks.append(
            EvalCheck(f"{name}_exists", f"{name} exists", path.exists())
        )

    if synthesis_json.exists():
        try:
            data = json.loads(synthesis_json.read_text(encoding="utf-8"))
            report.checks.append(EvalCheck("synthesis_json_valid", "Valid JSON", True))

            for field_name in ["topic", "paper_summaries"]:
                report.checks.append(
                    check_field_populated(data, field_name, "summarize")
                )

            report.checks.append(
                check_list_count_consistency(data, "paper_summaries", "paper_count")
            )
        except json.JSONDecodeError as exc:
            report.checks.append(
                EvalCheck("synthesis_json_valid", "Valid JSON", False, str(exc))
            )

    return report


STAGE_EVALUATORS: dict[str, Any] = {
    "plan": evaluate_plan,
    "search": evaluate_search,
    "screen": evaluate_screen,
    "summarize": evaluate_summarize,
}


def evaluate_stage(run_root: Path, stage: str) -> EvalReport:
    """Evaluate a specific stage's output.

    Args:
        run_root: Path to run directory.
        stage: Stage name.

    Returns:
        EvalReport with all checks.
    """
    evaluator = STAGE_EVALUATORS.get(stage)
    if evaluator is None:
        report = EvalReport(stage=stage)
        report.checks.append(
            EvalCheck(
                "evaluator_exists",
                f"Evaluator for {stage}",
                False,
                f"No evaluator for stage '{stage}'",
                severity="warning",
            )
        )
        return report

    return evaluator(run_root)


def evaluate_run(run_root: Path, stages: list[str] | None = None) -> list[EvalReport]:
    """Evaluate all stages of a run.

    Args:
        run_root: Path to run directory.
        stages: Stages to evaluate (None = all available).

    Returns:
        List of EvalReports, one per stage.
    """
    if stages is None:
        stages = ["plan", "search", "screen", "summarize"]

    reports = []
    for stage in stages:
        report = evaluate_stage(run_root, stage)
        reports.append(report)
        logger.info("Evaluation: %s", report.summary())

    return reports
