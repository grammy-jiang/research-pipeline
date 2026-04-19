"""Pass@k + Pass[k] dual evaluation metrics (Claw-Eval inspired).

Implements dual-metric evaluation from arXiv 2604.06132:
- Pass@k: probability that at least 1 of k samples is correct (capability ceiling)
- Pass[k]: probability that all k samples are correct (reliability floor)
- Safety gate: fabrication zeros entire score (multiplicative, not additive)

These metrics evaluate pipeline output quality across multiple runs or
samples of the same query, providing both optimistic (ceiling) and
pessimistic (floor) quality bounds.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Result of a single pipeline run/sample."""

    sample_id: str
    run_id: str
    correct: bool = False
    fabrication_detected: bool = False
    quality_score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DualMetricResult:
    """Combined Pass@k and Pass[k] result for a query."""

    query: str
    k: int
    n: int  # total samples evaluated
    c: int  # number of correct samples
    pass_at_k: float = 0.0
    pass_bracket_k: float = 0.0
    safety_gate: float = 1.0  # 0.0 if any fabrication detected
    gated_pass_at_k: float = 0.0
    gated_pass_bracket_k: float = 0.0
    fabrication_count: int = 0
    samples: list[SampleResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "query": self.query,
            "k": self.k,
            "n": self.n,
            "c": self.c,
            "pass_at_k": self.pass_at_k,
            "pass_bracket_k": self.pass_bracket_k,
            "safety_gate": self.safety_gate,
            "gated_pass_at_k": self.gated_pass_at_k,
            "gated_pass_bracket_k": self.gated_pass_bracket_k,
            "fabrication_count": self.fabrication_count,
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "run_id": s.run_id,
                    "correct": s.correct,
                    "fabrication_detected": s.fabrication_detected,
                    "quality_score": s.quality_score,
                }
                for s in self.samples
            ],
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple queries."""

    total_queries: int = 0
    mean_pass_at_k: float = 0.0
    mean_pass_bracket_k: float = 0.0
    mean_gated_pass_at_k: float = 0.0
    mean_gated_pass_bracket_k: float = 0.0
    safety_violation_rate: float = 0.0
    reliability_gap: float = 0.0  # pass_at_k - pass_bracket_k
    query_results: list[DualMetricResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "total_queries": self.total_queries,
            "mean_pass_at_k": self.mean_pass_at_k,
            "mean_pass_bracket_k": self.mean_pass_bracket_k,
            "mean_gated_pass_at_k": self.mean_gated_pass_at_k,
            "mean_gated_pass_bracket_k": self.mean_gated_pass_bracket_k,
            "safety_violation_rate": self.safety_violation_rate,
            "reliability_gap": self.reliability_gap,
            "query_results": [r.to_dict() for r in self.query_results],
        }


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute Pass@k: probability that at least 1 of k samples is correct.

    Uses the unbiased estimator: 1 - C(n-c, k) / C(n, k)
    where C is the binomial coefficient.

    Args:
        n: Total number of samples.
        c: Number of correct samples.
        k: Number of samples drawn.

    Returns:
        Pass@k probability in [0.0, 1.0].
    """
    if n < 1 or k < 1:
        return 0.0
    if c < 0:
        c = 0
    if c > n:
        c = n
    if k > n:
        k = n

    if c == 0:
        return 0.0
    if c >= n:
        return 1.0

    # Use log-space to avoid overflow with large binomial coefficients
    # Pass@k = 1 - C(n-c, k) / C(n, k)
    # In log-space: log(C(n-c,k)) - log(C(n,k))
    if n - c < k:
        return 1.0

    log_ratio = _log_comb(n - c, k) - _log_comb(n, k)
    return 1.0 - math.exp(log_ratio)


def compute_pass_bracket_k(n: int, c: int, k: int) -> float:
    """Compute Pass[k]: probability that all k samples are correct.

    Uses the unbiased estimator: C(c, k) / C(n, k)
    where C is the binomial coefficient.

    Args:
        n: Total number of samples.
        c: Number of correct samples.
        k: Number of samples drawn.

    Returns:
        Pass[k] probability in [0.0, 1.0].
    """
    if n < 1 or k < 1:
        return 0.0
    if c < 0:
        c = 0
    if c > n:
        c = n
    if k > n:
        k = n

    if c < k:
        return 0.0
    if c >= n:
        return 1.0

    log_ratio = _log_comb(c, k) - _log_comb(n, k)
    return math.exp(log_ratio)


def apply_safety_gate(
    samples: list[SampleResult],
) -> float:
    """Compute the safety gate multiplier.

    Returns 0.0 if any sample has fabrication detected (zeros the score),
    1.0 otherwise.

    Args:
        samples: List of sample results to check.

    Returns:
        Safety gate: 0.0 or 1.0.
    """
    if any(s.fabrication_detected for s in samples):
        return 0.0
    return 1.0


def _log_comb(n: int, k: int) -> float:
    """Compute log of binomial coefficient C(n, k) using log-gamma."""
    if k < 0 or k > n:
        return float("-inf")
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


# ---------------------------------------------------------------------------
# Sample evaluation
# ---------------------------------------------------------------------------


def evaluate_sample(
    run_dir: Path,
    run_id: str,
    *,
    reference_findings: list[str] | None = None,
    finding_match_threshold: float = 0.5,
) -> SampleResult:
    """Evaluate a single pipeline run as a sample.

    Checks:
    1. Whether the run produced valid synthesis output
    2. Quality of findings (if reference findings provided)
    3. Fabrication detection (findings without evidence)

    Args:
        run_dir: Path to the run output directory.
        run_id: Identifier for this run.
        reference_findings: Expected findings to match against.
        finding_match_threshold: Fraction of reference findings that must match.

    Returns:
        SampleResult for this run.
    """
    sample_id = f"{run_id}_sample"

    # Check for synthesis output
    synthesis_path = run_dir / "summarize" / "synthesis_report.json"
    if not synthesis_path.exists():
        return SampleResult(
            sample_id=sample_id,
            run_id=run_id,
            correct=False,
            details={"reason": "No synthesis output found"},
        )

    try:
        data = json.loads(synthesis_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return SampleResult(
            sample_id=sample_id,
            run_id=run_id,
            correct=False,
            details={"reason": f"Cannot read synthesis: {exc}"},
        )

    findings = _extract_findings(data)
    evidence_count = _count_evidence(data)

    # Fabrication detection: findings without any evidence backing
    fabrication = len(findings) > 0 and evidence_count == 0
    if not fabrication:
        # Also check per-paper summaries for evidence
        summaries = data.get("per_paper_summaries", [])
        for summary in summaries:
            s_findings = summary.get("findings", [])
            s_evidence = summary.get("evidence", [])
            if len(s_findings) > 3 and len(s_evidence) == 0:
                fabrication = True
                break

    quality = _compute_quality_score(data)

    # Correctness check
    correct = True
    if reference_findings:
        matched = _count_matching_findings(findings, reference_findings)
        match_ratio = matched / len(reference_findings) if reference_findings else 0.0
        correct = match_ratio >= finding_match_threshold
    else:
        # Without reference, correctness = has non-trivial findings
        correct = len(findings) >= 1 and quality > 0.3

    return SampleResult(
        sample_id=sample_id,
        run_id=run_id,
        correct=correct,
        fabrication_detected=fabrication,
        quality_score=quality,
        details={
            "finding_count": len(findings),
            "evidence_count": evidence_count,
        },
    )


def _extract_findings(data: dict[str, Any]) -> list[str]:
    """Extract all findings from synthesis data."""
    findings: list[str] = []

    for agreement in data.get("agreements", []):
        claim = agreement.get("claim", "")
        if claim:
            findings.append(claim)

    for disagreement in data.get("disagreements", []):
        claim = disagreement.get("claim", "")
        if claim:
            findings.append(claim)

    for summary in data.get("per_paper_summaries", []):
        for finding in summary.get("findings", []):
            findings.append(finding)

    for section in (
        "taxonomy",
        "recurring_patterns",
        "evidence_strength_map",
        "operational_implications",
        "production_readiness",
        "design_implications",
        "risk_register",
    ):
        for item in data.get(section, []):
            finding = item.get("finding", "")
            if finding:
                findings.append(finding)

    return findings


def _count_evidence(data: dict[str, Any]) -> int:
    """Count total evidence references in synthesis data."""
    count = 0
    for agreement in data.get("agreements", []):
        count += len(agreement.get("evidence", []))
        count += len(agreement.get("supporting_papers", []))

    for summary in data.get("per_paper_summaries", []):
        count += len(summary.get("evidence", []))

    for row in data.get("traceability_appendix", []):
        evidence_ids = row.get("evidence_ids", "")
        if isinstance(evidence_ids, str) and evidence_ids:
            count += len([item for item in evidence_ids.split(",") if item.strip()])
        elif isinstance(evidence_ids, list):
            count += len(evidence_ids)

    return count


def _compute_quality_score(data: dict[str, Any]) -> float:
    """Compute a quality score for the synthesis output."""
    score = 0.0

    findings = _extract_findings(data)
    evidence = _count_evidence(data)
    summaries = data.get("per_paper_summaries", [])
    gaps = data.get("gaps", []) or data.get("unresolved_questions", [])
    corpus = data.get("corpus", [])

    # Finding completeness (0-0.3)
    if len(findings) >= 5:
        score += 0.3
    elif findings:
        score += 0.3 * min(1.0, len(findings) / 5)

    # Evidence backing (0-0.3)
    if evidence > 0 and findings:
        ratio = min(1.0, evidence / len(findings))
        score += 0.3 * ratio

    # Paper coverage (0-0.2)
    paper_count = len(summaries) or len(corpus)
    if paper_count >= 3:
        score += 0.2
    elif paper_count:
        score += 0.2 * min(1.0, paper_count / 3)

    # Gap identification (0-0.2)
    if gaps:
        score += 0.2 * min(1.0, len(gaps) / 2)

    return round(min(1.0, score), 4)


def _count_matching_findings(actual: list[str], reference: list[str]) -> int:
    """Count how many reference findings are covered by actual findings."""
    matched = 0
    actual_lower = [f.lower() for f in actual]

    for ref in reference:
        ref_tokens = set(ref.lower().split())
        if len(ref_tokens) < 2:
            continue
        for act in actual_lower:
            act_tokens = set(act.split())
            overlap = len(ref_tokens & act_tokens)
            if overlap >= max(2, len(ref_tokens) * 0.4):
                matched += 1
                break

    return matched


# ---------------------------------------------------------------------------
# Dual metric computation
# ---------------------------------------------------------------------------


def compute_dual_metrics(
    query: str,
    samples: list[SampleResult],
    k: int = 5,
) -> DualMetricResult:
    """Compute Pass@k, Pass[k], and safety-gated variants.

    Args:
        query: The research query being evaluated.
        samples: List of sample results from multiple runs.
        k: Number of samples for Pass@k/Pass[k] computation.

    Returns:
        DualMetricResult with all metric values.
    """
    n = len(samples)
    c = sum(1 for s in samples if s.correct)
    fabrication_count = sum(1 for s in samples if s.fabrication_detected)

    effective_k = min(k, n)

    pass_at_k = compute_pass_at_k(n, c, effective_k)
    pass_bracket_k = compute_pass_bracket_k(n, c, effective_k)
    safety = apply_safety_gate(samples)

    return DualMetricResult(
        query=query,
        k=effective_k,
        n=n,
        c=c,
        pass_at_k=round(pass_at_k, 4),
        pass_bracket_k=round(pass_bracket_k, 4),
        safety_gate=safety,
        gated_pass_at_k=round(pass_at_k * safety, 4),
        gated_pass_bracket_k=round(pass_bracket_k * safety, 4),
        fabrication_count=fabrication_count,
        samples=samples,
    )


def aggregate_metrics(
    results: list[DualMetricResult],
) -> AggregateMetrics:
    """Compute aggregate metrics across multiple queries.

    Args:
        results: List of per-query DualMetricResult instances.

    Returns:
        AggregateMetrics with means and reliability gap.
    """
    if not results:
        return AggregateMetrics()

    total = len(results)
    mean_pak = sum(r.pass_at_k for r in results) / total
    mean_pbk = sum(r.pass_bracket_k for r in results) / total
    mean_gpak = sum(r.gated_pass_at_k for r in results) / total
    mean_gpbk = sum(r.gated_pass_bracket_k for r in results) / total
    safety_violations = sum(1 for r in results if r.safety_gate < 1.0)

    return AggregateMetrics(
        total_queries=total,
        mean_pass_at_k=round(mean_pak, 4),
        mean_pass_bracket_k=round(mean_pbk, 4),
        mean_gated_pass_at_k=round(mean_gpak, 4),
        mean_gated_pass_bracket_k=round(mean_gpbk, 4),
        safety_violation_rate=round(safety_violations / total, 4),
        reliability_gap=round(mean_pak - mean_pbk, 4),
        query_results=results,
    )


# ---------------------------------------------------------------------------
# SQLite metrics store
# ---------------------------------------------------------------------------

_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS dual_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    k INTEGER NOT NULL,
    n INTEGER NOT NULL,
    c INTEGER NOT NULL,
    pass_at_k REAL NOT NULL,
    pass_bracket_k REAL NOT NULL,
    safety_gate REAL NOT NULL,
    gated_pass_at_k REAL NOT NULL,
    gated_pass_bracket_k REAL NOT NULL,
    fabrication_count INTEGER NOT NULL,
    result_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_query
    ON dual_metrics(query);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
    ON dual_metrics(timestamp);
"""


class MetricsStore:
    """SQLite-backed storage for dual evaluation metrics."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_METRICS_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def store_result(self, result: DualMetricResult) -> int:
        """Store a dual metric result, return the record ID."""
        timestamp = datetime.now(tz=UTC).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO dual_metrics "
            "(query, timestamp, k, n, c, pass_at_k, pass_bracket_k, "
            "safety_gate, gated_pass_at_k, gated_pass_bracket_k, "
            "fabrication_count, result_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result.query,
                timestamp,
                result.k,
                result.n,
                result.c,
                result.pass_at_k,
                result.pass_bracket_k,
                result.safety_gate,
                result.gated_pass_at_k,
                result.gated_pass_bracket_k,
                result.fabrication_count,
                json.dumps(result.to_dict()),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get_history(self, query: str) -> list[DualMetricResult]:
        """Get metric history for a specific query."""
        rows = self._conn.execute(
            "SELECT result_json FROM dual_metrics "
            "WHERE query = ? ORDER BY timestamp DESC",
            (query,),
        ).fetchall()
        return [_dict_to_dual_result(json.loads(r[0])) for r in rows]

    def get_all(self) -> list[DualMetricResult]:
        """Get all stored metric results."""
        rows = self._conn.execute(
            "SELECT result_json FROM dual_metrics ORDER BY timestamp DESC"
        ).fetchall()
        return [_dict_to_dual_result(json.loads(r[0])) for r in rows]

    def get_latest(self, query: str) -> DualMetricResult | None:
        """Get the most recent result for a query."""
        row = self._conn.execute(
            "SELECT result_json FROM dual_metrics "
            "WHERE query = ? ORDER BY timestamp DESC LIMIT 1",
            (query,),
        ).fetchone()
        if row is None:
            return None
        return _dict_to_dual_result(json.loads(row[0]))


def _dict_to_dual_result(data: dict[str, Any]) -> DualMetricResult:
    """Convert serialized dict back to DualMetricResult."""
    samples = []
    for s in data.get("samples", []):
        samples.append(
            SampleResult(
                sample_id=s.get("sample_id", ""),
                run_id=s.get("run_id", ""),
                correct=s.get("correct", False),
                fabrication_detected=s.get("fabrication_detected", False),
                quality_score=s.get("quality_score", 0.0),
            )
        )
    return DualMetricResult(
        query=data.get("query", ""),
        k=data.get("k", 0),
        n=data.get("n", 0),
        c=data.get("c", 0),
        pass_at_k=data.get("pass_at_k", 0.0),
        pass_bracket_k=data.get("pass_bracket_k", 0.0),
        safety_gate=data.get("safety_gate", 1.0),
        gated_pass_at_k=data.get("gated_pass_at_k", 0.0),
        gated_pass_bracket_k=data.get("gated_pass_bracket_k", 0.0),
        fabrication_count=data.get("fabrication_count", 0),
        samples=samples,
    )


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def evaluate_runs(
    workspace: Path,
    query: str,
    run_ids: list[str],
    *,
    k: int = 5,
    reference_findings: list[str] | None = None,
    store_results: bool = True,
) -> DualMetricResult:
    """Evaluate multiple pipeline runs and compute dual metrics.

    Args:
        workspace: Path to workspace directory.
        query: The research query being evaluated.
        run_ids: List of run IDs to evaluate as samples.
        k: Number of samples for Pass@k/Pass[k].
        reference_findings: Optional reference findings for correctness.
        store_results: Whether to persist to SQLite.

    Returns:
        DualMetricResult with all computed metrics.
    """
    runs_dir = workspace / "runs"
    samples: list[SampleResult] = []

    for run_id in run_ids:
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            logger.warning("Run directory not found: %s", run_dir)
            samples.append(
                SampleResult(
                    sample_id=f"{run_id}_missing",
                    run_id=run_id,
                    correct=False,
                    details={"reason": "Run directory not found"},
                )
            )
            continue

        sample = evaluate_sample(
            run_dir,
            run_id,
            reference_findings=reference_findings,
        )
        samples.append(sample)

    result = compute_dual_metrics(query, samples, k=k)

    if store_results:
        db_path = workspace / ".dual_metrics.db"
        store = MetricsStore(db_path)
        try:
            store.store_result(result)
            logger.info(
                "Stored dual metrics for query '%s': "
                "Pass@%d=%.3f, Pass[%d]=%.3f, safety=%.1f",
                query,
                result.k,
                result.pass_at_k,
                result.k,
                result.pass_bracket_k,
                result.safety_gate,
            )
        finally:
            store.close()

    return result
