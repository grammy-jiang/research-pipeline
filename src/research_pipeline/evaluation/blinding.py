"""Epistemic blinding audits for detecting LLM prior contamination.

Implements an A/B blinding protocol inspired by arXiv 2604.06013:
- Blind (mask) identifying features from paper content (authors, title, venue, year)
- Compare analysis outputs between blinded and unblinded versions
- Score contamination level based on how much analysis depends on identity
- Track contamination metrics across pipeline runs

The module works heuristically without LLM access: it detects references
to identifying features in analysis outputs and measures information leakage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

FEATURE_TYPES = ("authors", "title", "venue", "year", "citations")


@dataclass
class BlindingMask:
    """Which identifying features to mask in a document."""

    authors: bool = True
    title: bool = True
    venue: bool = True
    year: bool = True
    citations: bool = True

    def active_features(self) -> list[str]:
        """Return list of feature names that are active for masking."""
        return [f for f in FEATURE_TYPES if getattr(self, f)]


@dataclass
class IdentifyingFeature:
    """A detected identifying feature in text."""

    feature_type: str  # one of FEATURE_TYPES
    value: str  # the actual string found
    locations: list[int] = field(default_factory=list)  # char offsets


@dataclass
class BlindedDocument:
    """A document with identifying features masked."""

    original_text: str
    blinded_text: str
    features_masked: list[IdentifyingFeature] = field(default_factory=list)
    mask_applied: BlindingMask = field(default_factory=BlindingMask)

    @property
    def mask_count(self) -> int:
        """Total number of masking operations applied."""
        return sum(len(f.locations) for f in self.features_masked)


@dataclass
class ContaminationScore:
    """Per-feature contamination scores for a single paper analysis."""

    paper_id: str
    feature_scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    identity_references: int = 0  # count of identity-leaking references
    total_claims: int = 0
    contaminated_claims: int = 0

    @property
    def contamination_ratio(self) -> float:
        """Fraction of claims that reference identifying features."""
        if self.total_claims == 0:
            return 0.0
        return self.contaminated_claims / self.total_claims


@dataclass
class BlindingAuditResult:
    """Full audit result for a pipeline run."""

    run_id: str
    timestamp: str
    paper_scores: list[ContaminationScore] = field(default_factory=list)
    aggregate_score: float = 0.0
    high_contamination_papers: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "paper_scores": [
                {
                    "paper_id": s.paper_id,
                    "feature_scores": s.feature_scores,
                    "overall_score": s.overall_score,
                    "identity_references": s.identity_references,
                    "total_claims": s.total_claims,
                    "contaminated_claims": s.contaminated_claims,
                    "contamination_ratio": s.contamination_ratio,
                }
                for s in self.paper_scores
            ],
            "aggregate_score": self.aggregate_score,
            "high_contamination_papers": self.high_contamination_papers,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# SQLite audit store
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS blinding_audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    aggregate_score REAL NOT NULL,
    high_contamination_count INTEGER NOT NULL,
    recommendation TEXT NOT NULL,
    result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_contamination (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_id INTEGER NOT NULL,
    paper_id TEXT NOT NULL,
    overall_score REAL NOT NULL,
    identity_references INTEGER NOT NULL,
    total_claims INTEGER NOT NULL,
    contaminated_claims INTEGER NOT NULL,
    feature_scores_json TEXT NOT NULL,
    FOREIGN KEY (audit_id) REFERENCES blinding_audits(id)
);

CREATE INDEX IF NOT EXISTS idx_audits_run
    ON blinding_audits(run_id);
CREATE INDEX IF NOT EXISTS idx_contamination_paper
    ON paper_contamination(paper_id);
"""


class AuditStore:
    """SQLite-backed storage for blinding audit results."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def store_audit(self, result: BlindingAuditResult) -> int:
        """Store an audit result, return the audit ID."""
        cursor = self._conn.execute(
            "INSERT INTO blinding_audits "
            "(run_id, timestamp, aggregate_score, high_contamination_count, "
            "recommendation, result_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.run_id,
                result.timestamp,
                result.aggregate_score,
                len(result.high_contamination_papers),
                result.recommendation,
                json.dumps(result.to_dict()),
            ),
        )
        audit_id = cursor.lastrowid
        if audit_id is None:
            msg = "Failed to insert audit record"
            raise RuntimeError(msg)

        for score in result.paper_scores:
            self._conn.execute(
                "INSERT INTO paper_contamination "
                "(audit_id, paper_id, overall_score, identity_references, "
                "total_claims, contaminated_claims, feature_scores_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    audit_id,
                    score.paper_id,
                    score.overall_score,
                    score.identity_references,
                    score.total_claims,
                    score.contaminated_claims,
                    json.dumps(score.feature_scores),
                ),
            )

        self._conn.commit()
        return audit_id

    def get_audit(self, audit_id: int) -> BlindingAuditResult | None:
        """Retrieve an audit result by ID."""
        row = self._conn.execute(
            "SELECT result_json FROM blinding_audits WHERE id = ?",
            (audit_id,),
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return _dict_to_audit_result(data)

    def get_audits_for_run(self, run_id: str) -> list[BlindingAuditResult]:
        """Get all audits for a specific run."""
        rows = self._conn.execute(
            "SELECT result_json FROM blinding_audits "
            "WHERE run_id = ? ORDER BY timestamp DESC",
            (run_id,),
        ).fetchall()
        return [_dict_to_audit_result(json.loads(r[0])) for r in rows]

    def get_paper_history(self, paper_id: str) -> list[dict[str, Any]]:
        """Get contamination history for a specific paper across audits."""
        rows = self._conn.execute(
            "SELECT ba.run_id, ba.timestamp, pc.overall_score, "
            "pc.identity_references, pc.total_claims, pc.contaminated_claims "
            "FROM paper_contamination pc "
            "JOIN blinding_audits ba ON pc.audit_id = ba.id "
            "WHERE pc.paper_id = ? ORDER BY ba.timestamp DESC",
            (paper_id,),
        ).fetchall()
        return [
            {
                "run_id": r[0],
                "timestamp": r[1],
                "overall_score": r[2],
                "identity_references": r[3],
                "total_claims": r[4],
                "contaminated_claims": r[5],
            }
            for r in rows
        ]

    def get_all_audits(self) -> list[BlindingAuditResult]:
        """Get all audit results."""
        rows = self._conn.execute(
            "SELECT result_json FROM blinding_audits ORDER BY timestamp DESC"
        ).fetchall()
        return [_dict_to_audit_result(json.loads(r[0])) for r in rows]


def _dict_to_audit_result(data: dict[str, Any]) -> BlindingAuditResult:
    """Convert a serialized dict back to BlindingAuditResult."""
    paper_scores = []
    for ps in data.get("paper_scores", []):
        paper_scores.append(
            ContaminationScore(
                paper_id=ps["paper_id"],
                feature_scores=ps.get("feature_scores", {}),
                overall_score=ps.get("overall_score", 0.0),
                identity_references=ps.get("identity_references", 0),
                total_claims=ps.get("total_claims", 0),
                contaminated_claims=ps.get("contaminated_claims", 0),
            )
        )
    return BlindingAuditResult(
        run_id=data["run_id"],
        timestamp=data["timestamp"],
        paper_scores=paper_scores,
        aggregate_score=data.get("aggregate_score", 0.0),
        high_contamination_papers=data.get("high_contamination_papers", []),
        recommendation=data.get("recommendation", ""),
    )


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------

# Common academic venue patterns
_VENUE_PATTERNS = [
    r"\b(?:NeurIPS|ICML|ICLR|AAAI|IJCAI|ACL|EMNLP|NAACL|CVPR|ICCV|ECCV)\b",
    r"\b(?:KDD|WWW|SIGMOD|VLDB|ICDE|SIGIR|CIKM|WSDM|RecSys)\b",
    r"\b(?:Nature|Science|PNAS|Cell|Lancet)\b",
    r"\b(?:JMLR|TACL|CL|AIJ|MLJ|JAIR)\b",
    r"\b(?:arXiv|bioRxiv|medRxiv)\b",
    r"\b(?:Proceedings|Conference|Workshop|Symposium|Journal)\b",
]
_VENUE_RE = re.compile("|".join(_VENUE_PATTERNS), re.IGNORECASE)

# Year patterns (4-digit years in plausible academic range)
_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")

# Citation patterns like [1], [2,3], (Author et al., 2023)
_CITATION_RE = re.compile(
    r"\[[\d,\s]+\]"
    r"|"
    r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-z]+))?,?\s*\d{4}\)"
)


def detect_identifying_features(
    text: str,
    authors: list[str] | None = None,
    title: str | None = None,
    venue: str | None = None,
    year: int | None = None,
    mask: BlindingMask | None = None,
) -> list[IdentifyingFeature]:
    """Detect identifying features in text.

    Args:
        text: The text to scan for identifying features.
        authors: Known author names to look for.
        title: Known paper title to look for.
        venue: Known venue name to look for.
        year: Known publication year to look for.
        mask: Which features to detect (all by default).

    Returns:
        List of detected identifying features with locations.
    """
    if mask is None:
        mask = BlindingMask()

    features: list[IdentifyingFeature] = []

    if mask.authors and authors:
        for author in authors:
            # Search for full name and last name
            parts = author.strip().split()
            search_terms = [author.strip()]
            if len(parts) > 1:
                search_terms.append(parts[-1])  # last name

            for term in search_terms:
                if len(term) < 3:
                    continue
                locations = [
                    m.start() for m in re.finditer(re.escape(term), text, re.IGNORECASE)
                ]
                if locations:
                    features.append(
                        IdentifyingFeature(
                            feature_type="authors",
                            value=term,
                            locations=locations,
                        )
                    )

    if mask.title and title:
        # Search for substantial fragments of the title (>= 5 words)
        title_words = title.strip().split()
        if len(title_words) >= 5:
            # Try full title first
            locations = [
                m.start()
                for m in re.finditer(re.escape(title.strip()), text, re.IGNORECASE)
            ]
            if locations:
                features.append(
                    IdentifyingFeature(
                        feature_type="title",
                        value=title.strip(),
                        locations=locations,
                    )
                )
            # Also try first 5 words as a fragment
            fragment = " ".join(title_words[:5])
            frag_locations = [
                m.start() for m in re.finditer(re.escape(fragment), text, re.IGNORECASE)
            ]
            if frag_locations and not locations:
                features.append(
                    IdentifyingFeature(
                        feature_type="title",
                        value=fragment,
                        locations=frag_locations,
                    )
                )

    if mask.venue:
        if venue:
            locations = [
                m.start()
                for m in re.finditer(re.escape(venue.strip()), text, re.IGNORECASE)
            ]
            if locations:
                features.append(
                    IdentifyingFeature(
                        feature_type="venue",
                        value=venue.strip(),
                        locations=locations,
                    )
                )
        # Also detect known venue patterns
        for match in _VENUE_RE.finditer(text):
            features.append(
                IdentifyingFeature(
                    feature_type="venue",
                    value=match.group(),
                    locations=[match.start()],
                )
            )

    if mask.year and year:
        year_str = str(year)
        locations = [m.start() for m in re.finditer(re.escape(year_str), text) if True]
        if locations:
            features.append(
                IdentifyingFeature(
                    feature_type="year",
                    value=year_str,
                    locations=locations,
                )
            )

    if mask.citations:
        for match in _CITATION_RE.finditer(text):
            features.append(
                IdentifyingFeature(
                    feature_type="citations",
                    value=match.group(),
                    locations=[match.start()],
                )
            )

    return features


# ---------------------------------------------------------------------------
# Document blinding
# ---------------------------------------------------------------------------


def blind_document(
    text: str,
    authors: list[str] | None = None,
    title: str | None = None,
    venue: str | None = None,
    year: int | None = None,
    mask: BlindingMask | None = None,
) -> BlindedDocument:
    """Apply blinding mask to a document, replacing identifying features.

    Args:
        text: Original document text.
        authors: Author names to mask.
        title: Paper title to mask.
        venue: Publication venue to mask.
        year: Publication year to mask.
        mask: Which features to mask.

    Returns:
        BlindedDocument with masked text and feature records.
    """
    if mask is None:
        mask = BlindingMask()

    features = detect_identifying_features(
        text, authors=authors, title=title, venue=venue, year=year, mask=mask
    )

    blinded = text
    # Sort features by location (reverse) to replace from end to start
    replacements: list[tuple[int, int, str, IdentifyingFeature]] = []
    for feat in features:
        placeholder = f"[MASKED_{feat.feature_type.upper()}]"
        for loc in feat.locations:
            replacements.append((loc, loc + len(feat.value), placeholder, feat))

    # Sort by start position descending for safe replacement
    replacements.sort(key=lambda r: r[0], reverse=True)

    # Deduplicate overlapping replacements (keep the first/longest)
    seen_ranges: list[tuple[int, int]] = []
    filtered: list[tuple[int, int, str, IdentifyingFeature]] = []
    for start, end, placeholder, feat in replacements:
        overlaps = any(not (end <= s or start >= e) for s, e in seen_ranges)
        if not overlaps:
            filtered.append((start, end, placeholder, feat))
            seen_ranges.append((start, end))

    for start, end, placeholder, _feat in filtered:
        blinded = blinded[:start] + placeholder + blinded[end:]

    return BlindedDocument(
        original_text=text,
        blinded_text=blinded,
        features_masked=features,
        mask_applied=mask,
    )


# ---------------------------------------------------------------------------
# Contamination scoring
# ---------------------------------------------------------------------------


def score_contamination(
    findings: list[str],
    authors: list[str] | None = None,
    title: str | None = None,
    venue: str | None = None,
    year: int | None = None,
) -> ContaminationScore:
    """Score how much analysis findings depend on identifying features.

    Checks each finding for references to known identifying features.
    High contamination means the analysis relies heavily on prior knowledge
    rather than evidence from the paper content itself.

    Args:
        findings: List of finding/claim strings from paper analysis.
        authors: Known author names.
        title: Paper title.
        venue: Publication venue.
        year: Publication year.

    Returns:
        ContaminationScore with per-feature and overall scores.
    """
    paper_id = _make_paper_id(title or "", authors or [])

    if not findings:
        return ContaminationScore(paper_id=paper_id)

    feature_hits: dict[str, int] = dict.fromkeys(FEATURE_TYPES, 0)
    contaminated_count = 0

    for finding in findings:
        finding_contaminated = False

        # Check author references
        if authors:
            for author in authors:
                parts = author.strip().split()
                search_terms = [author.strip()]
                if len(parts) > 1:
                    search_terms.append(parts[-1])
                for term in search_terms:
                    if len(term) >= 3 and term.lower() in finding.lower():
                        feature_hits["authors"] += 1
                        finding_contaminated = True
                        break

        # Check title references
        if title:
            title_words = title.strip().split()
            if len(title_words) >= 3:
                # Check if 3+ consecutive title words appear
                for i in range(len(title_words) - 2):
                    fragment = " ".join(title_words[i : i + 3])
                    if fragment.lower() in finding.lower():
                        feature_hits["title"] += 1
                        finding_contaminated = True
                        break

        # Check venue references
        if venue and venue.lower() in finding.lower():
            feature_hits["venue"] += 1
            finding_contaminated = True
        if _VENUE_RE.search(finding):
            feature_hits["venue"] += 1
            finding_contaminated = True

        # Check year references
        if year and str(year) in finding:
            feature_hits["year"] += 1
            finding_contaminated = True

        # Check citation pattern references
        if _CITATION_RE.search(finding):
            feature_hits["citations"] += 1
            finding_contaminated = True

        if finding_contaminated:
            contaminated_count += 1

    total = len(findings)
    total_hits = sum(feature_hits.values())
    identity_refs = total_hits

    # Per-feature score: hits / total findings, capped at 1.0
    feature_scores = {}
    for feat, hits in feature_hits.items():
        feature_scores[feat] = min(1.0, hits / total) if total > 0 else 0.0

    # Overall: weighted combination (authors and title weigh more)
    weights = {
        "authors": 0.30,
        "title": 0.25,
        "venue": 0.20,
        "year": 0.10,
        "citations": 0.15,
    }
    overall = sum(feature_scores.get(f, 0.0) * w for f, w in weights.items())

    return ContaminationScore(
        paper_id=paper_id,
        feature_scores=feature_scores,
        overall_score=round(overall, 4),
        identity_references=identity_refs,
        total_claims=total,
        contaminated_claims=contaminated_count,
    )


def _make_paper_id(title: str, authors: list[str]) -> str:
    """Create a stable paper identifier from title and authors."""
    key = f"{title.lower().strip()}|{'|'.join(a.lower().strip() for a in authors)}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Run-level audit
# ---------------------------------------------------------------------------

# Contamination thresholds
HIGH_CONTAMINATION_THRESHOLD = 0.4
MEDIUM_CONTAMINATION_THRESHOLD = 0.2


def audit_paper_summary(
    summary_data: dict[str, Any],
) -> ContaminationScore:
    """Audit a single paper summary for contamination.

    Args:
        summary_data: Dict with keys: title, authors, venue, year,
                      findings (list of strings).

    Returns:
        ContaminationScore for the paper.
    """
    title = summary_data.get("title", "")
    authors = summary_data.get("authors", [])
    venue = summary_data.get("venue")
    year = summary_data.get("year")
    findings = summary_data.get("findings", [])

    # Also check methodology and objective if present
    methodology = summary_data.get("methodology", "")
    objective = summary_data.get("objective", "")
    all_claims = list(findings)
    if methodology:
        all_claims.append(methodology)
    if objective:
        all_claims.append(objective)

    return score_contamination(
        findings=all_claims,
        authors=authors,
        title=title,
        venue=venue,
        year=year,
    )


def run_blinding_audit(
    run_dir: Path,
    run_id: str,
    *,
    contamination_threshold: float = HIGH_CONTAMINATION_THRESHOLD,
) -> BlindingAuditResult:
    """Run a full blinding audit on a pipeline run.

    Scans the summarize stage output for paper summaries and scores each
    for contamination from identifying features.

    Args:
        run_dir: Path to the pipeline run directory.
        run_id: Identifier for this run.
        contamination_threshold: Score above which a paper is flagged.

    Returns:
        BlindingAuditResult with per-paper and aggregate scores.
    """
    timestamp = datetime.now(tz=UTC).isoformat()

    # Find summary files
    summarize_dir = run_dir / "summarize"
    paper_scores: list[ContaminationScore] = []

    # Check for synthesis report (has per-paper data)
    synthesis_path = summarize_dir / "synthesis_report.json"
    if synthesis_path.exists():
        paper_scores.extend(_audit_synthesis_report(synthesis_path))

    # Check for individual paper summaries
    summaries_dir = summarize_dir / "summaries"
    if summaries_dir.exists():
        for summary_file in sorted(summaries_dir.glob("*.json")):
            try:
                data = json.loads(summary_file.read_text())
                score = audit_paper_summary(data)
                paper_scores.append(score)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping %s: %s", summary_file.name, exc)

    # Also scan screen stage for candidate data enrichment
    screen_dir = run_dir / "screen"
    candidates_path = screen_dir / "screened.json"
    if candidates_path.exists():
        paper_scores.extend(_audit_screened_candidates(candidates_path))

    if not paper_scores:
        return BlindingAuditResult(
            run_id=run_id,
            timestamp=timestamp,
            recommendation="No paper summaries found to audit.",
        )

    # Aggregate
    high_contamination = [
        s.paper_id for s in paper_scores if s.overall_score >= contamination_threshold
    ]

    aggregate = (
        sum(s.overall_score for s in paper_scores) / len(paper_scores)
        if paper_scores
        else 0.0
    )

    recommendation = _generate_recommendation(
        aggregate, len(high_contamination), len(paper_scores)
    )

    return BlindingAuditResult(
        run_id=run_id,
        timestamp=timestamp,
        paper_scores=paper_scores,
        aggregate_score=round(aggregate, 4),
        high_contamination_papers=high_contamination,
        recommendation=recommendation,
    )


def _audit_synthesis_report(path: Path) -> list[ContaminationScore]:
    """Extract and audit papers from a synthesis report."""
    scores: list[ContaminationScore] = []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot read synthesis report: %s", exc)
        return scores

    # Check per_paper_summaries if present
    for summary in data.get("per_paper_summaries", []):
        score = audit_paper_summary(summary)
        scores.append(score)

    # Check agreements and disagreements for contamination
    findings: list[str] = []
    for agreement in data.get("agreements", []):
        claim = agreement.get("claim", "")
        if claim:
            findings.append(claim)
    for disagreement in data.get("disagreements", []):
        claim = disagreement.get("claim", "")
        if claim:
            findings.append(claim)
    for gap in data.get("gaps", []):
        desc = gap.get("description", "") or gap.get("gap", "")
        if desc:
            findings.append(desc)

    if findings:
        synthesis_score = score_contamination(
            findings=findings,
            authors=None,
            title=None,
            venue=None,
            year=None,
        )
        synthesis_score.paper_id = "synthesis_aggregate"
        scores.append(synthesis_score)

    return scores


def _audit_screened_candidates(path: Path) -> list[ContaminationScore]:
    """Audit screened candidates for feature leakage into screening rationale."""
    scores: list[ContaminationScore] = []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot read screened candidates: %s", exc)
        return scores

    candidates = data if isinstance(data, list) else data.get("candidates", [])
    for cand in candidates:
        rationale = cand.get("rationale", "")
        if not rationale:
            continue
        title = cand.get("title", "")
        authors = cand.get("authors", [])
        venue = cand.get("venue")
        year = cand.get("year")

        score = score_contamination(
            findings=[rationale],
            authors=authors,
            title=title,
            venue=venue,
            year=year,
        )
        scores.append(score)

    return scores


def _generate_recommendation(
    aggregate_score: float,
    high_count: int,
    total_count: int,
) -> str:
    """Generate a human-readable recommendation based on audit results."""
    if total_count == 0:
        return "No papers audited."

    ratio = high_count / total_count

    if aggregate_score < MEDIUM_CONTAMINATION_THRESHOLD:
        return (
            f"LOW contamination (score={aggregate_score:.3f}). "
            f"Analysis appears evidence-based with minimal identity reliance."
        )
    elif aggregate_score < HIGH_CONTAMINATION_THRESHOLD:
        return (
            f"MEDIUM contamination (score={aggregate_score:.3f}). "
            f"{high_count}/{total_count} papers show identity references. "
            f"Consider re-running analysis with blinded inputs for flagged papers."
        )
    else:
        return (
            f"HIGH contamination (score={aggregate_score:.3f}). "
            f"{high_count}/{total_count} papers ({ratio:.0%}) heavily reference "
            f"identifying features. Strongly recommend re-analysis with "
            f"blinded document inputs to verify findings are evidence-based."
        )


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def run_blinding_audit_for_workspace(
    workspace: Path,
    *,
    run_id: str | None = None,
    contamination_threshold: float = HIGH_CONTAMINATION_THRESHOLD,
    store_results: bool = True,
) -> BlindingAuditResult:
    """Run blinding audit for a workspace, optionally storing results.

    Args:
        workspace: Path to the workspace directory.
        run_id: Specific run to audit (latest if None).
        contamination_threshold: Threshold for flagging papers.
        store_results: Whether to persist results to SQLite.

    Returns:
        BlindingAuditResult for the audited run.
    """
    runs_dir = workspace / "runs"
    if not runs_dir.exists():
        return BlindingAuditResult(
            run_id=run_id or "unknown",
            timestamp=datetime.now(tz=UTC).isoformat(),
            recommendation="No runs directory found in workspace.",
        )

    # Find target run
    if run_id:
        run_dir = runs_dir / run_id
    else:
        # Use latest run
        run_dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        if not run_dirs:
            return BlindingAuditResult(
                run_id="unknown",
                timestamp=datetime.now(tz=UTC).isoformat(),
                recommendation="No runs found in workspace.",
            )
        run_dir = run_dirs[0]
        run_id = run_dir.name

    if not run_dir.exists():
        return BlindingAuditResult(
            run_id=run_id,
            timestamp=datetime.now(tz=UTC).isoformat(),
            recommendation=f"Run directory not found: {run_dir}",
        )

    result = run_blinding_audit(
        run_dir, run_id, contamination_threshold=contamination_threshold
    )

    if store_results:
        db_path = workspace / ".blinding_audits.db"
        store = AuditStore(db_path)
        try:
            store.store_audit(result)
            logger.info(
                "Stored blinding audit for run %s (score=%.3f)",
                run_id,
                result.aggregate_score,
            )
        finally:
            store.close()

    return result
