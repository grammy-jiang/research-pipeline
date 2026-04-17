"""Case-Based Reasoning (CBR) for non-parametric self-evolution.

Stores successful research strategies from past pipeline runs and retrieves
similar cases to inform new runs. Implements the classic CBR cycle:
Retrieve → Reuse → Revise → Retain.

Based on the Deep Research Agents Roadmap (arXiv 2506.18096) finding that
CBR enables agents to improve without retraining by accumulating
institutional knowledge from past usage patterns.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CBR_DIR = Path.home() / ".cache" / "research-pipeline"
DEFAULT_CBR_PATH = DEFAULT_CBR_DIR / "cbr_cases.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    query_terms TEXT DEFAULT '[]',
    sources_used TEXT DEFAULT '[]',
    screening_config TEXT DEFAULT '{}',
    pipeline_profile TEXT DEFAULT 'standard',
    paper_count INTEGER DEFAULT 0,
    shortlist_count INTEGER DEFAULT 0,
    synthesis_quality REAL DEFAULT 0.0,
    pass_at_k REAL DEFAULT 0.0,
    contamination_score REAL DEFAULT 0.0,
    outcome TEXT DEFAULT 'unknown',
    strategy_notes TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_cases_topic ON cases(topic);
CREATE INDEX IF NOT EXISTS idx_cases_outcome ON cases(outcome);
CREATE INDEX IF NOT EXISTS idx_cases_quality ON cases(synthesis_quality);

CREATE TABLE IF NOT EXISTS case_adaptations (
    adaptation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_case_id TEXT NOT NULL,
    target_case_id TEXT NOT NULL,
    adaptation_type TEXT NOT NULL,
    changes_applied TEXT DEFAULT '{}',
    quality_delta REAL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_case_id) REFERENCES cases(case_id),
    FOREIGN KEY (target_case_id) REFERENCES cases(case_id)
);
CREATE INDEX IF NOT EXISTS idx_adapt_source ON case_adaptations(source_case_id);
"""


@dataclass
class Case:
    """A stored research strategy case."""

    case_id: str
    topic: str
    query_terms: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    screening_config: dict = field(default_factory=dict)
    pipeline_profile: str = "standard"
    paper_count: int = 0
    shortlist_count: int = 0
    synthesis_quality: float = 0.0
    pass_at_k: float = 0.0
    contamination_score: float = 0.0
    outcome: str = "unknown"
    strategy_notes: str = ""
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CaseMatch:
    """A retrieved case with similarity score."""

    case: Case
    similarity: float
    match_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "case": self.case.to_dict(),
            "similarity": self.similarity,
            "match_reasons": self.match_reasons,
        }


@dataclass
class Adaptation:
    """Record of adapting a source case for a new target."""

    source_case_id: str
    target_case_id: str
    adaptation_type: str
    changes_applied: dict = field(default_factory=dict)
    quality_delta: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class StrategyRecommendation:
    """A recommended strategy derived from past cases."""

    recommended_sources: list[str] = field(default_factory=list)
    recommended_profile: str = "standard"
    recommended_query_terms: list[str] = field(default_factory=list)
    screening_hints: dict = field(default_factory=dict)
    confidence: float = 0.0
    basis_cases: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Text similarity (BM25-inspired term overlap)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _term_overlap_similarity(query_tokens: list[str], case_tokens: list[str]) -> float:
    """Compute Jaccard-like term overlap between query and case tokens."""
    if not query_tokens or not case_tokens:
        return 0.0
    query_set = set(query_tokens)
    case_set = set(case_tokens)
    intersection = query_set & case_set
    union = query_set | case_set
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _idf_weighted_similarity(
    query_tokens: list[str],
    case_tokens: list[str],
    doc_freq: dict[str, int],
    total_docs: int,
) -> float:
    """IDF-weighted term overlap similarity."""
    if not query_tokens or not case_tokens:
        return 0.0

    query_set = set(query_tokens)
    case_set = set(case_tokens)
    common = query_set & case_set

    if not common:
        return 0.0

    score = 0.0
    max_score = 0.0
    for term in query_set:
        df = doc_freq.get(term, 0)
        idf = math.log((total_docs + 1) / (df + 1)) + 1.0
        max_score += idf
        if term in common:
            score += idf

    if max_score == 0.0:
        return 0.0
    return score / max_score


# ---------------------------------------------------------------------------
# Outcome-based quality weighting
# ---------------------------------------------------------------------------

_OUTCOME_WEIGHTS = {
    "excellent": 1.0,
    "good": 0.8,
    "adequate": 0.6,
    "poor": 0.3,
    "failed": 0.0,
    "unknown": 0.4,
}


def _outcome_weight(outcome: str) -> float:
    """Get quality weight for an outcome label."""
    return _OUTCOME_WEIGHTS.get(outcome.lower(), 0.4)


# ---------------------------------------------------------------------------
# CaseStore — SQLite persistence
# ---------------------------------------------------------------------------


class CaseStore:
    """SQLite-backed case store for CBR."""

    def __init__(self, db_path: Path = DEFAULT_CBR_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def store_case(self, case: Case) -> str:
        """Store a case and return its ID."""
        self._conn.execute(
            """INSERT OR REPLACE INTO cases
               (case_id, topic, query_terms, sources_used, screening_config,
                pipeline_profile, paper_count, shortlist_count,
                synthesis_quality, pass_at_k, contamination_score,
                outcome, strategy_notes, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case.case_id,
                case.topic,
                json.dumps(case.query_terms),
                json.dumps(case.sources_used),
                json.dumps(case.screening_config),
                case.pipeline_profile,
                case.paper_count,
                case.shortlist_count,
                case.synthesis_quality,
                case.pass_at_k,
                case.contamination_score,
                case.outcome,
                case.strategy_notes,
                json.dumps(case.metadata),
            ),
        )
        self._conn.commit()
        return case.case_id

    def get_case(self, case_id: str) -> Case | None:
        """Retrieve a case by ID."""
        row = self._conn.execute(
            "SELECT * FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_case(row)

    def get_all_cases(self) -> list[Case]:
        """Retrieve all stored cases."""
        rows = self._conn.execute(
            "SELECT * FROM cases ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_case(r) for r in rows]

    def get_successful_cases(self, min_quality: float = 0.5) -> list[Case]:
        """Retrieve cases with synthesis quality above threshold."""
        sql = (
            "SELECT * FROM cases"
            " WHERE synthesis_quality >= ?"
            " ORDER BY synthesis_quality DESC"
        )
        rows = self._conn.execute(
            sql,
            (min_quality,),
        ).fetchall()
        return [self._row_to_case(r) for r in rows]

    def store_adaptation(self, adaptation: Adaptation) -> int:
        """Store an adaptation record and return its ID."""
        cursor = self._conn.execute(
            """INSERT INTO case_adaptations
               (source_case_id, target_case_id, adaptation_type,
                changes_applied, quality_delta)
               VALUES (?, ?, ?, ?, ?)""",
            (
                adaptation.source_case_id,
                adaptation.target_case_id,
                adaptation.adaptation_type,
                json.dumps(adaptation.changes_applied),
                adaptation.quality_delta,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get_adaptations(self, case_id: str) -> list[Adaptation]:
        """Get all adaptations where case_id was the source."""
        rows = self._conn.execute(
            "SELECT * FROM case_adaptations WHERE source_case_id = ?",
            (case_id,),
        ).fetchall()
        return [
            Adaptation(
                source_case_id=r["source_case_id"],
                target_case_id=r["target_case_id"],
                adaptation_type=r["adaptation_type"],
                changes_applied=json.loads(r["changes_applied"]),
                quality_delta=r["quality_delta"],
            )
            for r in rows
        ]

    def case_count(self) -> int:
        """Return total number of stored cases."""
        row = self._conn.execute("SELECT COUNT(*) FROM cases").fetchone()
        return row[0] if row else 0

    @staticmethod
    def _row_to_case(row: sqlite3.Row) -> Case:
        """Convert a database row to a Case object."""
        return Case(
            case_id=row["case_id"],
            topic=row["topic"],
            query_terms=json.loads(row["query_terms"]),
            sources_used=json.loads(row["sources_used"]),
            screening_config=json.loads(row["screening_config"]),
            pipeline_profile=row["pipeline_profile"],
            paper_count=row["paper_count"],
            shortlist_count=row["shortlist_count"],
            synthesis_quality=row["synthesis_quality"],
            pass_at_k=row["pass_at_k"],
            contamination_score=row["contamination_score"],
            outcome=row["outcome"],
            strategy_notes=row["strategy_notes"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"]),
        )


# ---------------------------------------------------------------------------
# RETRIEVE — find similar past cases
# ---------------------------------------------------------------------------


def retrieve_similar_cases(
    topic: str,
    store: CaseStore,
    *,
    max_results: int = 5,
    min_similarity: float = 0.1,
    min_quality: float = 0.0,
) -> list[CaseMatch]:
    """Retrieve cases similar to the given topic.

    Uses IDF-weighted term overlap when enough cases exist, otherwise
    falls back to simple Jaccard similarity.

    Args:
        topic: The research topic to match against.
        store: CaseStore to search.
        max_results: Maximum number of results to return.
        min_similarity: Minimum similarity score to include.
        min_quality: Minimum synthesis quality to consider.

    Returns:
        List of CaseMatch objects sorted by similarity (descending).
    """
    all_cases = store.get_all_cases()
    if not all_cases:
        return []

    query_tokens = _tokenize(topic)
    if not query_tokens:
        return []

    # Build document frequency for IDF
    doc_freq: dict[str, int] = {}
    case_tokens_map: dict[str, list[str]] = {}
    for case in all_cases:
        tokens = _tokenize(case.topic) + [t.lower() for t in case.query_terms]
        case_tokens_map[case.case_id] = tokens
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    total_docs = len(all_cases)
    matches: list[CaseMatch] = []

    for case in all_cases:
        if case.synthesis_quality < min_quality:
            continue

        case_tokens = case_tokens_map[case.case_id]

        # Compute similarity
        if total_docs >= 3:
            sim = _idf_weighted_similarity(
                query_tokens, case_tokens, doc_freq, total_docs
            )
        else:
            sim = _term_overlap_similarity(query_tokens, case_tokens)

        # Boost by outcome quality
        outcome_boost = _outcome_weight(case.outcome) * 0.2
        sim = min(1.0, sim + outcome_boost)

        if sim < min_similarity:
            continue

        # Build match reasons
        reasons: list[str] = []
        query_set = set(query_tokens)
        case_set = set(case_tokens)
        common = query_set & case_set
        if common:
            reasons.append(f"shared terms: {', '.join(sorted(common)[:5])}")
        if case.outcome in ("excellent", "good"):
            reasons.append(f"high-quality outcome: {case.outcome}")
        if case.synthesis_quality >= 0.7:
            reasons.append(f"quality={case.synthesis_quality:.2f}")

        matches.append(CaseMatch(case=case, similarity=sim, match_reasons=reasons))

    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches[:max_results]


# ---------------------------------------------------------------------------
# REUSE — generate strategy recommendation from retrieved cases
# ---------------------------------------------------------------------------


def recommend_strategy(
    topic: str,
    matches: list[CaseMatch],
) -> StrategyRecommendation:
    """Generate a strategy recommendation from matched cases.

    Aggregates successful strategies from similar past cases, weighted
    by similarity and outcome quality.

    Args:
        topic: Current research topic.
        matches: Retrieved similar cases with similarity scores.

    Returns:
        A StrategyRecommendation with weighted suggestions.
    """
    if not matches:
        return StrategyRecommendation(
            recommended_sources=["arxiv"],
            recommended_profile="standard",
            confidence=0.0,
            reasoning="No similar past cases found; using defaults.",
        )

    # Weighted vote on sources
    source_scores: dict[str, float] = {}
    profile_scores: dict[str, float] = {}
    all_terms: dict[str, float] = {}
    screening_hints: dict[str, float] = {}

    total_weight = 0.0
    basis_ids: list[str] = []

    for match in matches:
        weight = match.similarity * _outcome_weight(match.case.outcome)
        total_weight += weight
        basis_ids.append(match.case.case_id)

        for source in match.case.sources_used:
            source_scores[source] = source_scores.get(source, 0.0) + weight

        profile_scores[match.case.pipeline_profile] = (
            profile_scores.get(match.case.pipeline_profile, 0.0) + weight
        )

        for term in match.case.query_terms:
            all_terms[term.lower()] = all_terms.get(term.lower(), 0.0) + weight

        for key, value in match.case.screening_config.items():
            if isinstance(value, int | float):
                screening_hints[key] = screening_hints.get(key, 0.0) + value * weight

    # Normalize
    if total_weight > 0:
        for key in screening_hints:
            screening_hints[key] /= total_weight

    # Pick top sources
    sorted_sources = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
    rec_sources = [s for s, _ in sorted_sources[:3]] if sorted_sources else ["arxiv"]

    # Pick best profile
    sorted_profiles = sorted(profile_scores.items(), key=lambda x: x[1], reverse=True)
    rec_profile = sorted_profiles[0][0] if sorted_profiles else "standard"

    # Pick top query terms not already in topic
    topic_tokens = set(_tokenize(topic))
    sorted_terms = sorted(all_terms.items(), key=lambda x: x[1], reverse=True)
    rec_terms = [t for t, _ in sorted_terms if t not in topic_tokens][:10]

    # Confidence based on match quality
    best_sim = matches[0].similarity if matches else 0.0
    avg_quality = (
        sum(m.case.synthesis_quality for m in matches) / len(matches)
        if matches
        else 0.0
    )
    confidence = min(1.0, best_sim * 0.6 + avg_quality * 0.4)

    # Build reasoning
    reasoning_parts = [
        f"Based on {len(matches)} similar past case(s).",
        f"Best match similarity: {best_sim:.2f}.",
        f"Average past quality: {avg_quality:.2f}.",
    ]
    if rec_sources:
        reasoning_parts.append(f"Recommended sources: {', '.join(rec_sources)}.")
    if rec_terms:
        reasoning_parts.append(
            f"Suggested additional terms: {', '.join(rec_terms[:5])}."
        )

    return StrategyRecommendation(
        recommended_sources=rec_sources,
        recommended_profile=rec_profile,
        recommended_query_terms=rec_terms,
        screening_hints=screening_hints,
        confidence=confidence,
        basis_cases=basis_ids,
        reasoning=" ".join(reasoning_parts),
    )


# ---------------------------------------------------------------------------
# RETAIN — create a new case from a completed run
# ---------------------------------------------------------------------------


def create_case_from_run(
    run_id: str,
    topic: str,
    workspace: Path,
    *,
    outcome: str = "unknown",
    strategy_notes: str = "",
) -> Case:
    """Create a Case from a completed pipeline run's artifacts.

    Reads the run manifest and synthesis report to extract strategy info.

    Args:
        run_id: The pipeline run identifier.
        topic: Research topic for this run.
        workspace: Workspace directory path.
        outcome: Quality outcome label (excellent/good/adequate/poor/failed).
        strategy_notes: Free-text notes about the strategy used.

    Returns:
        A populated Case object (not yet stored).
    """
    run_dir = workspace / "runs" / run_id

    # Extract query terms from plan
    query_terms: list[str] = []
    plan_dir = run_dir / "plan"
    if plan_dir.exists():
        for plan_file in plan_dir.glob("*.json"):
            try:
                plan_data = json.loads(plan_file.read_text())
                if isinstance(plan_data, dict):
                    for key in ("query_variants", "queries", "terms"):
                        if key in plan_data and isinstance(plan_data[key], list):
                            query_terms.extend(
                                str(t) for t in plan_data[key] if isinstance(t, str)
                            )
            except (json.JSONDecodeError, OSError):
                pass

    # Extract source info from search
    sources_used: list[str] = []
    search_dir = run_dir / "search"
    if search_dir.exists():
        for search_file in search_dir.glob("*.jsonl"):
            try:
                for line in search_file.read_text().splitlines():
                    rec = json.loads(line)
                    if isinstance(rec, dict) and "source" in rec:
                        src = rec["source"]
                        if src not in sources_used:
                            sources_used.append(src)
            except (json.JSONDecodeError, OSError):
                pass

    # Extract paper counts
    paper_count = 0
    shortlist_count = 0
    screen_dir = run_dir / "screen"
    if screen_dir.exists():
        for screen_file in screen_dir.glob("*.jsonl"):
            try:
                lines = screen_file.read_text().strip().splitlines()
                paper_count = len(lines)
                for line in lines:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and rec.get("decision") == "INCLUDE":
                        shortlist_count += 1
            except (json.JSONDecodeError, OSError):
                pass

    # Extract synthesis quality
    synthesis_quality = 0.0
    summarize_dir = run_dir / "summarize"
    if summarize_dir.exists():
        synthesis_file = summarize_dir / "synthesis_report.json"
        if synthesis_file.exists():
            try:
                data = json.loads(synthesis_file.read_text())
                if isinstance(data, dict):
                    summaries = data.get("per_paper_summaries", [])
                    findings_count = sum(
                        len(s.get("findings", []))
                        for s in summaries
                        if isinstance(s, dict)
                    )
                    evidence_count = sum(
                        len(s.get("evidence", []))
                        for s in summaries
                        if isinstance(s, dict)
                    )
                    gaps = data.get("gaps", [])

                    # Heuristic quality score
                    findings_score = min(1.0, findings_count / 10.0) * 0.4
                    evidence_score = (
                        min(1.0, evidence_count / findings_count)
                        if findings_count > 0
                        else 0.0
                    ) * 0.3
                    gap_score = min(1.0, len(gaps) / 3.0) * 0.2 if gaps else 0.0
                    coverage_score = (
                        min(1.0, len(summaries) / 5.0) * 0.1 if summaries else 0.0
                    )
                    synthesis_quality = (
                        findings_score + evidence_score + gap_score + coverage_score
                    )
            except (json.JSONDecodeError, OSError):
                pass

    # Read manifest for profile
    pipeline_profile = "standard"
    manifest_file = run_dir / "run_manifest.json"
    if manifest_file.exists():
        try:
            manifest = json.loads(manifest_file.read_text())
            if isinstance(manifest, dict):
                pipeline_profile = manifest.get("profile", "standard")
        except (json.JSONDecodeError, OSError):
            pass

    return Case(
        case_id=run_id,
        topic=topic,
        query_terms=query_terms[:20],
        sources_used=sources_used,
        pipeline_profile=pipeline_profile,
        paper_count=paper_count,
        shortlist_count=shortlist_count,
        synthesis_quality=synthesis_quality,
        outcome=outcome,
        strategy_notes=strategy_notes,
    )


# ---------------------------------------------------------------------------
# REVISE — record adaptation when a case is modified for reuse
# ---------------------------------------------------------------------------


def record_adaptation(
    store: CaseStore,
    source_case_id: str,
    target_case_id: str,
    *,
    adaptation_type: str = "parameter_adjustment",
    changes: dict | None = None,
    quality_delta: float = 0.0,
) -> Adaptation:
    """Record that a source case was adapted for a target run.

    Args:
        store: CaseStore for persistence.
        source_case_id: The case that was adapted from.
        target_case_id: The new case that used the adaptation.
        adaptation_type: Type of adaptation (e.g., parameter_adjustment,
            source_swap, query_expansion).
        changes: Dictionary describing what changed.
        quality_delta: Change in quality (positive = improvement).

    Returns:
        The stored Adaptation record.
    """
    adaptation = Adaptation(
        source_case_id=source_case_id,
        target_case_id=target_case_id,
        adaptation_type=adaptation_type,
        changes_applied=changes or {},
        quality_delta=quality_delta,
    )
    store.store_adaptation(adaptation)
    return adaptation


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def cbr_lookup(
    topic: str,
    workspace: Path,
    *,
    db_path: Path | None = None,
    max_results: int = 5,
    min_quality: float = 0.0,
) -> StrategyRecommendation:
    """Look up past cases and recommend a strategy for a new topic.

    Convenience function that opens the store, retrieves similar cases,
    and generates a recommendation.

    Args:
        topic: Research topic for the new run.
        workspace: Workspace directory path.
        db_path: Optional override for database path.
        max_results: Maximum cases to consider.
        min_quality: Minimum synthesis quality filter.

    Returns:
        StrategyRecommendation with suggested parameters.
    """
    path = db_path or (workspace / ".cbr_cases.db")
    store = CaseStore(path)
    try:
        matches = retrieve_similar_cases(
            topic, store, max_results=max_results, min_quality=min_quality
        )
        return recommend_strategy(topic, matches)
    finally:
        store.close()


def cbr_retain(
    run_id: str,
    topic: str,
    workspace: Path,
    *,
    outcome: str = "unknown",
    strategy_notes: str = "",
    db_path: Path | None = None,
) -> Case:
    """Create and store a case from a completed pipeline run.

    Args:
        run_id: The pipeline run identifier.
        topic: Research topic for this run.
        workspace: Workspace directory path.
        outcome: Quality outcome label.
        strategy_notes: Free-text notes.
        db_path: Optional override for database path.

    Returns:
        The stored Case object.
    """
    path = db_path or (workspace / ".cbr_cases.db")
    case = create_case_from_run(
        run_id, topic, workspace, outcome=outcome, strategy_notes=strategy_notes
    )
    store = CaseStore(path)
    try:
        store.store_case(case)
        return case
    finally:
        store.close()
