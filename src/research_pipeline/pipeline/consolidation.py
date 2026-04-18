"""Memory consolidation engine for cross-run knowledge management.

Implements episodic → semantic consolidation following the deep research
three-tier memory architecture (SEA, MLMF, ES-Mem patterns):
- Compress old run episodes into thematic summaries
- Promote repeated findings to stable rules
- Prune stale or superseded knowledge entries
- Detect and flag semantic drift between runs

Based on evidence from 10+ papers:
- SEA (2604.07269): explicit consolidate action improves 92.46% vs 47.41%
- MLMF (2603.29194): retention regularization prevents drift (FMR 5.1→6.9%)
- ES-Mem (2601.07582): episodic→semantic promotion with domain templates
- WebATLAS (2510.22732): curated memory +12.6pp vs raw -3.1pp
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Configuration defaults ──────────────────────────────────────────

EPISODE_CAPACITY = 100
CONSOLIDATION_THRESHOLD = 0.8  # trigger at 80% capacity
STALENESS_DAYS = 90  # prune entries older than this if unreferenced
MIN_SUPPORT_FOR_RULE = 2  # minimum run appearances to promote to rule
DRIFT_PENALTY_WEIGHT = 0.15  # retention regularization strength


# ── Data classes ────────────────────────────────────────────────────


@dataclass
class Episode:
    """A single run episode — summary of one pipeline run."""

    run_id: str
    topic: str
    timestamp: str  # ISO 8601
    paper_count: int = 0
    findings: list[str] = field(default_factory=list)
    agreements: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def content_hash(self) -> str:
        """SHA-256 hash of episode content for dedup."""
        blob = json.dumps(
            {
                "run_id": self.run_id,
                "topic": self.topic,
                "findings": sorted(self.findings),
                "agreements": sorted(self.agreements),
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Rule:
    """A consolidated semantic rule promoted from recurring findings."""

    rule_id: str
    statement: str
    supporting_runs: list[str] = field(default_factory=list)
    confidence: float = 0.0  # 0.0 - 1.0
    created_at: str = ""
    updated_at: str = ""
    source_findings: list[str] = field(default_factory=list)


@dataclass
class DriftMetric:
    """Semantic drift measurement between consecutive episodes."""

    run_a: str
    run_b: str
    finding_overlap: float = 0.0  # Jaccard of findings
    topic_shift: float = 0.0  # 1 - cosine of topic words
    drift_score: float = 0.0  # composite


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""

    episodes_before: int = 0
    episodes_after: int = 0
    rules_created: int = 0
    rules_updated: int = 0
    entries_pruned: int = 0
    drift_metrics: list[DriftMetric] = field(default_factory=list)
    timestamp: str = ""


# ── Episode store (SQLite-backed) ──────────────────────────────────

_EPISODE_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    run_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    paper_count INTEGER DEFAULT 0,
    findings TEXT DEFAULT '[]',
    agreements TEXT DEFAULT '[]',
    open_questions TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    content_hash TEXT DEFAULT '',
    consolidated INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rules (
    rule_id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    supporting_runs TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source_findings TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS drift_log (
    run_a TEXT NOT NULL,
    run_b TEXT NOT NULL,
    finding_overlap REAL DEFAULT 0.0,
    topic_shift REAL DEFAULT 0.0,
    drift_score REAL DEFAULT 0.0,
    measured_at TEXT NOT NULL,
    PRIMARY KEY (run_a, run_b)
);
"""


class EpisodeStore:
    """SQLite-backed store for episodes, rules, and drift metrics."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_EPISODE_SCHEMA)
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Episode CRUD ────────────────────────────────────────────────

    def add_episode(self, episode: Episode) -> None:
        """Insert or replace an episode."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO episodes
               (run_id, topic, timestamp, paper_count, findings,
                agreements, open_questions, metadata, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.run_id,
                episode.topic,
                episode.timestamp,
                episode.paper_count,
                json.dumps(episode.findings),
                json.dumps(episode.agreements),
                json.dumps(episode.open_questions),
                json.dumps(episode.metadata),
                episode.content_hash(),
            ),
        )
        conn.commit()

    def get_episode(self, run_id: str) -> Episode | None:
        """Retrieve an episode by run_id."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM episodes WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_episode(row)

    def list_episodes(
        self,
        *,
        consolidated: bool | None = None,
        limit: int | None = None,
    ) -> list[Episode]:
        """List episodes, optionally filtered by consolidation status."""
        conn = self._get_conn()
        query = "SELECT * FROM episodes"
        params: list[object] = []
        if consolidated is not None:
            query += " WHERE consolidated = ?"
            params.append(1 if consolidated else 0)
        query += " ORDER BY timestamp ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [_row_to_episode(r) for r in rows]

    def count_episodes(self, *, consolidated: bool | None = None) -> int:
        """Count episodes, optionally filtered."""
        conn = self._get_conn()
        query = "SELECT COUNT(*) FROM episodes"
        params: list[object] = []
        if consolidated is not None:
            query += " WHERE consolidated = ?"
            params.append(1 if consolidated else 0)
        return conn.execute(query, params).fetchone()[0]

    def mark_consolidated(self, run_ids: list[str]) -> None:
        """Mark episodes as consolidated."""
        if not run_ids:
            return
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in run_ids)
        conn.execute(
            f"UPDATE episodes SET consolidated = 1 WHERE run_id IN ({placeholders})",  # nosec B608
            run_ids,
        )
        conn.commit()

    def delete_episodes(self, run_ids: list[str]) -> int:
        """Delete episodes by run_id. Returns count deleted."""
        if not run_ids:
            return 0
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in run_ids)
        cursor = conn.execute(
            f"DELETE FROM episodes WHERE run_id IN ({placeholders})",  # nosec B608
            run_ids,
        )
        conn.commit()
        return cursor.rowcount

    # ── Rule CRUD ───────────────────────────────────────────────────

    def add_rule(self, rule: Rule) -> None:
        """Insert or replace a rule."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO rules
               (rule_id, statement, supporting_runs, confidence,
                created_at, updated_at, source_findings)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                rule.rule_id,
                rule.statement,
                json.dumps(rule.supporting_runs),
                rule.confidence,
                rule.created_at,
                rule.updated_at,
                json.dumps(rule.source_findings),
            ),
        )
        conn.commit()

    def get_rule(self, rule_id: str) -> Rule | None:
        """Retrieve a rule by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM rules WHERE rule_id = ?", (rule_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_rule(row)

    def list_rules(self, *, min_confidence: float = 0.0) -> list[Rule]:
        """List rules above a confidence threshold."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM rules WHERE confidence >= ? ORDER BY confidence DESC",
            (min_confidence,),
        ).fetchall()
        return [_row_to_rule(r) for r in rows]

    def delete_rules(self, rule_ids: list[str]) -> int:
        """Delete rules by ID. Returns count deleted."""
        if not rule_ids:
            return 0
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in rule_ids)
        cursor = conn.execute(
            f"DELETE FROM rules WHERE rule_id IN ({placeholders})",  # nosec B608
            rule_ids,
        )
        conn.commit()
        return cursor.rowcount

    # ── Drift log ───────────────────────────────────────────────────

    def log_drift(self, metric: DriftMetric) -> None:
        """Record a drift measurement."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO drift_log
               (run_a, run_b, finding_overlap, topic_shift,
                drift_score, measured_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                metric.run_a,
                metric.run_b,
                metric.finding_overlap,
                metric.topic_shift,
                metric.drift_score,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()

    def list_drift(self) -> list[DriftMetric]:
        """List all drift measurements."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM drift_log ORDER BY measured_at ASC"
        ).fetchall()
        return [
            DriftMetric(
                run_a=r["run_a"],
                run_b=r["run_b"],
                finding_overlap=r["finding_overlap"],
                topic_shift=r["topic_shift"],
                drift_score=r["drift_score"],
            )
            for r in rows
        ]


# ── Helper functions ────────────────────────────────────────────────


def _row_to_episode(row: sqlite3.Row) -> Episode:
    """Convert a database row to an Episode."""
    return Episode(
        run_id=row["run_id"],
        topic=row["topic"],
        timestamp=row["timestamp"],
        paper_count=row["paper_count"],
        findings=json.loads(row["findings"]),
        agreements=json.loads(row["agreements"]),
        open_questions=json.loads(row["open_questions"]),
        metadata=json.loads(row["metadata"]),
    )


def _row_to_rule(row: sqlite3.Row) -> Rule:
    """Convert a database row to a Rule."""
    return Rule(
        rule_id=row["rule_id"],
        statement=row["statement"],
        supporting_runs=json.loads(row["supporting_runs"]),
        confidence=row["confidence"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        source_findings=json.loads(row["source_findings"]),
    )


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase word set (stop words removed)."""
    stop = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "and",
        "but",
        "or",
        "not",
        "no",
        "that",
        "this",
        "it",
        "its",
    }
    words = set(text.lower().split())
    return words - stop


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalize_finding(text: str) -> str:
    """Normalize a finding string for comparison."""
    return " ".join(text.lower().split())


# ── Core consolidation logic ────────────────────────────────────────


def extract_episode_from_run(
    workspace: Path,
    run_id: str,
) -> Episode | None:
    """Extract an episode from a run's synthesis output.

    Reads the synthesis_report.json from the run's summarize directory.

    Args:
        workspace: Base workspace directory.
        run_id: Run identifier.

    Returns:
        Episode if synthesis data found, None otherwise.
    """
    run_dir = workspace / run_id
    synthesis_path = run_dir / "summarize" / "synthesis_report.json"

    if not synthesis_path.exists():
        logger.debug("No synthesis report for run %s", run_id)
        return None

    try:
        data = json.loads(synthesis_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read synthesis for %s: %s", run_id, exc)
        return None

    topic = data.get("topic", "")
    paper_count = data.get("paper_count", 0)

    findings: list[str] = []
    agreements: list[str] = []
    open_questions: list[str] = []

    for a in data.get("agreements", []):
        claim = a.get("claim", "")
        if claim:
            agreements.append(claim)
            findings.append(claim)

    for d in data.get("disagreements", []):
        topic_str = d.get("topic", "")
        if topic_str:
            findings.append(f"[contested] {topic_str}")

    for q in data.get("open_questions", []):
        if q:
            open_questions.append(q)

    return Episode(
        run_id=run_id,
        topic=topic,
        timestamp=datetime.now(UTC).isoformat(),
        paper_count=paper_count,
        findings=findings,
        agreements=agreements,
        open_questions=open_questions,
    )


def measure_drift(ep_a: Episode, ep_b: Episode) -> DriftMetric:
    """Measure semantic drift between two consecutive episodes.

    Args:
        ep_a: Earlier episode.
        ep_b: Later episode.

    Returns:
        DriftMetric with overlap and shift scores.
    """
    # Finding overlap (Jaccard of normalized findings)
    findings_a = {_normalize_finding(f) for f in ep_a.findings}
    findings_b = {_normalize_finding(f) for f in ep_b.findings}

    finding_tokens_a = set()
    for f in findings_a:
        finding_tokens_a.update(_tokenize(f))
    finding_tokens_b = set()
    for f in findings_b:
        finding_tokens_b.update(_tokenize(f))

    finding_overlap = _jaccard(finding_tokens_a, finding_tokens_b)

    # Topic shift
    topic_a = _tokenize(ep_a.topic)
    topic_b = _tokenize(ep_b.topic)
    topic_similarity = _jaccard(topic_a, topic_b)
    topic_shift = 1.0 - topic_similarity

    # Composite drift (higher = more drift)
    drift_score = (1.0 - finding_overlap) * 0.7 + topic_shift * 0.3

    return DriftMetric(
        run_a=ep_a.run_id,
        run_b=ep_b.run_id,
        finding_overlap=finding_overlap,
        topic_shift=topic_shift,
        drift_score=drift_score,
    )


def find_recurring_findings(
    episodes: list[Episode],
    min_support: int = MIN_SUPPORT_FOR_RULE,
) -> list[tuple[str, list[str]]]:
    """Find findings that recur across multiple episodes.

    Args:
        episodes: List of episodes to analyze.
        min_support: Minimum number of runs a finding must appear in.

    Returns:
        List of (canonical_finding, [supporting_run_ids]).
    """
    # Group findings by normalized form
    finding_runs: dict[str, list[str]] = {}
    finding_canonical: dict[str, str] = {}

    for ep in episodes:
        seen_in_ep: set[str] = set()
        for f in ep.findings:
            norm = _normalize_finding(f)
            if norm in seen_in_ep:
                continue
            seen_in_ep.add(norm)

            # Use token-based fuzzy matching
            norm_tokens = _tokenize(norm)
            matched = False
            for existing_norm in list(finding_runs.keys()):
                existing_tokens = _tokenize(existing_norm)
                if _jaccard(norm_tokens, existing_tokens) > 0.6:
                    if ep.run_id not in finding_runs[existing_norm]:
                        finding_runs[existing_norm].append(ep.run_id)
                    matched = True
                    break

            if not matched:
                finding_runs[norm] = [ep.run_id]
                finding_canonical[norm] = f

    # Filter by minimum support
    results: list[tuple[str, list[str]]] = []
    for norm, runs in finding_runs.items():
        if len(runs) >= min_support:
            canonical = finding_canonical.get(norm, norm)
            results.append((canonical, runs))

    return sorted(results, key=lambda x: len(x[1]), reverse=True)


def promote_to_rules(
    recurring: list[tuple[str, list[str]]],
    existing_rules: list[Rule],
) -> tuple[list[Rule], list[Rule]]:
    """Promote recurring findings to rules or update existing ones.

    Args:
        recurring: List of (finding, supporting_runs) from find_recurring.
        existing_rules: Currently stored rules.

    Returns:
        Tuple of (new_rules, updated_rules).
    """
    now = datetime.now(UTC).isoformat()
    existing_by_norm: dict[str, Rule] = {}
    for r in existing_rules:
        existing_by_norm[_normalize_finding(r.statement)] = r

    new_rules: list[Rule] = []
    updated_rules: list[Rule] = []

    for finding, runs in recurring:
        norm = _normalize_finding(finding)
        norm_tokens = _tokenize(norm)

        # Check if matches an existing rule
        matched_rule: Rule | None = None
        for existing_norm, rule in existing_by_norm.items():
            existing_tokens = _tokenize(existing_norm)
            if _jaccard(norm_tokens, existing_tokens) > 0.6:
                matched_rule = rule
                break

        if matched_rule is not None:
            # Update existing rule
            all_runs = list(set(matched_rule.supporting_runs) | set(runs))
            matched_rule.supporting_runs = sorted(all_runs)
            matched_rule.confidence = min(1.0, len(all_runs) / 5.0)
            matched_rule.updated_at = now
            if norm not in matched_rule.source_findings:
                matched_rule.source_findings.append(norm)
            updated_rules.append(matched_rule)
        else:
            # Create new rule
            rule_id = f"rule-{hashlib.sha256(norm.encode()).hexdigest()[:12]}"
            confidence = min(1.0, len(runs) / 5.0)
            new_rule = Rule(
                rule_id=rule_id,
                statement=finding,
                supporting_runs=sorted(runs),
                confidence=confidence,
                created_at=now,
                updated_at=now,
                source_findings=[norm],
            )
            new_rules.append(new_rule)

    return new_rules, updated_rules


def prune_stale_episodes(
    episodes: list[Episode],
    rules: list[Rule],
    staleness_days: int = STALENESS_DAYS,
) -> list[str]:
    """Identify stale episodes safe to prune.

    An episode is stale if:
    - It's older than staleness_days
    - It's already consolidated
    - Its findings are covered by existing rules

    Args:
        episodes: All episodes.
        rules: Current rules.
        staleness_days: Age threshold in days.

    Returns:
        List of run_ids safe to prune.
    """
    now = datetime.now(UTC)
    rule_tokens: list[set[str]] = [_tokenize(r.statement) for r in rules]

    prunable: list[str] = []
    for ep in episodes:
        try:
            ep_time = datetime.fromisoformat(ep.timestamp)
            if ep_time.tzinfo is None:
                ep_time = ep_time.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            continue

        age_days = (now - ep_time).days
        if age_days < staleness_days:
            continue

        # Check if findings are covered by rules
        if not ep.findings:
            prunable.append(ep.run_id)
            continue

        covered_count = 0
        for f in ep.findings:
            f_tokens = _tokenize(f)
            for rt in rule_tokens:
                if _jaccard(f_tokens, rt) > 0.5:
                    covered_count += 1
                    break

        coverage = covered_count / len(ep.findings)
        if coverage >= 0.7:
            prunable.append(ep.run_id)

    return prunable


# ── Main consolidation orchestrator ─────────────────────────────────


def consolidate(
    store: EpisodeStore,
    *,
    capacity: int = EPISODE_CAPACITY,
    threshold: float = CONSOLIDATION_THRESHOLD,
    min_support: int = MIN_SUPPORT_FOR_RULE,
    staleness_days: int = STALENESS_DAYS,
    dry_run: bool = False,
) -> ConsolidationResult:
    """Run the full consolidation cycle.

    Steps:
    1. Check if episode count exceeds capacity threshold
    2. Measure drift between consecutive episodes
    3. Find recurring findings across episodes
    4. Promote to rules (create or update)
    5. Prune stale consolidated episodes
    6. Mark remaining as consolidated

    Args:
        store: EpisodeStore instance.
        capacity: Maximum episode capacity.
        threshold: Fraction of capacity triggering consolidation.
        min_support: Minimum runs for rule promotion.
        staleness_days: Age threshold for pruning.
        dry_run: If True, compute but don't modify store.

    Returns:
        ConsolidationResult with metrics.
    """
    result = ConsolidationResult(
        timestamp=datetime.now(UTC).isoformat(),
    )

    episodes = store.list_episodes()
    result.episodes_before = len(episodes)

    if not episodes:
        result.episodes_after = 0
        return result

    # Step 1: measure drift between consecutive episodes
    sorted_eps = sorted(episodes, key=lambda e: e.timestamp)
    for i in range(len(sorted_eps) - 1):
        drift = measure_drift(sorted_eps[i], sorted_eps[i + 1])
        result.drift_metrics.append(drift)
        if not dry_run:
            store.log_drift(drift)

    # Step 2: check if we need consolidation
    count = len(episodes)
    needs_consolidation = count >= int(capacity * threshold)

    if not needs_consolidation:
        logger.info(
            "Episode count %d below threshold %d, skipping consolidation",
            count,
            int(capacity * threshold),
        )
        result.episodes_after = count
        return result

    logger.info(
        "Episode count %d >= threshold %d, starting consolidation",
        count,
        int(capacity * threshold),
    )

    # Step 3: find recurring findings
    unconsolidated = store.list_episodes(consolidated=False)
    recurring = find_recurring_findings(unconsolidated, min_support)
    logger.info("Found %d recurring findings", len(recurring))

    # Step 4: promote to rules
    existing_rules = store.list_rules()
    new_rules, updated_rules = promote_to_rules(recurring, existing_rules)
    result.rules_created = len(new_rules)
    result.rules_updated = len(updated_rules)

    if not dry_run:
        for r in new_rules:
            store.add_rule(r)
        for r in updated_rules:
            store.add_rule(r)
        # Mark unconsolidated episodes
        store.mark_consolidated([e.run_id for e in unconsolidated])

    # Step 5: prune stale episodes
    all_rules = existing_rules + new_rules
    prunable = prune_stale_episodes(episodes, all_rules, staleness_days)
    result.entries_pruned = len(prunable)

    if not dry_run and prunable:
        store.delete_episodes(prunable)
        logger.info("Pruned %d stale episodes", len(prunable))

    result.episodes_after = result.episodes_before - result.entries_pruned

    return result


def run_consolidation(
    workspace: Path,
    run_ids: list[str] | None = None,
    *,
    db_path: Path | None = None,
    capacity: int = EPISODE_CAPACITY,
    threshold: float = CONSOLIDATION_THRESHOLD,
    min_support: int = MIN_SUPPORT_FOR_RULE,
    staleness_days: int = STALENESS_DAYS,
    dry_run: bool = False,
    output: Path | None = None,
) -> ConsolidationResult:
    """High-level consolidation entry point.

    Ingests episodes from run directories, then runs consolidation.

    Args:
        workspace: Base workspace directory.
        run_ids: Optional list of run IDs to ingest. If None, scans workspace.
        db_path: Path to consolidation database.
            Defaults to workspace/.consolidation.db.
        capacity: Episode capacity.
        threshold: Consolidation trigger threshold.
        min_support: Minimum support for rule promotion.
        staleness_days: Staleness threshold for pruning.
        dry_run: If True, compute but don't modify.
        output: Optional path to write result JSON.

    Returns:
        ConsolidationResult.
    """
    if db_path is None:
        db_path = workspace / ".consolidation.db"

    store = EpisodeStore(db_path)

    try:
        # Ingest episodes from runs
        if run_ids is None:
            # Scan workspace for run directories
            if workspace.exists():
                run_ids = [
                    d.name
                    for d in sorted(workspace.iterdir())
                    if d.is_dir() and not d.name.startswith(".")
                ]
            else:
                run_ids = []

        ingested = 0
        for rid in run_ids:
            if store.get_episode(rid) is not None:
                continue
            ep = extract_episode_from_run(workspace, rid)
            if ep is not None:
                store.add_episode(ep)
                ingested += 1

        if ingested:
            logger.info("Ingested %d new episodes", ingested)

        # Run consolidation
        result = consolidate(
            store,
            capacity=capacity,
            threshold=threshold,
            min_support=min_support,
            staleness_days=staleness_days,
            dry_run=dry_run,
        )

        # Write output
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(asdict(result), indent=2))
            logger.info("Wrote consolidation report to %s", output)

        return result

    finally:
        store.close()
