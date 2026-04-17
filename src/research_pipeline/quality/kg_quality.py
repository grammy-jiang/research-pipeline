"""Knowledge graph quality evaluation framework.

Three-layer composable architecture based on:
- KG Quality Management Survey (TKDE 2022): 5-dimension taxonomy
- Text2KGBench (arXiv 2308.02357): hallucination metrics
- LLM-KG-Bench (arXiv 2308.16622): structural evaluation

Layer 1: Structural metrics (ICR, CI, density, component analysis)
Layer 2: Internal + External consistency (IC contradiction detection,
         EC provenance coverage)
Layer 3: Scalable evaluation via TWCS (Type-Weighted Cluster Sampling)

5-dimension quality framework:
- Accuracy: entity/relation correctness, hallucination detection
- Consistency: IC (no contradictions) + EC (agrees with sources)
- Completeness: type and relation coverage
- Timeliness: temporal freshness of triples
- Redundancy: duplicate triple and near-duplicate entity detection
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructuralMetrics:
    """Layer 1: Structural quality metrics for a knowledge graph.

    ICR  — Inverse Connectivity Ratio: fraction of entities with ≥1 edge.
    CI   — Connectivity Index: ratio of edges to maximum possible edges.
    density — edge count / entity count.
    num_entities — total entity count.
    num_triples  — total triple count.
    connected_components — number of weakly-connected components.
    largest_component_fraction — fraction of entities in the largest component.
    avg_degree — average number of edges per entity.
    type_distribution — entity count per type.
    relation_distribution — triple count per relation type.
    """

    num_entities: int = 0
    num_triples: int = 0
    icr: float = 0.0
    ci: float = 0.0
    density: float = 0.0
    connected_components: int = 0
    largest_component_fraction: float = 0.0
    avg_degree: float = 0.0
    type_distribution: dict[str, int] = field(default_factory=dict)
    relation_distribution: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ConsistencyMetrics:
    """Layer 2: Internal and external consistency metrics.

    ic_contradiction_count — number of detected internal contradictions.
    ic_score — 1.0 - (contradictions / total_triples), clamped [0, 1].
    ec_provenance_coverage — fraction of triples with non-empty provenance.
    ec_score — provenance coverage (proxy for external consistency).
    duplicate_triples — count of exact-duplicate triples.
    """

    ic_contradiction_count: int = 0
    ic_score: float = 1.0
    ec_provenance_coverage: float = 0.0
    ec_score: float = 0.0
    duplicate_triples: int = 0


@dataclass(frozen=True)
class CompletenessMetrics:
    """Coverage metrics for KG completeness.

    entity_type_coverage — fraction of defined entity types present.
    relation_type_coverage — fraction of defined relation types present.
    avg_properties_per_entity — mean number of properties on entities.
    orphan_entities — entities with zero edges.
    missing_entity_types — entity types with zero instances.
    missing_relation_types — relation types with zero instances.
    """

    entity_type_coverage: float = 0.0
    relation_type_coverage: float = 0.0
    avg_properties_per_entity: float = 0.0
    orphan_entities: int = 0
    missing_entity_types: list[str] = field(default_factory=list)
    missing_relation_types: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TimelinessMetrics:
    """Temporal freshness metrics.

    avg_age_days — mean age of triples in days from evaluation time.
    staleness_ratio — fraction of triples older than threshold.
    newest_triple_age_days — age of the most recent triple.
    oldest_triple_age_days — age of the oldest triple.
    """

    avg_age_days: float = 0.0
    staleness_ratio: float = 0.0
    newest_triple_age_days: float = 0.0
    oldest_triple_age_days: float = 0.0


@dataclass(frozen=True)
class RedundancyMetrics:
    """Duplicate and near-duplicate detection metrics.

    exact_duplicate_triples — count of identical (s, r, o) triples.
    near_duplicate_entities — pairs of entities with similar names.
    redundancy_ratio — fraction of redundant content.
    """

    exact_duplicate_triples: int = 0
    near_duplicate_entities: int = 0
    redundancy_ratio: float = 0.0


@dataclass(frozen=True)
class KGQualityScore:
    """Composite KG quality score across all 5 dimensions.

    Each dimension score is [0, 1]. The composite is a weighted average.
    """

    accuracy: float = 0.0
    consistency: float = 0.0
    completeness: float = 0.0
    timeliness: float = 0.0
    redundancy: float = 0.0
    composite: float = 0.0
    structural: StructuralMetrics = field(default_factory=StructuralMetrics)
    consistency_detail: ConsistencyMetrics = field(default_factory=ConsistencyMetrics)
    completeness_detail: CompletenessMetrics = field(
        default_factory=CompletenessMetrics
    )
    timeliness_detail: TimelinessMetrics = field(default_factory=TimelinessMetrics)
    redundancy_detail: RedundancyMetrics = field(default_factory=RedundancyMetrics)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        from dataclasses import asdict

        return asdict(self)


# ---------------------------------------------------------------------------
# Layer 1: Structural analysis
# ---------------------------------------------------------------------------


def compute_structural_metrics(conn: sqlite3.Connection) -> StructuralMetrics:
    """Compute structural metrics from a KG SQLite database.

    Args:
        conn: SQLite connection with KG schema (entities, triples tables).

    Returns:
        StructuralMetrics with ICR, CI, density, components, etc.
    """
    conn.row_factory = sqlite3.Row

    num_entities = _count_rows(conn, "entities")
    num_triples = _count_rows(conn, "triples")

    if num_entities == 0:
        return StructuralMetrics()

    connected = _count_connected_entities(conn)
    icr = connected / num_entities if num_entities > 0 else 0.0

    max_edges = num_entities * (num_entities - 1)
    ci = num_triples / max_edges if max_edges > 0 else 0.0

    density = num_triples / num_entities if num_entities > 0 else 0.0

    adj = _build_adjacency(conn)
    components = _find_components(adj, _all_entity_ids(conn))
    largest_frac = max(len(c) for c in components) / num_entities if components else 0.0

    degree_sum = sum(len(nbrs) for nbrs in adj.values())
    avg_degree = degree_sum / num_entities if num_entities > 0 else 0.0

    type_dist = _distribution(conn, "entities", "entity_type")
    rel_dist = _distribution(conn, "triples", "relation")

    return StructuralMetrics(
        num_entities=num_entities,
        num_triples=num_triples,
        icr=round(icr, 4),
        ci=round(ci, 6),
        density=round(density, 4),
        connected_components=len(components),
        largest_component_fraction=round(largest_frac, 4),
        avg_degree=round(avg_degree, 4),
        type_distribution=type_dist,
        relation_distribution=rel_dist,
    )


# ---------------------------------------------------------------------------
# Layer 2: Consistency
# ---------------------------------------------------------------------------


def compute_consistency_metrics(
    conn: sqlite3.Connection,
) -> ConsistencyMetrics:
    """Compute internal and external consistency metrics.

    Internal consistency (IC): detects contradictions — pairs of triples
    where the same subject–object pair has both SUPPORTS and CONTRADICTS
    relations (or similar opposing relations).

    External consistency (EC): fraction of triples with provenance
    (provenance_paper or provenance_run is non-empty).

    Args:
        conn: SQLite connection with KG schema.

    Returns:
        ConsistencyMetrics.
    """
    conn.row_factory = sqlite3.Row
    num_triples = _count_rows(conn, "triples")

    if num_triples == 0:
        return ConsistencyMetrics()

    contradictions = _count_contradictions(conn)
    ic_score = max(0.0, 1.0 - contradictions / num_triples)

    provenance_count = conn.execute(
        "SELECT COUNT(*) FROM triples"
        " WHERE provenance_paper != '' OR provenance_run != ''"
    ).fetchone()[0]
    ec_coverage = provenance_count / num_triples

    duplicates = _count_duplicate_triples(conn)

    return ConsistencyMetrics(
        ic_contradiction_count=contradictions,
        ic_score=round(ic_score, 4),
        ec_provenance_coverage=round(ec_coverage, 4),
        ec_score=round(ec_coverage, 4),
        duplicate_triples=duplicates,
    )


# ---------------------------------------------------------------------------
# Layer 2.5: Completeness
# ---------------------------------------------------------------------------

# Canonical entity and relation types from the schema
_ENTITY_TYPES = [
    "paper",
    "concept",
    "method",
    "experiment",
    "claim",
    "author",
    "venue",
]
_RELATION_TYPES = [
    "cites",
    "uses_method",
    "proposes_method",
    "addresses_concept",
    "evaluates_on",
    "related_to",
    "supports_claim",
    "contradicts_claim",
    "authored_by",
    "published_in",
]


def compute_completeness_metrics(
    conn: sqlite3.Connection,
    expected_entity_types: list[str] | None = None,
    expected_relation_types: list[str] | None = None,
) -> CompletenessMetrics:
    """Compute KG completeness metrics.

    Args:
        conn: SQLite connection with KG schema.
        expected_entity_types: Expected entity types (default: all 7).
        expected_relation_types: Expected relation types (default: all 10).

    Returns:
        CompletenessMetrics.
    """
    conn.row_factory = sqlite3.Row

    e_types = expected_entity_types or _ENTITY_TYPES
    r_types = expected_relation_types or _RELATION_TYPES

    type_dist = _distribution(conn, "entities", "entity_type")
    rel_dist = _distribution(conn, "triples", "relation")

    present_e = [t for t in e_types if type_dist.get(t, 0) > 0]
    present_r = [r for r in r_types if rel_dist.get(r, 0) > 0]

    e_coverage = len(present_e) / len(e_types) if e_types else 0.0
    r_coverage = len(present_r) / len(r_types) if r_types else 0.0

    num_entities = _count_rows(conn, "entities")
    avg_props = _avg_property_count(conn) if num_entities > 0 else 0.0
    orphans = _count_orphan_entities(conn)

    missing_e = [t for t in e_types if t not in present_e]
    missing_r = [r for r in r_types if r not in present_r]

    return CompletenessMetrics(
        entity_type_coverage=round(e_coverage, 4),
        relation_type_coverage=round(r_coverage, 4),
        avg_properties_per_entity=round(avg_props, 4),
        orphan_entities=orphans,
        missing_entity_types=missing_e,
        missing_relation_types=missing_r,
    )


# ---------------------------------------------------------------------------
# Layer 2.5: Timeliness
# ---------------------------------------------------------------------------


def compute_timeliness_metrics(
    conn: sqlite3.Connection,
    staleness_days: float = 365.0,
    now: datetime | None = None,
) -> TimelinessMetrics:
    """Compute temporal freshness of triples.

    Args:
        conn: SQLite connection with KG schema.
        staleness_days: Threshold in days for a triple to be "stale".
        now: Reference time (default: UTC now).

    Returns:
        TimelinessMetrics.
    """
    conn.row_factory = sqlite3.Row
    ref = now or datetime.now(UTC)

    rows = conn.execute("SELECT created_at FROM triples").fetchall()
    if not rows:
        return TimelinessMetrics()

    ages: list[float] = []
    for row in rows:
        ts = row["created_at"]
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            age = (ref - dt).total_seconds() / 86400.0
            ages.append(max(0.0, age))
        except (ValueError, TypeError):
            continue

    if not ages:
        return TimelinessMetrics()

    avg_age = sum(ages) / len(ages)
    stale = sum(1 for a in ages if a > staleness_days)
    staleness_ratio = stale / len(ages)

    return TimelinessMetrics(
        avg_age_days=round(avg_age, 2),
        staleness_ratio=round(staleness_ratio, 4),
        newest_triple_age_days=round(min(ages), 2),
        oldest_triple_age_days=round(max(ages), 2),
    )


# ---------------------------------------------------------------------------
# Layer 2.5: Redundancy
# ---------------------------------------------------------------------------


def compute_redundancy_metrics(
    conn: sqlite3.Connection,
    similarity_threshold: float = 0.85,
) -> RedundancyMetrics:
    """Detect duplicate triples and near-duplicate entities.

    Args:
        conn: SQLite connection with KG schema.
        similarity_threshold: Jaccard threshold for near-duplicate names.

    Returns:
        RedundancyMetrics.
    """
    conn.row_factory = sqlite3.Row

    dup_triples = _count_duplicate_triples(conn)
    near_dup_entities = _count_near_duplicate_entities(conn, similarity_threshold)

    num_triples = _count_rows(conn, "triples")
    num_entities = _count_rows(conn, "entities")
    total = num_triples + num_entities
    redundant = dup_triples + near_dup_entities
    ratio = redundant / total if total > 0 else 0.0

    return RedundancyMetrics(
        exact_duplicate_triples=dup_triples,
        near_duplicate_entities=near_dup_entities,
        redundancy_ratio=round(ratio, 4),
    )


# ---------------------------------------------------------------------------
# TWCS Sampling (Layer 3)
# ---------------------------------------------------------------------------


def sample_triples_twcs(
    conn: sqlite3.Connection,
    sample_size: int = 100,
    seed: int | None = None,
) -> list[dict]:
    """Type-Weighted Cluster Sampling for scalable KG evaluation.

    Samples triples proportional to relation type distribution, ensuring
    every relation type is represented. Uses reservoir sampling within
    each type cluster.

    Args:
        conn: SQLite connection with KG schema.
        sample_size: Target total sample size.
        seed: Random seed for reproducibility.

    Returns:
        List of triple dicts from the sample.
    """
    import random

    conn.row_factory = sqlite3.Row
    rng = random.Random(seed)  # nosec B311 — not for crypto

    rel_dist = _distribution(conn, "triples", "relation")
    total = sum(rel_dist.values())

    if total == 0:
        return []

    allocations: dict[str, int] = {}
    for rel_type, count in rel_dist.items():
        alloc = max(1, round(sample_size * count / total))
        allocations[rel_type] = min(alloc, count)

    while sum(allocations.values()) > sample_size:
        largest = max(allocations, key=lambda k: allocations[k])
        allocations[largest] -= 1

    sampled: list[dict] = []
    for rel_type, n in allocations.items():
        if n <= 0:
            continue
        rows = conn.execute(
            "SELECT subject_id, relation, object_id,"
            " provenance_paper, provenance_run, confidence, created_at"
            " FROM triples WHERE relation = ?",
            (rel_type,),
        ).fetchall()
        chosen = rng.sample(rows, min(n, len(rows)))
        for row in chosen:
            sampled.append(dict(row))

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def evaluate_kg_quality(
    conn: sqlite3.Connection,
    weights: dict[str, float] | None = None,
    staleness_days: float = 365.0,
    now: datetime | None = None,
) -> KGQualityScore:
    """Evaluate overall KG quality across all 5 dimensions.

    Args:
        conn: SQLite connection with KG schema.
        weights: Dimension weights (accuracy, consistency, completeness,
            timeliness, redundancy). Default: equal weights.
        staleness_days: Threshold for timeliness staleness.
        now: Reference time for timeliness computation.

    Returns:
        KGQualityScore with per-dimension and composite scores.
    """
    w = weights or {
        "accuracy": 0.25,
        "consistency": 0.25,
        "completeness": 0.20,
        "timeliness": 0.15,
        "redundancy": 0.15,
    }
    total_w = sum(w.values())
    if total_w <= 0:
        total_w = 1.0

    struct = compute_structural_metrics(conn)
    consist = compute_consistency_metrics(conn)
    complete = compute_completeness_metrics(conn)
    timely = compute_timeliness_metrics(conn, staleness_days=staleness_days, now=now)
    redund = compute_redundancy_metrics(conn)

    accuracy_score = _accuracy_from_structural(struct, consist)
    consistency_score = (consist.ic_score + consist.ec_score) / 2.0
    completeness_score = (
        complete.entity_type_coverage + complete.relation_type_coverage
    ) / 2.0
    timeliness_score = max(0.0, 1.0 - timely.staleness_ratio)
    redundancy_score = max(0.0, 1.0 - redund.redundancy_ratio)

    composite = (
        w.get("accuracy", 0.0) * accuracy_score
        + w.get("consistency", 0.0) * consistency_score
        + w.get("completeness", 0.0) * completeness_score
        + w.get("timeliness", 0.0) * timeliness_score
        + w.get("redundancy", 0.0) * redundancy_score
    ) / total_w

    return KGQualityScore(
        accuracy=round(accuracy_score, 4),
        consistency=round(consistency_score, 4),
        completeness=round(completeness_score, 4),
        timeliness=round(timeliness_score, 4),
        redundancy=round(redundancy_score, 4),
        composite=round(composite, 4),
        structural=struct,
        consistency_detail=consist,
        completeness_detail=complete,
        timeliness_detail=timely,
        redundancy_detail=redund,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    """Count rows in a table (safe — table name is always a literal)."""
    valid = {"entities", "triples", "case_adaptations", "cases"}
    if table not in valid:
        raise ValueError(f"Invalid table: {table}")
    sql = f"SELECT COUNT(*) FROM {table}"  # noqa: S608  # nosec B608
    return conn.execute(sql).fetchone()[0]


def _count_connected_entities(conn: sqlite3.Connection) -> int:
    """Count entities that participate in at least one triple."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT x) FROM ("
        " SELECT subject_id AS x FROM triples"
        " UNION"
        " SELECT object_id AS x FROM triples"
        ")"
    ).fetchone()
    return row[0] if row else 0


def _build_adjacency(
    conn: sqlite3.Connection,
) -> dict[str, set[str]]:
    """Build undirected adjacency map from triples."""
    adj: dict[str, set[str]] = defaultdict(set)
    rows = conn.execute("SELECT subject_id, object_id FROM triples").fetchall()
    for row in rows:
        s, o = row["subject_id"], row["object_id"]
        adj[s].add(o)
        adj[o].add(s)
    return adj


def _all_entity_ids(conn: sqlite3.Connection) -> set[str]:
    """Get all entity IDs."""
    rows = conn.execute("SELECT entity_id FROM entities").fetchall()
    return {r["entity_id"] for r in rows}


def _find_components(adj: dict[str, set[str]], all_ids: set[str]) -> list[set[str]]:
    """Find weakly-connected components via BFS."""
    visited: set[str] = set()
    components: list[set[str]] = []

    for node in all_ids:
        if node in visited:
            continue
        component: set[str] = set()
        queue = [node]
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in adj.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        if component:
            components.append(component)

    return components


def _distribution(conn: sqlite3.Connection, table: str, column: str) -> dict[str, int]:
    """Get value distribution for a column."""
    valid_tables = {"entities", "triples"}
    valid_columns = {"entity_type", "relation"}
    if table not in valid_tables or column not in valid_columns:
        raise ValueError(f"Invalid table/column: {table}.{column}")
    sql = f"SELECT {column}, COUNT(*) as cnt FROM {table} GROUP BY {column}"  # noqa: S608  # nosec B608
    rows = conn.execute(sql).fetchall()
    return {r[0]: r[1] for r in rows}


def _count_contradictions(conn: sqlite3.Connection) -> int:
    """Detect contradictions: same (s, o) with supports + contradicts."""
    rows = conn.execute(
        "SELECT COUNT(*) FROM triples t1"
        " INNER JOIN triples t2"
        " ON t1.subject_id = t2.subject_id"
        " AND t1.object_id = t2.object_id"
        " AND t1.relation = 'supports_claim'"
        " AND t2.relation = 'contradicts_claim'"
    ).fetchone()
    return rows[0] if rows else 0


def _count_duplicate_triples(conn: sqlite3.Connection) -> int:
    """Count exact-duplicate (subject, relation, object) triples."""
    rows = conn.execute(
        "SELECT SUM(cnt - 1) FROM ("
        " SELECT COUNT(*) as cnt"
        " FROM triples"
        " GROUP BY subject_id, relation, object_id"
        " HAVING cnt > 1"
        ")"
    ).fetchone()
    return rows[0] or 0


def _avg_property_count(conn: sqlite3.Connection) -> float:
    """Average number of properties per entity."""
    rows = conn.execute("SELECT properties FROM entities").fetchall()
    if not rows:
        return 0.0
    total = 0
    for row in rows:
        props = row["properties"]
        if props:
            try:
                d = (
                    props
                    if isinstance(props, dict)
                    else __import__("json").loads(props)
                )
                total += len(d)
            except (ValueError, TypeError):
                pass
    return total / len(rows)


def _count_orphan_entities(conn: sqlite3.Connection) -> int:
    """Count entities that appear in zero triples."""
    num_entities = _count_rows(conn, "entities")
    connected = _count_connected_entities(conn)
    return max(0, num_entities - connected)


def _name_tokens(name: str) -> set[str]:
    """Tokenize a name into lowercase word tokens."""
    return set(name.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _count_near_duplicate_entities(
    conn: sqlite3.Connection, threshold: float = 0.85
) -> int:
    """Count near-duplicate entity pairs by name Jaccard similarity.

    Groups by entity_type to avoid cross-type false positives.
    Limited to first 500 entities per type for scalability.
    """
    rows = conn.execute("SELECT entity_id, entity_type, name FROM entities").fetchall()

    by_type: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for row in rows:
        tokens = _name_tokens(row["name"])
        by_type[row["entity_type"]].append((row["entity_id"], tokens))

    count = 0
    for _etype, entities in by_type.items():
        capped = entities[:500]
        for i in range(len(capped)):
            for j in range(i + 1, len(capped)):
                if _jaccard(capped[i][1], capped[j][1]) >= threshold:
                    count += 1

    return count


def _accuracy_from_structural(
    struct: StructuralMetrics, consist: ConsistencyMetrics
) -> float:
    """Derive an accuracy proxy from structural + consistency data.

    Combines ICR (connectivity), absence of contradictions, and absence
    of duplicates into a [0, 1] score.
    """
    icr_score = struct.icr
    no_contradiction = consist.ic_score
    dup_penalty = min(
        1.0,
        consist.duplicate_triples / max(1, struct.num_triples),
    )
    dup_score = max(0.0, 1.0 - dup_penalty)

    return icr_score * 0.4 + no_contradiction * 0.4 + dup_score * 0.2
