"""Multi-session coherence evaluation.

Evaluates knowledge coherence across multiple pipeline runs on the same
or related topics.  Implements a subset of the seven-dimension framework
from the deep research literature (A-MBER, TiMem, AMA, MemoryOS):

1. Factual Consistency — findings that appear in multiple runs agree
2. Temporal Ordering — knowledge evolution is chronologically sound
3. Knowledge Update Fidelity — superseded findings are properly replaced
4. Contradiction Detection — conflicting claims are surfaced
5. Coherence Score — composite metric across dimensions

Reference: FD#9 Multi-Session Coherence Evaluation (10 papers).
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single finding extracted from a run's synthesis."""

    text: str
    confidence: str = "medium"  # high / medium / low
    run_id: str = ""
    timestamp: str = ""
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class Contradiction:
    """A pair of contradictory findings across runs."""

    finding_a: Finding
    finding_b: Finding
    similarity: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "finding_a": self.finding_a.to_dict(),
            "finding_b": self.finding_b.to_dict(),
            "similarity": self.similarity,
            "explanation": self.explanation,
        }


@dataclass
class KnowledgeUpdate:
    """Tracks how a finding evolved across runs."""

    topic: str
    old_finding: Finding | None = None
    new_finding: Finding | None = None
    update_type: str = "new"  # new / revised / superseded / retracted

    def to_dict(self) -> dict[str, object]:
        return {
            "topic": self.topic,
            "old_finding": self.old_finding.to_dict() if self.old_finding else None,
            "new_finding": self.new_finding.to_dict() if self.new_finding else None,
            "update_type": self.update_type,
        }


@dataclass
class CoherenceScore:
    """Composite coherence evaluation result."""

    factual_consistency: float = 0.0  # 0–1
    temporal_ordering: float = 0.0  # 0–1
    knowledge_update_fidelity: float = 0.0  # 0–1
    contradiction_rate: float = 0.0  # 0–1 (lower is better)
    overall: float = 0.0  # weighted composite

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class CoherenceReport:
    """Full coherence evaluation report across runs."""

    run_ids: list[str]
    score: CoherenceScore
    contradictions: list[Contradiction] = field(default_factory=list)
    knowledge_updates: list[KnowledgeUpdate] = field(default_factory=list)
    finding_count: int = 0
    common_finding_count: int = 0
    topic_overlap: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "run_ids": self.run_ids,
            "score": self.score.to_dict(),
            "contradictions": [c.to_dict() for c in self.contradictions],
            "knowledge_updates": [u.to_dict() for u in self.knowledge_updates],
            "finding_count": self.finding_count,
            "common_finding_count": self.common_finding_count,
            "topic_overlap": self.topic_overlap,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip, collapse spaces)."""
    return " ".join(text.lower().split())


def _word_set(text: str) -> set[str]:
    """Extract a set of content words from text."""
    stop_words = {
        "the",
        "a",
        "an",
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
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "from",
        "by",
        "about",
        "as",
        "into",
        "through",
        "during",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "us",
        "he",
        "she",
        "him",
        "her",
        "his",
    }
    words = set(_normalize_text(text).split())
    return words - stop_words


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard similarity between two texts.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Jaccard similarity in [0.0, 1.0].
    """
    words_a = _word_set(text_a)
    words_b = _word_set(text_b)
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_findings(
    synthesis: dict[str, object],
    run_id: str = "",
    timestamp: str = "",
) -> list[Finding]:
    """Extract findings from a synthesis_results.json.

    Args:
        synthesis: Parsed synthesis results dict.
        run_id: Run ID for provenance.
        timestamp: Timestamp string for temporal ordering.

    Returns:
        List of Finding objects.
    """
    findings: list[Finding] = []

    # Extract from confidence-graded findings
    graded = synthesis.get("confidence_graded_findings", {})
    if isinstance(graded, dict):
        for level in ("high", "medium", "low"):
            for item in graded.get(level, []):
                text = ""
                evidence: list[str] = []
                if isinstance(item, dict):
                    text = str(item.get("finding", item.get("text", "")))
                    evidence = list(item.get("evidence_ids", []))
                elif isinstance(item, str):
                    text = item
                if text:
                    findings.append(
                        Finding(
                            text=text,
                            confidence=level,
                            run_id=run_id,
                            timestamp=timestamp,
                            evidence_ids=evidence,
                        )
                    )

    # Extract from themes
    themes = synthesis.get("themes", [])
    if isinstance(themes, list):
        for theme in themes:
            if isinstance(theme, dict):
                text = str(theme.get("description", theme.get("theme", "")))
                if text:
                    findings.append(
                        Finding(
                            text=text,
                            confidence="medium",
                            run_id=run_id,
                            timestamp=timestamp,
                        )
                    )

    return findings


def load_synthesis(run_root: Path) -> dict[str, object] | None:
    """Load synthesis results from a run directory.

    Args:
        run_root: Root directory of a pipeline run.

    Returns:
        Parsed synthesis dict or None.
    """
    for candidate in [
        run_root / "synthesis" / "synthesis_results.json",
        run_root / "summarize" / "synthesis_results.json",
    ]:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load synthesis from %s: %s", candidate, exc)
    return None


def load_run_timestamp(run_root: Path) -> str:
    """Extract timestamp from run manifest or directory name.

    Args:
        run_root: Root directory of a pipeline run.

    Returns:
        ISO timestamp string.
    """
    manifest = run_root / "run_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            ts = data.get("started_at", data.get("created_at", ""))
            if ts:
                return str(ts)
        except (json.JSONDecodeError, OSError):
            pass
    # Fall back to directory mtime
    try:
        mtime = run_root.stat().st_mtime
        return datetime.fromtimestamp(mtime).isoformat()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Coherence dimensions
# ---------------------------------------------------------------------------

# Similarity threshold for considering two findings as "about the same topic"
TOPIC_SIMILARITY_THRESHOLD = 0.35

# Threshold above which similar findings with opposing sentiment are contradictions
CONTRADICTION_SIMILARITY_THRESHOLD = 0.40

# Negation indicators
NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "cannot",
    "unable",
    "fail",
    "failed",
    "fails",
    "lack",
    "lacking",
    "lacks",
    "without",
    "insufficient",
    "inadequate",
    "ineffective",
    "worse",
    "decline",
    "declined",
    "inferior",
    "outperformed",
}


def _has_negation_disagreement(text_a: str, text_b: str) -> bool:
    """Check if two similar texts disagree through negation patterns.

    A simple heuristic: if one text contains negation words that the
    other does not (asymmetric negation), they may contradict.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        True if negation disagreement detected.
    """
    words_a = set(_normalize_text(text_a).split())
    words_b = set(_normalize_text(text_b).split())
    neg_a = words_a & NEGATION_WORDS
    neg_b = words_b & NEGATION_WORDS
    # Asymmetric negation: one has negation, the other doesn't
    if neg_a and not neg_b:
        return True
    return bool(neg_b and not neg_a)


def detect_contradictions(
    findings_by_run: dict[str, list[Finding]],
) -> list[Contradiction]:
    """Detect contradictory findings across runs.

    Two findings contradict if they are topically similar but contain
    asymmetric negation patterns.

    Args:
        findings_by_run: Dict mapping run_id to list of findings.

    Returns:
        List of detected contradictions.
    """
    contradictions: list[Contradiction] = []
    run_ids = sorted(findings_by_run.keys())

    for i, rid_a in enumerate(run_ids):
        for rid_b in run_ids[i + 1 :]:
            for fa in findings_by_run[rid_a]:
                for fb in findings_by_run[rid_b]:
                    sim = jaccard_similarity(fa.text, fb.text)
                    if sim < CONTRADICTION_SIMILARITY_THRESHOLD:
                        continue
                    if _has_negation_disagreement(fa.text, fb.text):
                        contradictions.append(
                            Contradiction(
                                finding_a=fa,
                                finding_b=fb,
                                similarity=sim,
                                explanation=(
                                    "Negation disagreement detected between "
                                    f"run {rid_a} and run {rid_b}"
                                ),
                            )
                        )

    return contradictions


def track_knowledge_updates(
    findings_by_run: dict[str, list[Finding]],
    run_order: list[str],
) -> list[KnowledgeUpdate]:
    """Track how findings evolve across runs in chronological order.

    For each pair of consecutive runs, identify new, revised,
    superseded, and retracted findings.

    Args:
        findings_by_run: Dict mapping run_id to list of findings.
        run_order: Run IDs in chronological order.

    Returns:
        List of knowledge updates.
    """
    updates: list[KnowledgeUpdate] = []

    for idx in range(len(run_order) - 1):
        rid_old = run_order[idx]
        rid_new = run_order[idx + 1]
        old_findings = findings_by_run.get(rid_old, [])
        new_findings = findings_by_run.get(rid_new, [])

        matched_old: set[int] = set()
        matched_new: set[int] = set()

        # Match findings between consecutive runs
        for ni, fn in enumerate(new_findings):
            best_sim = 0.0
            best_oi = -1
            for oi, fo in enumerate(old_findings):
                if oi in matched_old:
                    continue
                sim = jaccard_similarity(fn.text, fo.text)
                if sim > best_sim:
                    best_sim = sim
                    best_oi = oi

            if best_sim >= TOPIC_SIMILARITY_THRESHOLD and best_oi >= 0:
                matched_old.add(best_oi)
                matched_new.add(ni)
                # Check if finding was revised (similar but not identical)
                if best_sim < 0.95:
                    updates.append(
                        KnowledgeUpdate(
                            topic=fn.text[:100],
                            old_finding=old_findings[best_oi],
                            new_finding=fn,
                            update_type="revised",
                        )
                    )
                # else: identical/near-identical — no update needed

        # New findings (not matched to any old finding)
        for ni, fn in enumerate(new_findings):
            if ni not in matched_new:
                updates.append(
                    KnowledgeUpdate(
                        topic=fn.text[:100],
                        new_finding=fn,
                        update_type="new",
                    )
                )

        # Retracted/superseded findings (old findings with no match)
        for oi, fo in enumerate(old_findings):
            if oi not in matched_old:
                updates.append(
                    KnowledgeUpdate(
                        topic=fo.text[:100],
                        old_finding=fo,
                        update_type="retracted",
                    )
                )

    return updates


def compute_factual_consistency(
    findings_by_run: dict[str, list[Finding]],
) -> float:
    """Compute factual consistency score across runs.

    Measures how often findings that appear in multiple runs are
    consistent (not contradictory). Score of 1.0 means no contradictions.

    Args:
        findings_by_run: Dict mapping run_id to list of findings.

    Returns:
        Consistency score in [0.0, 1.0].
    """
    if len(findings_by_run) < 2:
        return 1.0

    total_comparisons = 0
    consistent_count = 0

    run_ids = sorted(findings_by_run.keys())
    for i, rid_a in enumerate(run_ids):
        for rid_b in run_ids[i + 1 :]:
            for fa in findings_by_run[rid_a]:
                for fb in findings_by_run[rid_b]:
                    sim = jaccard_similarity(fa.text, fb.text)
                    if sim >= TOPIC_SIMILARITY_THRESHOLD:
                        total_comparisons += 1
                        if not _has_negation_disagreement(fa.text, fb.text):
                            consistent_count += 1

    if total_comparisons == 0:
        return 1.0
    return consistent_count / total_comparisons


def compute_temporal_ordering(
    findings_by_run: dict[str, list[Finding]],
    run_order: list[str],
) -> float:
    """Evaluate temporal ordering quality.

    Checks that knowledge updates follow chronological order and newer
    runs don't regress to older, superseded findings.

    Args:
        findings_by_run: Dict mapping run_id to list of findings.
        run_order: Run IDs in chronological order.

    Returns:
        Temporal ordering score in [0.0, 1.0].
    """
    if len(run_order) < 2:
        return 1.0

    updates = track_knowledge_updates(findings_by_run, run_order)
    if not updates:
        return 1.0

    # Score: ratio of clean updates (new/revised) vs problematic (retracted
    # can be fine, but high retraction rate suggests instability)
    total = len(updates)
    clean = sum(1 for u in updates if u.update_type in ("new", "revised"))
    return clean / total if total > 0 else 1.0


def compute_knowledge_update_fidelity(
    findings_by_run: dict[str, list[Finding]],
    run_order: list[str],
) -> float:
    """Measure how well knowledge updates are tracked.

    A high score means findings evolve cleanly: revisions maintain
    provenance, and superseded findings are properly replaced.

    Args:
        findings_by_run: Dict mapping run_id to list of findings.
        run_order: Run IDs in chronological order.

    Returns:
        Update fidelity score in [0.0, 1.0].
    """
    if len(run_order) < 2:
        return 1.0

    updates = track_knowledge_updates(findings_by_run, run_order)
    if not updates:
        return 1.0

    # Revised findings that still share evidence with their predecessor
    # indicate good provenance. New findings are good. Retracted without
    # replacement indicates potential information loss.
    scored = 0
    good: float = 0
    for u in updates:
        scored += 1
        if u.update_type == "new":
            good += 1
        elif u.update_type == "revised":
            # Revised is good — knowledge is evolving
            good += 1
        elif u.update_type == "retracted":
            # Retracted without replacement is neutral (might be correct)
            good += 0.5

    return good / scored if scored > 0 else 1.0


def compute_topic_overlap(
    findings_by_run: dict[str, list[Finding]],
) -> float:
    """Compute topic overlap ratio across runs.

    Measures how many findings in the latest run have a corresponding
    topic-match in at least one other run.

    Args:
        findings_by_run: Dict mapping run_id to list of findings.

    Returns:
        Overlap ratio in [0.0, 1.0].
    """
    if len(findings_by_run) < 2:
        return 0.0

    run_ids = sorted(findings_by_run.keys())
    latest = run_ids[-1]
    other_runs = run_ids[:-1]

    latest_findings = findings_by_run[latest]
    if not latest_findings:
        return 0.0

    matched = 0
    for fn in latest_findings:
        for rid in other_runs:
            found = False
            for fo in findings_by_run[rid]:
                if jaccard_similarity(fn.text, fo.text) >= TOPIC_SIMILARITY_THRESHOLD:
                    found = True
                    break
            if found:
                matched += 1
                break

    return matched / len(latest_findings)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def evaluate_coherence(
    run_roots: list[Path],
    run_ids: list[str],
    weights: dict[str, float] | None = None,
) -> CoherenceReport:
    """Evaluate multi-session coherence across pipeline runs.

    Loads synthesis results from each run and computes coherence
    dimensions.

    Args:
        run_roots: List of run root directories.
        run_ids: Corresponding run IDs.
        weights: Optional weights for composite score.
            Keys: factual_consistency, temporal_ordering,
            knowledge_update_fidelity, contradiction_rate.
            Default: equal weights.

    Returns:
        CoherenceReport with scores and details.
    """
    if len(run_roots) != len(run_ids):
        raise ValueError("run_roots and run_ids must have the same length")

    if not weights:
        weights = {
            "factual_consistency": 0.30,
            "temporal_ordering": 0.25,
            "knowledge_update_fidelity": 0.25,
            "contradiction_rate": 0.20,
        }

    # Load findings from each run
    findings_by_run: dict[str, list[Finding]] = {}
    timestamps: dict[str, str] = {}

    for root, rid in zip(run_roots, run_ids, strict=False):
        ts = load_run_timestamp(root)
        timestamps[rid] = ts
        synthesis = load_synthesis(root)
        if synthesis:
            findings_by_run[rid] = extract_findings(synthesis, rid, ts)
        else:
            findings_by_run[rid] = []
            logger.warning("No synthesis found for run %s at %s", rid, root)

    # Sort runs chronologically
    run_order = sorted(
        run_ids,
        key=lambda r: timestamps.get(r, ""),
    )

    # Compute dimensions
    factual = compute_factual_consistency(findings_by_run)
    temporal = compute_temporal_ordering(findings_by_run, run_order)
    update_fidelity = compute_knowledge_update_fidelity(findings_by_run, run_order)

    contradictions = detect_contradictions(findings_by_run)
    total_findings = sum(len(f) for f in findings_by_run.values())
    contradiction_rate = (
        len(contradictions) / total_findings if total_findings > 0 else 0.0
    )

    knowledge_updates = track_knowledge_updates(findings_by_run, run_order)
    topic_overlap = compute_topic_overlap(findings_by_run)

    # Composite score (higher is better for all except contradiction_rate)
    w = weights
    overall = (
        w.get("factual_consistency", 0.30) * factual
        + w.get("temporal_ordering", 0.25) * temporal
        + w.get("knowledge_update_fidelity", 0.25) * update_fidelity
        + w.get("contradiction_rate", 0.20) * (1.0 - min(contradiction_rate, 1.0))
    )

    score = CoherenceScore(
        factual_consistency=round(factual, 4),
        temporal_ordering=round(temporal, 4),
        knowledge_update_fidelity=round(update_fidelity, 4),
        contradiction_rate=round(contradiction_rate, 4),
        overall=round(overall, 4),
    )

    # Count common findings
    common_count = 0
    if len(run_ids) >= 2:
        texts_per_run = {
            rid: {_normalize_text(f.text) for f in findings}
            for rid, findings in findings_by_run.items()
        }
        all_texts = set()
        for texts in texts_per_run.values():
            all_texts.update(texts)
        for text in all_texts:
            in_count = sum(1 for texts in texts_per_run.values() if text in texts)
            if in_count >= 2:
                common_count += 1

    report = CoherenceReport(
        run_ids=run_order,
        score=score,
        contradictions=contradictions,
        knowledge_updates=knowledge_updates,
        finding_count=total_findings,
        common_finding_count=common_count,
        topic_overlap=round(topic_overlap, 4),
    )

    logger.info(
        "Coherence evaluation: overall=%.2f, factual=%.2f, temporal=%.2f, "
        "updates=%.2f, contradictions=%d (rate=%.3f)",
        score.overall,
        score.factual_consistency,
        score.temporal_ordering,
        score.knowledge_update_fidelity,
        len(contradictions),
        score.contradiction_rate,
    )

    return report


def run_coherence(
    run_ids: list[str],
    workspace: Path,
    output: Path | None = None,
    weights: dict[str, float] | None = None,
) -> CoherenceReport:
    """Evaluate coherence across specified runs and write report.

    Args:
        run_ids: List of run IDs to evaluate.
        workspace: Base workspace directory.
        output: Output path for report JSON. Default: auto-generated.
        weights: Optional dimension weights.

    Returns:
        CoherenceReport.
    """
    if len(run_ids) < 2:
        raise ValueError("At least 2 run IDs required for coherence evaluation")

    run_roots = [workspace / rid for rid in run_ids]

    # Verify run directories exist
    for root, rid in zip(run_roots, run_ids, strict=False):
        if not root.exists():
            raise FileNotFoundError(f"Run directory not found: {root} (run {rid})")

    report = evaluate_coherence(run_roots, run_ids, weights)

    out_path = output or (workspace / f"coherence_{'_'.join(run_ids[:3])}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    logger.info("Coherence report written to %s", out_path)

    return report
