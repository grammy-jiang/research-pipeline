"""Scientific KG benchmark framework.

Gold-standard knowledge-graph benchmarking inspired by:
- Text2KGBench (arXiv 2308.02357): precision/recall/F1 on entity & relation extraction
- ORKG (Open Research Knowledge Graph): structured scholarly assertions
- PubMed Knowledge Base: biomedical entity linking

Provides:
- GoldTriple / GoldDataset: schema for gold-standard KG triples
- ExtractionResult: container for system-extracted triples
- BenchmarkMetrics: precision, recall, F1, hallucination & missing rates
- KGBenchmarkRunner: compare extracted KG against gold standard
- NormalizationStrategy: configurable entity normalization (exact / lowercase / lemma / alias)
- PredefinedSeeds: small built-in seed datasets for smoke testing
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization strategies
# ---------------------------------------------------------------------------


class NormalizationStrategy(Enum):
    """How to normalize entity/relation strings before comparison."""

    EXACT = "exact"
    LOWERCASE = "lowercase"
    LEMMA_LIKE = "lemma_like"
    ALIAS = "alias"


def _normalize_exact(text: str) -> str:
    return text.strip()


def _normalize_lowercase(text: str) -> str:
    return text.strip().lower()


def _normalize_lemma_like(text: str) -> str:
    """Cheap pseudo-lemmatisation: lowercase, collapse whitespace, strip trailing 's'."""
    t = re.sub(r"\s+", " ", text.strip().lower())
    if t.endswith("s") and len(t) > 3:
        t = t[:-1]
    return t


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldTriple:
    """A single (subject, predicate, object) from the gold-standard KG."""

    subject: str
    predicate: str
    obj: str
    source_paper_id: str = ""
    confidence: float = 1.0

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.obj)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "source_paper_id": self.source_paper_id,
            "confidence": self.confidence,
        }


@dataclass
class GoldDataset:
    """A named collection of gold-standard triples with metadata."""

    name: str
    domain: str
    triples: list[GoldTriple] = field(default_factory=list)
    entity_aliases: dict[str, list[str]] = field(default_factory=dict)
    description: str = ""

    @property
    def num_triples(self) -> int:
        return len(self.triples)

    @property
    def entities(self) -> set[str]:
        ents: set[str] = set()
        for t in self.triples:
            ents.add(t.subject)
            ents.add(t.obj)
        return ents

    @property
    def predicates(self) -> set[str]:
        return {t.predicate for t in self.triples}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "num_triples": self.num_triples,
            "num_entities": len(self.entities),
            "num_predicates": len(self.predicates),
            "triples": [t.to_dict() for t in self.triples],
        }


@dataclass(frozen=True)
class ExtractedTriple:
    """A triple extracted by the pipeline under evaluation."""

    subject: str
    predicate: str
    obj: str
    extraction_confidence: float = 0.0
    source_span: str = ""

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.obj)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "extraction_confidence": self.extraction_confidence,
            "source_span": self.source_span,
        }


@dataclass
class ExtractionResult:
    """Collection of triples extracted by a system from a set of papers."""

    system_name: str
    triples: list[ExtractedTriple] = field(default_factory=list)

    @property
    def num_triples(self) -> int:
        return len(self.triples)


# ---------------------------------------------------------------------------
# Benchmark metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Evaluation metrics comparing extracted KG against gold standard."""

    precision: float
    recall: float
    f1: float
    hallucination_rate: float
    missing_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_gold: int
    total_extracted: int

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "missing_rate": round(self.missing_rate, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_gold": self.total_gold,
            "total_extracted": self.total_extracted,
        }


@dataclass(frozen=True)
class EntityMetrics:
    """Entity-level evaluation metrics."""

    precision: float
    recall: float
    f1: float
    gold_entities: int
    extracted_entities: int
    matched: int

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "gold_entities": self.gold_entities,
            "extracted_entities": self.extracted_entities,
            "matched": self.matched,
        }


@dataclass(frozen=True)
class PredicateMetrics:
    """Predicate/relation-level evaluation metrics."""

    precision: float
    recall: float
    f1: float
    gold_predicates: int
    extracted_predicates: int
    matched: int

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "gold_predicates": self.gold_predicates,
            "extracted_predicates": self.extracted_predicates,
            "matched": self.matched,
        }


@dataclass
class FullBenchmarkReport:
    """Complete benchmark evaluation report."""

    dataset_name: str
    system_name: str
    triple_metrics: BenchmarkMetrics
    entity_metrics: EntityMetrics
    predicate_metrics: PredicateMetrics
    per_predicate_f1: dict[str, float] = field(default_factory=dict)
    hallucinated_triples: list[dict] = field(default_factory=list)
    missing_triples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "system_name": self.system_name,
            "triple_metrics": self.triple_metrics.to_dict(),
            "entity_metrics": self.entity_metrics.to_dict(),
            "predicate_metrics": self.predicate_metrics.to_dict(),
            "per_predicate_f1": {
                k: round(v, 4) for k, v in self.per_predicate_f1.items()
            },
            "num_hallucinated": len(self.hallucinated_triples),
            "num_missing": len(self.missing_triples),
        }


# ---------------------------------------------------------------------------
# KG Benchmark Runner
# ---------------------------------------------------------------------------


class KGBenchmarkRunner:
    """Compare extracted KG against gold-standard datasets.

    Parameters
    ----------
    strategy:
        Entity/relation normalization strategy.
    alias_map:
        Optional mapping from canonical entity → list of known aliases.
        Only used when ``strategy`` is ``ALIAS``.
    """

    def __init__(
        self,
        strategy: NormalizationStrategy = NormalizationStrategy.LOWERCASE,
        alias_map: dict[str, list[str]] | None = None,
    ) -> None:
        self._strategy = strategy
        self._alias_map = alias_map or {}
        self._normalizer = self._build_normalizer()
        self._alias_lookup: dict[str, str] = {}
        if self._strategy == NormalizationStrategy.ALIAS:
            self._build_alias_lookup()

    def _build_normalizer(self) -> Callable[[str], str]:
        if self._strategy == NormalizationStrategy.EXACT:
            return _normalize_exact
        if self._strategy == NormalizationStrategy.LOWERCASE:
            return _normalize_lowercase
        if self._strategy == NormalizationStrategy.LEMMA_LIKE:
            return _normalize_lemma_like
        # ALIAS uses lowercase as base
        return _normalize_lowercase

    def _build_alias_lookup(self) -> None:
        """Build reverse map: alias → canonical (all lowercased)."""
        for canonical, aliases in self._alias_map.items():
            key = canonical.lower()
            self._alias_lookup[key] = key
            for alias in aliases:
                self._alias_lookup[alias.lower()] = key

    def _normalize(self, text: str) -> str:
        base = self._normalizer(text)
        if self._strategy == NormalizationStrategy.ALIAS:
            return self._alias_lookup.get(base, base)
        return base

    def _normalize_triple(
        self, s: str, p: str, o: str
    ) -> tuple[str, str, str]:
        return (self._normalize(s), self._normalize(p), self._normalize(o))

    def evaluate_triples(
        self,
        gold: GoldDataset,
        extracted: ExtractionResult,
    ) -> BenchmarkMetrics:
        """Compute triple-level precision, recall, F1."""
        gold_set: set[tuple[str, str, str]] = set()
        for t in gold.triples:
            gold_set.add(self._normalize_triple(t.subject, t.predicate, t.obj))

        extracted_set: set[tuple[str, str, str]] = set()
        for t in extracted.triples:
            extracted_set.add(
                self._normalize_triple(t.subject, t.predicate, t.obj)
            )

        tp = len(gold_set & extracted_set)
        fp = len(extracted_set - gold_set)
        fn = len(gold_set - extracted_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        total_extracted = len(extracted_set)
        hallucination_rate = fp / total_extracted if total_extracted > 0 else 0.0
        total_gold = len(gold_set)
        missing_rate = fn / total_gold if total_gold > 0 else 0.0

        return BenchmarkMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            hallucination_rate=hallucination_rate,
            missing_rate=missing_rate,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            total_gold=total_gold,
            total_extracted=total_extracted,
        )

    def evaluate_entities(
        self,
        gold: GoldDataset,
        extracted: ExtractionResult,
    ) -> EntityMetrics:
        """Entity-level evaluation."""
        gold_ents = {self._normalize(e) for e in gold.entities}
        ext_ents: set[str] = set()
        for t in extracted.triples:
            ext_ents.add(self._normalize(t.subject))
            ext_ents.add(self._normalize(t.obj))

        matched = len(gold_ents & ext_ents)
        precision = matched / len(ext_ents) if ext_ents else 0.0
        recall = matched / len(gold_ents) if gold_ents else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return EntityMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            gold_entities=len(gold_ents),
            extracted_entities=len(ext_ents),
            matched=matched,
        )

    def evaluate_predicates(
        self,
        gold: GoldDataset,
        extracted: ExtractionResult,
    ) -> PredicateMetrics:
        """Predicate/relation-level evaluation."""
        gold_preds = {self._normalize(p) for p in gold.predicates}
        ext_preds = {self._normalize(t.predicate) for t in extracted.triples}

        matched = len(gold_preds & ext_preds)
        precision = matched / len(ext_preds) if ext_preds else 0.0
        recall = matched / len(gold_preds) if gold_preds else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return PredicateMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            gold_predicates=len(gold_preds),
            extracted_predicates=len(ext_preds),
            matched=matched,
        )

    def per_predicate_f1(
        self,
        gold: GoldDataset,
        extracted: ExtractionResult,
    ) -> dict[str, float]:
        """F1 score broken down by predicate type."""
        gold_by_pred: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for t in gold.triples:
            p = self._normalize(t.predicate)
            gold_by_pred[p].add((self._normalize(t.subject), self._normalize(t.obj)))

        ext_by_pred: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for t in extracted.triples:
            p = self._normalize(t.predicate)
            ext_by_pred[p].add((self._normalize(t.subject), self._normalize(t.obj)))

        all_preds = set(gold_by_pred.keys()) | set(ext_by_pred.keys())
        result: dict[str, float] = {}
        for pred in sorted(all_preds):
            gset = gold_by_pred.get(pred, set())
            eset = ext_by_pred.get(pred, set())
            tp = len(gset & eset)
            fp = len(eset - gset)
            fn = len(gset - eset)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            )
            result[pred] = f1
        return result

    def find_hallucinated(
        self,
        gold: GoldDataset,
        extracted: ExtractionResult,
    ) -> list[ExtractedTriple]:
        """Return extracted triples not present in gold."""
        gold_set: set[tuple[str, str, str]] = set()
        for t in gold.triples:
            gold_set.add(self._normalize_triple(t.subject, t.predicate, t.obj))

        hallucinated: list[ExtractedTriple] = []
        for t in extracted.triples:
            norm = self._normalize_triple(t.subject, t.predicate, t.obj)
            if norm not in gold_set:
                hallucinated.append(t)
        return hallucinated

    def find_missing(
        self,
        gold: GoldDataset,
        extracted: ExtractionResult,
    ) -> list[GoldTriple]:
        """Return gold triples not present in extraction."""
        ext_set: set[tuple[str, str, str]] = set()
        for t in extracted.triples:
            ext_set.add(self._normalize_triple(t.subject, t.predicate, t.obj))

        missing: list[GoldTriple] = []
        for t in gold.triples:
            norm = self._normalize_triple(t.subject, t.predicate, t.obj)
            if norm not in ext_set:
                missing.append(t)
        return missing

    def full_report(
        self,
        gold: GoldDataset,
        extracted: ExtractionResult,
    ) -> FullBenchmarkReport:
        """Run all evaluation metrics and return a comprehensive report."""
        triple_m = self.evaluate_triples(gold, extracted)
        entity_m = self.evaluate_entities(gold, extracted)
        predicate_m = self.evaluate_predicates(gold, extracted)
        ppf1 = self.per_predicate_f1(gold, extracted)
        hallucinated = self.find_hallucinated(gold, extracted)
        missing = self.find_missing(gold, extracted)

        return FullBenchmarkReport(
            dataset_name=gold.name,
            system_name=extracted.system_name,
            triple_metrics=triple_m,
            entity_metrics=entity_m,
            predicate_metrics=predicate_m,
            per_predicate_f1=ppf1,
            hallucinated_triples=[t.to_dict() for t in hallucinated],
            missing_triples=[t.to_dict() for t in missing],
        )


# ---------------------------------------------------------------------------
# Predefined seed datasets for smoke testing
# ---------------------------------------------------------------------------


class PredefinedSeeds:
    """Small built-in gold-standard datasets for smoke testing."""

    @staticmethod
    def nlp_basics() -> GoldDataset:
        """Minimal NLP domain KG (10 triples)."""
        triples = [
            GoldTriple("BERT", "is_a", "language model", "1810.04805"),
            GoldTriple("BERT", "uses", "transformer", "1810.04805"),
            GoldTriple("GPT-2", "is_a", "language model", "radford2019"),
            GoldTriple("GPT-2", "uses", "transformer", "radford2019"),
            GoldTriple("transformer", "introduced_by", "Vaswani et al.", "1706.03762"),
            GoldTriple("attention", "is_component_of", "transformer", "1706.03762"),
            GoldTriple("BERT", "trained_on", "BookCorpus", "1810.04805"),
            GoldTriple("BERT", "trained_on", "Wikipedia", "1810.04805"),
            GoldTriple("word2vec", "is_a", "word embedding", "mikolov2013"),
            GoldTriple("GloVe", "is_a", "word embedding", "pennington2014"),
        ]
        return GoldDataset(
            name="nlp_basics",
            domain="natural language processing",
            triples=triples,
            entity_aliases={
                "BERT": ["Bidirectional Encoder Representations from Transformers"],
                "GPT-2": ["Generative Pre-trained Transformer 2"],
            },
            description="Minimal NLP domain KG for smoke testing.",
        )

    @staticmethod
    def ml_methods() -> GoldDataset:
        """Minimal ML methods KG (8 triples)."""
        triples = [
            GoldTriple("random forest", "is_a", "ensemble method", "breiman2001"),
            GoldTriple("gradient boosting", "is_a", "ensemble method", "friedman2001"),
            GoldTriple("SVM", "is_a", "classifier", "cortes1995"),
            GoldTriple("SVM", "uses", "kernel trick", "cortes1995"),
            GoldTriple("neural network", "has_component", "layer", "rumelhart1986"),
            GoldTriple("dropout", "regularizes", "neural network", "srivastava2014"),
            GoldTriple("batch normalization", "accelerates", "training", "ioffe2015"),
            GoldTriple("Adam", "is_a", "optimizer", "kingma2015"),
        ]
        return GoldDataset(
            name="ml_methods",
            domain="machine learning",
            triples=triples,
            entity_aliases={
                "SVM": ["Support Vector Machine", "support vector machine"],
            },
            description="Minimal ML methods KG for smoke testing.",
        )

    @staticmethod
    def list_datasets() -> list[str]:
        """Return names of all predefined datasets."""
        return ["nlp_basics", "ml_methods"]

    @staticmethod
    def get(name: str) -> GoldDataset:
        """Get a predefined dataset by name."""
        datasets: dict[str, Callable[[], GoldDataset]] = {
            "nlp_basics": PredefinedSeeds.nlp_basics,
            "ml_methods": PredefinedSeeds.ml_methods,
        }
        if name not in datasets:
            msg = f"Unknown dataset: {name}. Available: {list(datasets.keys())}"
            raise ValueError(msg)
        return datasets[name]()
